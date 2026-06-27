#!/usr/bin/env python3
"""Train a dedicated leaf-disease image classifier, optimized for **validation
Macro-F1** (the primary criterion for a final-year project where per-class
quality matters more than majority-class accuracy).

Default architecture is **ResNet50** (the project's selected production model);
others are selectable via ``--arch`` (see ``model_factory``) so the benchmark is
a fair like-for-like.

Design choices for maximum, *realistic* predictive performance:
- **Best checkpoint = highest validation Macro-F1** (not accuracy).
- **Strong but realistic augmentation** suited to leaf disease, whose symptoms
  are *local* patterns (spots, lesions, discoloration): random rotation, H/V
  flips (valid for top-down leaves), brightness/contrast + colour jitter,
  small scale variation, mild Gaussian blur, mild Gaussian noise, mild
  autocontrast. **MixUp/CutMix are OFF by default** because mixing whole leaves
  can fabricate unrealistic symptom layouts; enable them only via ``--mixup`` /
  ``--cutmix`` to test whether they actually raise validation Macro-F1.
- **Class-weighted** loss (inverse frequency) + label smoothing for imbalance.
- Frozen-backbone warm-up -> progressive unfreezing, cosine LR, **early
  stopping on val Macro-F1** (training runs until it triggers naturally).
- **Full reproducibility**: seeds, augmentation config, optimizer/scheduler
  settings, architecture/hyper-parameters, and per-class/per-split dataset
  statistics are all written to ``experiment_metadata.json``.

Artifacts (to ``weights/`` and the reports dir):
    weights/leaf_classifier.pt   (checkpoint with config + classes + normalization)
    weights/classes.json
    <reports>/training_metrics.json   (per-epoch train + val metrics)
    <reports>/experiment_metadata.json
    <reports>/training_curves.png, confusion_matrix.png

Usage:
    python training/disease_classification/train_classifier.py --device mps
    python training/disease_classification/train_classifier.py --arch resnet50 --mixup 0.2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "disease_classification"))
from model_factory import SUPPORTED_ARCHS, build_model  # noqa: E402
from smart_leaf_detection.device_utils import resolve_torch_device  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def seed_worker(_worker_id: int) -> None:
    ws = torch.initial_seed() % (2 ** 32)
    np.random.seed(ws)
    random.seed(ws)


# --------------------------------------------------------------------------
# Realistic augmentation
# --------------------------------------------------------------------------
class AddGaussianNoise:
    """Add mild zero-mean Gaussian noise to a [0,1] tensor (pre-normalization)."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return t
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


def make_transforms(img: int, blur_p: float, noise_p: float, noise_std: float):
    """Build (train_tf, eval_tf, aug_config). Realistic, symptom-preserving augs."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.18)),  # small scale var
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),                 # valid for top-down leaves
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.25, 0.25, 0.20, 0.05),     # brightness/contrast/saturation/hue
        transforms.RandomAutocontrast(p=0.2),               # mild lighting variation
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=blur_p),  # mild blur
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(noise_std)], p=noise_p),  # mild noise
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    aug_config = {
        "random_resized_crop": {"size": img, "scale": [0.7, 1.0], "ratio": [0.85, 1.18]},
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.5,
        "rotation_degrees": 25,
        "color_jitter": {"brightness": 0.25, "contrast": 0.25, "saturation": 0.20, "hue": 0.05},
        "random_autocontrast_p": 0.2,
        "gaussian_blur": {"p": blur_p, "kernel": 3, "sigma": [0.1, 1.5]},
        "gaussian_noise": {"p": noise_p, "std": noise_std},
        "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "mixup_cutmix": "off by default (see --mixup/--cutmix); local symptoms can be "
                        "distorted by image mixing, so enable only if it raises val Macro-F1",
        "eval": "Resize(1.15x) -> CenterCrop -> Normalize (no augmentation)",
    }
    return train_tf, eval_tf, aug_config


# --------------------------------------------------------------------------
# Optional MixUp / CutMix (disabled unless alpha > 0)
# --------------------------------------------------------------------------
def _rand_bbox(h: int, w: int, lam: float):
    cut = np.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h)
    x2, y2 = np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def mix_batch(x, y, mixup_alpha, cutmix_alpha):
    """Return (x, y_a, y_b, lam, applied). applied=False means no mixing this batch."""
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    use_mix = mixup_alpha > 0 and not use_cut
    if not (use_cut or use_mix):
        return x, y, y, 1.0, False
    idx = torch.randperm(x.size(0), device=x.device)
    if use_mix:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1 - lam) * x[idx]
    else:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        x1, y1, x2, y2 = _rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    return x, y, y[idx], lam, True


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def macro_f1_from_conf(conf: np.ndarray) -> float:
    tp = np.diag(conf).astype(np.float64)
    pred_tot = conf.sum(0).astype(np.float64)
    true_tot = conf.sum(1).astype(np.float64)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def set_backbone_trainable(model: nn.Module, head_names: list[str], trainable: bool) -> None:
    head_params = set()
    for name in head_names:
        head_params.update(id(p) for p in getattr(model, name).parameters())
    for p in model.parameters():
        p.requires_grad = trainable if id(p) not in head_params else True


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1_from_conf(conf), conf


def class_counts(samples, classes) -> dict[str, int]:
    c = np.bincount([y for _, y in samples], minlength=len(classes))
    return {classes[i]: int(c[i]) for i in range(len(classes))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/leaf_clf")
    ap.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs; early stopping ends it naturally")
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12, help="Early-stopping patience on val Macro-F1")
    # Realistic-augmentation knobs
    ap.add_argument("--blur-prob", type=float, default=0.2)
    ap.add_argument("--noise-prob", type=float, default=0.2)
    ap.add_argument("--noise-std", type=float, default=0.02)
    # Optional aggressive mixing (OFF by default)
    ap.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 = disabled)")
    ap.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 = disabled)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="weights/leaf_classifier.pt")
    ap.add_argument("--reports-dir", default=None,
                    help="Where to write metrics/plots (default: training/disease_classification/reports). "
                         "The benchmark orchestrator sets a per-architecture directory.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(resolve_torch_device(args.device))
    data_root = (PROJECT_ROOT / args.data).resolve()
    reports_dir = (Path(args.reports_dir).resolve() if args.reports_dir
                   else PROJECT_ROOT / "training" / "disease_classification" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir = data_root / "train", data_root / "validation"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Expected {train_dir} and {val_dir}. Run build_classification_dataset.py first.")

    train_tf, eval_tf, aug_config = make_transforms(
        args.img_size, args.blur_prob, args.noise_prob, args.noise_std)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    classes = train_ds.classes
    if val_ds.classes != classes:
        raise SystemExit(f"Class mismatch train{classes} vs val{val_ds.classes}")
    num_classes = len(classes)
    print(f"Arch={args.arch} device={device} classes={classes}")

    gen = torch.Generator(); gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"),
                              worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Inverse-frequency class weights to counter imbalance.
    counts = np.bincount([y for _, y in train_ds.samples], minlength=num_classes).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1)) / num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model, head_names = build_model(args.arch, num_classes, pretrained=True)
    model.to(device)
    set_backbone_trainable(model, head_names, trainable=False)  # warm-up: head only

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mixing = args.mixup > 0 or args.cutmix > 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_f1, best_acc, best_conf, best_state, best_epoch, bad = -1.0, 0.0, None, None, -1, 0
    unfrozen = False

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and not unfrozen:
            set_backbone_trainable(model, head_names, trainable=True)  # progressive unfreeze
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.freeze_epochs))
            unfrozen = True
            print(f"[epoch {epoch}] unfroze backbone")

        model.train()
        run_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if mixing:
                x, y_a, y_b, lam, applied = mix_batch(x, y, args.mixup, args.cutmix)
                out = model(x)
                loss = (lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)) if applied \
                    else criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)
            loss.backward(); optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
        scheduler.step()

        tr_loss, tr_acc = run_loss / total, correct / total
        val_loss, val_acc, val_f1, conf = evaluate(model, val_loader, device, num_classes)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1); history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"epoch {epoch+1:3d}/{args.epochs} | train {tr_loss:.3f}/{tr_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f} macroF1 {val_f1:.3f}")

        # Best checkpoint = highest validation Macro-F1.
        if val_f1 > best_f1:
            best_f1, best_acc, best_conf, best_epoch, bad = val_f1, val_acc, conf, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(best val Macro-F1 {best_f1:.3f} @ epoch {best_epoch+1})")
                break

    # ---- Save best checkpoint + reproducibility metadata ----
    weights_path = PROJECT_ROOT / args.out
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    aug_config["mixup_alpha"] = args.mixup
    aug_config["cutmix_alpha"] = args.cutmix
    config = vars(args).copy()
    optimizer_cfg = {"type": "AdamW", "lr_head_warmup": args.lr, "lr_after_unfreeze": args.lr * 0.1,
                     "weight_decay": args.weight_decay}
    scheduler_cfg = {"type": "CosineAnnealingLR", "T_max_phase1": args.epochs,
                     "T_max_phase2": max(1, args.epochs - args.freeze_epochs)}
    dataset_stats = {
        "root": str(data_root), "classes": classes,
        "train_total": len(train_ds), "val_total": len(val_ds),
        "train_per_class": class_counts(train_ds.samples, classes),
        "val_per_class": class_counts(val_ds.samples, classes),
    }

    ckpt = {
        "arch": args.arch, "classes": classes, "img_size": args.img_size,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "state_dict": best_state,
        "val_macro_f1": best_f1, "val_accuracy": best_acc, "best_epoch": best_epoch + 1,
        "seed": args.seed, "config": config, "augmentation": aug_config,
    }
    torch.save(ckpt, weights_path)
    (PROJECT_ROOT / "weights" / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    (reports_dir / "training_metrics.json").write_text(json.dumps({
        "arch": args.arch, "classes": classes,
        "best_val_macro_f1": best_f1, "best_val_accuracy": best_acc, "best_epoch": best_epoch + 1,
        "selection_criterion": "validation_macro_f1",
        "history": history,
    }, indent=2), encoding="utf-8")

    (reports_dir / "experiment_metadata.json").write_text(json.dumps({
        "experiment": {"arch": args.arch, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                       "selection_criterion": "validation_macro_f1"},
        "seed": args.seed, "deterministic": "best_effort (cudnn deterministic; MPS not fully guaranteed)",
        "config": config,
        "augmentation": aug_config,
        "optimizer": optimizer_cfg,
        "scheduler": scheduler_cfg,
        "loss": {"type": "CrossEntropyLoss", "label_smoothing": args.label_smoothing,
                 "class_weighted": True, "class_weights": {c: round(float(w), 4) for c, w in zip(classes, weights)}},
        "model": {"arch": args.arch, "img_size": args.img_size, "num_classes": num_classes,
                  "pretrained": True, "head_modules": head_names},
        "dataset": dataset_stats,
        "result": {"best_val_macro_f1": best_f1, "best_val_accuracy": best_acc, "best_epoch": best_epoch + 1},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved best model ({args.arch}, val Macro-F1 {best_f1:.3f}) -> {weights_path}")

    # ---- Plots (best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = range(1, len(history["train_loss"]) + 1)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
        a1.plot(ep, history["train_loss"], label="train"); a1.plot(ep, history["val_loss"], label="val")
        a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
        a2.plot(ep, history["train_acc"], label="train"); a2.plot(ep, history["val_acc"], label="val")
        a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
        a3.plot(ep, history["val_macro_f1"], label="val", color="#16a34a")
        a3.axvline(best_epoch + 1, ls="--", color="gray"); a3.set_title("Validation Macro-F1")
        a3.set_xlabel("epoch"); a3.legend()
        fig.tight_layout(); fig.savefig(reports_dir / "training_curves.png", dpi=130); plt.close(fig)

        if best_conf is not None:
            cm = best_conf.astype(np.float64)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes * 0.9)))
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"Confusion matrix (val best, {args.arch})")
            fig.colorbar(im); fig.tight_layout()
            fig.savefig(reports_dir / "confusion_matrix.png", dpi=130); plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(plots skipped: {exc})")


if __name__ == "__main__":
    main()
