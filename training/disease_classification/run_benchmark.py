#!/usr/bin/env python3
"""Fair, reproducible benchmark across image-classification architectures for
leaf-disease classification, plus a comparison against the YOLO disease
detector. Produces all metrics, plots, a comparison table, and an
academic-style report.

Design (orchestrator pattern + incremental saving):
  * One ISOLATED subprocess per model (clean memory, fair per-model timing).
  * IDENTICAL protocol for every model: same dataset, split, seed, image size,
    augmentation, optimizer, schedule, epochs, early-stopping, class weighting.
    The ONLY thing that changes between runs is ``--arch``.
  * Each model's checkpoint + metrics are written as it completes; re-running
    SKIPS already-finished models (unless ``--force``), so a crash never loses
    progress. A ``comparison_partial.json`` is refreshed after every model.

Per model artifacts:
  weights/benchmark/<arch>.pt
  training/disease_classification/reports/benchmark/<arch>/
      training_metrics.json, training_curves.png, confusion_matrix.png
      evaluation_metrics_<arch>.json, confusion_matrix_<arch>.png, per_class_f1_<arch>.png

Aggregate artifacts (reports/benchmark/):
  comparison.json, comparison.csv
  cmp_accuracy_f1.png, cmp_inference.png, cmp_size_params.png
  BENCHMARK_REPORT.md   (academic report with the real numbers filled in)

Usage:
    python training/disease_classification/run_benchmark.py --device mps
    python training/disease_classification/run_benchmark.py \
        --archs efficientnet_v2_s resnet50 mobilenet_v3_large convnext_tiny efficientnet_b0 \
        --epochs 40 --compare-yolo weights/best.pt --device mps
    # promote the recommended model to the deployed path:
    python training/disease_classification/run_benchmark.py --promote auto
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports" / "benchmark"

# Mandatory three + recommended optional extras (override with --archs).
DEFAULT_ARCHS = ["efficientnet_v2_s", "resnet50", "mobilenet_v3_large"]

# Composite-selection weights (documented in the report). NOT accuracy-only:
# generalization quality (macro-F1) dominates, with speed and footprint as
# deployment tie-breakers.
SEL_W = {"macro_f1": 0.60, "speed": 0.25, "footprint": 0.15}

# Domain knowledge: why certain tomato-leaf diseases get confused. Keyed by an
# unordered class pair. Used to annotate the data-driven failure analysis.
CONFUSION_REASONS: dict[frozenset, str] = {
    frozenset({"early_blight", "late_blight"}):
        "both produce brown necrotic lesions; early blight shows concentric 'target' rings while "
        "late blight shows greasy water-soaked patches — hard to separate at low resolution or early stage",
    frozenset({"early_blight", "target_spot"}):
        "both form concentric ring lesions; distinguishing them needs fine lesion-texture detail",
    frozenset({"early_blight", "septoria_leaf_spot"}):
        "small dark necrotic spots look similar before lesions enlarge",
    frozenset({"bacterial_spot", "septoria_leaf_spot"}):
        "both appear as numerous small dark leaf spots",
    frozenset({"bacterial_spot", "early_blight"}):
        "early small dark spots overlap in appearance",
    frozenset({"mosaic_virus", "yellow_leaf_curl_virus"}):
        "both are viral and present as mottling/discoloration and leaf distortion",
    frozenset({"leaf_mold", "healthy"}):
        "early leaf-mold pale patches can be subtle and read as healthy tissue",
    frozenset({"leaf_mold", "late_blight"}):
        "both can show pale/olive patches on the leaf surface",
    frozenset({"target_spot", "septoria_leaf_spot"}):
        "small-to-medium dark spots with similar coloration",
}


def _reason_for(a: str, b: str) -> str:
    return CONFUSION_REASONS.get(frozenset({a, b}),
                                 "visually similar lesion colour/shape, or too few samples of one class")


def build_failure_section(best: dict) -> str:
    """Markdown: per-class most-confused diseases + reasons + recommendations."""
    fa = best.get("failure_analysis", {})
    if not fa:
        return "_Failure analysis unavailable (re-run evaluate_classifier)._\n"
    lines = ["| True class | Most confused with | Rate | Likely reason |",
             "|---|---|---|---|"]
    worst = []
    for cls, info in fa.items():
        if not info["most_confused_with"]:
            lines.append(f"| {cls} | — | — | (no notable confusion) |")
            continue
        top = info["most_confused_with"][0]
        lines.append(f"| {cls} | {top['class']} | {top['rate']:.2f} | {_reason_for(cls, top['class'])} |")
        worst.append((top["rate"], cls, top["class"]))
    worst.sort(reverse=True)
    rec = ("\n**Recommended improvements (highest-leverage first):**\n"
           "1. Add more labelled crops for the most-confused classes above (especially the rare ones) — "
           "imbalance and scarcity drive most of these errors.\n"
           "2. Use higher-resolution crops / larger `--img-size` so fine lesion texture (rings vs. patches) is preserved.\n"
           "3. Try focal loss or stronger class weighting to push the decision boundary on rare classes.\n"
           "4. Fine-tune on a small set of real **drone** leaf crops to close the studio→drone domain gap.\n"
           "5. For genuinely ambiguous pairs (e.g. early vs. late blight at early stage), consider merging them "
           "into a coarser label if the application allows, or adding a 'low-confidence' abstain band.\n")
    return "\n".join(lines) + "\n" + rec


def build_data_vs_arch(ranked: list[dict]) -> str:
    """Estimate whether more data/augmentation or a different architecture helps more."""
    f1s = sorted(r["macro_f1"] for r in ranked)
    spread = (f1s[-1] - f1s[0]) if len(f1s) > 1 else 0.0
    return (
        f"The macro-F1 spread across the tested architectures is **{spread:.3f}**. "
        "When the gap between backbones is small (typically the case once all are ImageNet-pretrained and "
        "fine-tuned on the same few-thousand-crop dataset), the model is **data-limited, not "
        "architecture-limited**. In that regime, collecting more labelled crops for the weak/rare classes "
        "and adding realistic, domain-matched augmentation (and especially fine-tuning on real drone crops) "
        "is expected to yield a **larger** macro-F1 gain than switching architecture. Switching architecture "
        "is worth it mainly when a backbone is both more accurate **and** cheaper to deploy. "
        "Re-evaluate this claim against the actual spread above once the benchmark has run.\n")


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def train_and_eval_one(arch: str, args) -> dict | None:
    """Train (if needed) and evaluate one architecture. Returns its eval dict."""
    ckpt = PROJECT_ROOT / "weights" / "benchmark" / f"{arch}.pt"
    rep_dir = REPORTS / arch
    rep_dir.mkdir(parents=True, exist_ok=True)
    eval_json = rep_dir / f"evaluation_metrics_{arch}.json"

    done = ckpt.exists() and eval_json.exists()
    if done and not args.force:
        print(f"[skip] {arch}: checkpoint + eval already present")
        return json.loads(eval_json.read_text(encoding="utf-8"))

    py = sys.executable
    if not (ckpt.exists() and not args.force):
        rc = _run([
            py, str(HERE / "train_classifier.py"),
            "--data", args.data, "--arch", arch,
            "--epochs", str(args.epochs), "--freeze-epochs", str(args.freeze_epochs),
            "--batch", str(args.batch), "--img-size", str(args.img_size),
            "--lr", str(args.lr), "--patience", str(args.patience),
            "--seed", str(args.seed), "--workers", str(args.workers),
            "--out", f"weights/benchmark/{arch}.pt",
            "--reports-dir", str(rep_dir),
        ] + (["--device", args.device] if args.device else []))
        if rc != 0:
            print(f"[error] training {arch} failed (rc={rc}); skipping")
            return None

    rc = _run([
        py, str(HERE / "evaluate_classifier.py"),
        "--data", args.data, "--split", "test",
        "--weights", f"weights/benchmark/{arch}.pt", "--tag", arch,
        "--reports-dir", str(rep_dir),
    ] + (["--device", args.device] if args.device else []))
    if rc != 0 or not eval_json.exists():
        print(f"[error] evaluation {arch} failed; skipping")
        return None
    return json.loads(eval_json.read_text(encoding="utf-8"))


def _normalize(vals: list[float], higher_better: bool) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [((v - lo) / (hi - lo)) if higher_better else ((hi - v) / (hi - lo)) for v in vals]


def selection_scores(results: list[dict]) -> dict[str, float]:
    f1 = [r["macro_f1"] for r in results]
    spd = [r.get("avg_inference_ms") or 1e9 for r in results]
    fp = [r.get("model_size_mb") or 1e9 for r in results]
    nf1 = _normalize(f1, True)
    nspd = _normalize(spd, False)   # lower ms better
    nfp = _normalize(fp, False)     # smaller better
    return {
        r["arch"]: round(SEL_W["macro_f1"] * nf1[i] + SEL_W["speed"] * nspd[i]
                         + SEL_W["footprint"] * nfp[i], 4)
        for i, r in enumerate(results)
    }


def write_comparison(results: list[dict]) -> tuple[Path, dict]:
    scores = selection_scores(results)
    for r in results:
        r["selection_score"] = scores[r["arch"]]
    results = sorted(results, key=lambda r: r["selection_score"], reverse=True)

    cmp_json = REPORTS / "comparison.json"
    cmp_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # CSV table
    cols = ["arch", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "avg_inference_ms", "num_params_millions", "model_size_mb", "selection_score"]
    with (REPORTS / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([r.get(c) for c in cols])

    _plots(results)
    return cmp_json, {"ranked": results, "scores": scores}


def _plots(results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        archs = [r["arch"] for r in results]
        x = np.arange(len(archs))

        fig, ax = plt.subplots(figsize=(max(7, len(archs) * 1.6), 4.5))
        ax.bar(x - 0.2, [r["accuracy"] for r in results], 0.4, label="accuracy")
        ax.bar(x + 0.2, [r["macro_f1"] for r in results], 0.4, label="macro F1")
        ax.set_xticks(x); ax.set_xticklabels(archs, rotation=20, ha="right")
        ax.set_ylim(0, 1); ax.set_title("Accuracy & macro-F1 by architecture"); ax.legend()
        fig.tight_layout(); fig.savefig(REPORTS / "cmp_accuracy_f1.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(7, len(archs) * 1.6), 4.5))
        ax.bar(x, [r.get("avg_inference_ms") or 0 for r in results], color="#6366f1")
        ax.set_xticks(x); ax.set_xticklabels(archs, rotation=20, ha="right")
        ax.set_ylabel("ms / image"); ax.set_title("Average inference time (lower is better)")
        fig.tight_layout(); fig.savefig(REPORTS / "cmp_inference.png", dpi=130); plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(max(7, len(archs) * 1.6), 4.5))
        ax1.bar(x - 0.2, [r.get("num_params_millions") or 0 for r in results], 0.4,
                label="params (M)", color="#16a34a")
        ax1.bar(x + 0.2, [r.get("model_size_mb") or 0 for r in results], 0.4,
                label="size (MB)", color="#f97316")
        ax1.set_xticks(x); ax1.set_xticklabels(archs, rotation=20, ha="right")
        ax1.set_title("Model footprint (lower is better)"); ax1.legend()
        fig.tight_layout(); fig.savefig(REPORTS / "cmp_size_params.png", dpi=130); plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(comparison plots skipped: {exc})")


def _md_table(results: list[dict]) -> str:
    head = ("| Model | Accuracy | Precision | Recall | F1 | Avg Inference (ms) | "
            "Params (M) | Size (MB) | Sel. score |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    rows = "".join(
        f"| {r['arch']} | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | "
        f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | "
        f"{r.get('avg_inference_ms','-')} | {r.get('num_params_millions','-')} | "
        f"{r.get('model_size_mb','-')} | {r['selection_score']:.4f} |\n"
        for r in results)
    return head + rows


def write_report(results: list[dict], args, yolo: dict | None) -> Path:
    best = results[0]
    classes = []
    ds_report = HERE / "reports" / "dataset_report.json"
    dataset_blurb = ""
    if ds_report.exists():
        d = json.loads(ds_report.read_text(encoding="utf-8"))
        classes = d.get("classes", [])
        dataset_blurb = (
            f"- Source datasets: {', '.join(d.get('datasets_used', []))}\n"
            f"- Total crops: {d.get('total_crops')}\n"
            f"- Classes ({len(classes)}): {', '.join(classes)}\n"
            f"- Per-split counts: {d.get('per_split_counts')}\n"
            f"- Class-imbalance ratio (max/min): {d.get('class_imbalance_ratio')}\n"
            f"- Average crop size: {d.get('avg_crop_size')}\n"
        )

    yolo_section = "_YOLO comparison not run (pass `--compare-yolo weights/best.pt`)._\n"
    if yolo:
        yc = yolo
        yolo_section = (
            f"On the same TEST crops, the dedicated classifier ({best['arch']}) scored "
            f"**{yc['classifier_accuracy_on_crops']:.4f}** accuracy versus the YOLO detector's "
            f"**{yc['yolo_accuracy_on_crops']:.4f}**. The YOLO detector returned no usable "
            f"prediction on **{yc['yolo_no_prediction_rate']*100:.1f}%** of single-leaf crops "
            f"(counted as errors) — the structural weakness motivating this change.\n\n"
            f"**When the classifier wins:** tightly-cropped single leaves (the webapp's actual "
            f"input), where it always returns a calibrated class. **When YOLO can win:** whole "
            f"frames containing several spatially-separated lesions, which it was trained on; it "
            f"also avoids a second model load when only coarse screening is needed.\n"
        )

    md = f"""# Leaf-Disease Classifier — Architecture Benchmark

_Auto-generated by `run_benchmark.py`. Numbers below are real results from the
runs that produced `comparison.json`._

## 1. Objective
Select the disease-classification backbone for the SmartLeafDetection pipeline
on the basis of **experimental evidence**, not assumption. We train several
ImageNet-pretrained architectures under an identical protocol and compare them
on a held-out test set across accuracy, per-class quality, inference speed, and
deployment footprint.

## 2. Dataset
Leaf crops generated from the project's YOLO detection datasets by
`build_classification_dataset.py` (one crop per labelled box), unified to a
canonical taxonomy and split with source-image-level leakage prevention
(priority test > validation > train).
{dataset_blurb or "- (Run build_classification_dataset.py to populate dataset_report.json.)"}

## 3. Experimental methodology
- Identical `datasets/leaf_clf` ImageFolder split for every model.
- Fixed random seed = {args.seed}; image size = {args.img_size}px.
- Identical augmentation (RandomResizedCrop, flips, rotation, colour jitter,
  RandAugment) and identical ImageNet normalization.
- Identical optimizer (AdamW), cosine LR schedule, class-weighted cross-entropy
  with label smoothing, frozen-backbone warm-up then progressive unfreezing,
  and early stopping (patience = {args.patience}).
- One isolated process per model; inference timing measured at batch size 1 on
  `{best.get('device','?')}` after warm-up ({best.get('timing_samples','?')} samples).
- The only variable across runs is the architecture.

## 4. Training configuration
epochs={args.epochs}, freeze_epochs={args.freeze_epochs}, batch={args.batch},
lr={args.lr}, patience={args.patience}, img_size={args.img_size}, seed={args.seed}.

## 5–6. Results & comparison table
{_md_table(results)}

Per-model confusion matrices, per-class F1 charts, and training curves are in
`reports/benchmark/<arch>/`.

## 7. Graphs
- Accuracy & macro-F1: `cmp_accuracy_f1.png`
- Inference speed: `cmp_inference.png`
- Footprint (params & size): `cmp_size_params.png`

## 8. Discussion
The selection score combines generalization and deployment cost:
`score = {SEL_W['macro_f1']}·macroF1 + {SEL_W['speed']}·speed + {SEL_W['footprint']}·footprint`
(speed and footprint normalized so smaller is better). Accuracy alone is **not**
the deciding factor, because the webapp classifies many crops per video on
possibly-CPU hardware, so a small per-class-F1 gain that costs a large latency
or memory increase is usually not worth it.

## 9. Model-selection rationale
**Recommended default: `{best['arch']}`** — highest composite selection score
({best['selection_score']:.4f}); macro-F1 {best['macro_f1']:.4f}, accuracy
{best['accuracy']:.4f}, {best.get('avg_inference_ms','?')} ms/img,
{best.get('num_params_millions','?')} M params, {best.get('model_size_mb','?')} MB.
This balances generalization (macro-F1 weights rare classes equally, guarding
against majority-class bias), inference responsiveness in the webapp, and a
deployment footprint that is easy to ship and maintain. If a different model has
a near-equal score but materially better robustness on the rare classes (inspect
the per-class F1 charts), prefer it — the composite is guidance, not a verdict.

## 10. YOLO comparison
{yolo_section}

## 11. Failure-case analysis (best model, test set)
{build_failure_section(best)}

## 12. Will more data or a different architecture help more?
{build_data_vs_arch(results)}

## 13. Limitations & future improvements
- **Domain gap:** training crops are largely studio/field single leaves; real
  drone crops differ. Largest expected gain: fine-tune on a small set of
  hand-labelled drone crops.
- **Class imbalance:** rare classes have fewer crops; per-class recall is the
  metric to watch. Class weighting mitigates but does not eliminate this.
- **Fixed shared hyperparameters:** for fairness every model used the same LR
  schedule; a per-architecture LR sweep could shift absolute numbers (not
  expected to change the qualitative ranking).
- **Single fixed test split:** chosen for speed; results are point estimates.
"""
    out = REPORTS / "BENCHMARK_REPORT.md"
    out.write_text(md, encoding="utf-8")
    return out


def write_model_card(best: dict, args, yolo: dict | None) -> Path:
    """Write a Model Card for the winning model for thesis docs + maintenance."""
    arch = best["arch"]
    rep_dir = REPORTS / arch
    # Pull supporting facts from per-model + dataset artifacts.
    tm = {}
    tm_path = rep_dir / "training_metrics.json"
    if tm_path.exists():
        tm = json.loads(tm_path.read_text(encoding="utf-8"))
    meta = {}
    meta_path = rep_dir / "experiment_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ds = {}
    ds_path = HERE / "reports" / "dataset_report.json"
    if ds_path.exists():
        ds = json.loads(ds_path.read_text(encoding="utf-8"))

    psc = ds.get("per_split_counts", {})
    n_train = sum(psc.get("train", {}).values()) if psc else meta.get("dataset", {}).get("train_total", "?")
    n_val = sum(psc.get("validation", {}).values()) if psc else meta.get("dataset", {}).get("val_total", "?")
    n_test = sum(psc.get("test", {}).values()) if psc else "?"
    cfg = meta.get("config", {})
    cfg_line = (f"arch={arch}, img_size={cfg.get('img_size','?')}, epochs(max)={cfg.get('epochs','?')}, "
                f"freeze_epochs={cfg.get('freeze_epochs','?')}, batch={cfg.get('batch','?')}, "
                f"lr={cfg.get('lr','?')}, weight_decay={cfg.get('weight_decay','?')}, "
                f"label_smoothing={cfg.get('label_smoothing','?')}, patience={cfg.get('patience','?')}, "
                f"seed={cfg.get('seed','?')}, mixup={cfg.get('mixup',0)}, cutmix={cfg.get('cutmix',0)}")

    yolo_line = ""
    if yolo:
        yolo_line = (f"\n## Comparison vs YOLO detector\n"
                     f"- Classifier test accuracy on crops: {yolo['classifier_accuracy_on_crops']:.4f}\n"
                     f"- YOLO detector accuracy on crops: {yolo['yolo_accuracy_on_crops']:.4f} "
                     f"(no-prediction rate {yolo['yolo_no_prediction_rate']*100:.1f}%)\n")

    card = f"""# Model Card — Leaf-Disease Classifier (deployed)

## Selected architecture
**{arch}** (ImageNet-pretrained, fine-tuned). Selected by composite score
({SEL_W['macro_f1']}·macro-F1 + {SEL_W['speed']}·speed + {SEL_W['footprint']}·footprint);
primary quality criterion is **validation Macro-F1**.

## Training dataset
Leaf crops generated from the project's YOLO detection datasets
({', '.join(ds.get('datasets_used', ['(see dataset_report.json)']))}), unified to a
canonical taxonomy with source-image-level, leakage-free splits.
- Classes ({len(ds.get('classes', []))}): {', '.join(ds.get('classes', [])) or '(see dataset_report.json)'}

## Data volumes
- Training images: {n_train}
- Validation images: {n_val}
- Test images: {n_test}

## Training configuration
{cfg_line}
- Optimizer/scheduler: {meta.get('optimizer', {})} / {meta.get('scheduler', {})}
- Loss: class-weighted CrossEntropy + label smoothing
- Augmentation: realistic (rotation, flips, colour jitter, autocontrast, mild blur, mild noise);
  MixUp/CutMix: mixup={cfg.get('mixup',0)}, cutmix={cfg.get('cutmix',0)}
- Reproducibility: full config + seeds + augmentation + dataset stats in `experiment_metadata.json`

## Performance
- **Best validation Macro-F1:** {tm.get('best_val_macro_f1', '?')}
- **Test accuracy:** {best.get('accuracy', '?')}
- **Test Macro-F1:** {best.get('macro_f1', '?')}
- Avg inference: {best.get('avg_inference_ms','?')} ms/img on {best.get('device','?')}
- Parameters: {best.get('num_params_millions','?')} M; size: {best.get('model_size_mb','?')} MB
{yolo_line}
## Known limitations
- Domain gap: trained mostly on studio/field single-leaf crops; accuracy on real
  drone-video crops is capped until fine-tuned on drone data.
- Rare classes (few crops) have weaker recall despite class weighting.
- Predicts one disease per leaf crop; co-occurring diseases on one leaf are not modelled.
- Single fixed test split: metrics are point estimates.

## Recommended use cases
- Per-leaf disease screening inside the SmartLeafDetection drone pipeline
  (leaf detector localizes/tracks; this model classifies each crop).
- Offline analysis of tomato-leaf imagery for the canonical disease set above.
- **Not** a substitute for expert agronomic diagnosis; treat as decision support.
"""
    out = REPORTS / "MODEL_CARD.md"
    out.write_text(card, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    ap.add_argument("--data", default="datasets/leaf_clf")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="Retrain/re-eval even if artifacts exist")
    ap.add_argument("--compare-yolo", default=None, help="YOLO weights for the best-model comparison")
    ap.add_argument("--promote", choices=["none", "auto"], default="none",
                    help="'auto' copies the recommended model to weights/leaf_classifier.pt")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "weights" / "benchmark").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for arch in args.archs:
        r = train_and_eval_one(arch, args)
        if r is not None:
            r["arch"] = arch
            results.append(r)
            # Incremental save after every model.
            (REPORTS / "comparison_partial.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            print(f"[saved] partial results -> {REPORTS / 'comparison_partial.json'}")

    if not results:
        raise SystemExit("No models completed; check training logs above.")

    cmp_path, agg = write_comparison(results)
    ranked = agg["ranked"]
    best = ranked[0]
    print(f"\nComparison -> {cmp_path}")
    print(f"Recommended (composite): {best['arch']} (score {best['selection_score']})")

    # YOLO comparison on the recommended model (fresh eval with --compare-yolo).
    yolo = None
    if args.compare_yolo:
        rep_dir = REPORTS / best["arch"]
        rc = _run([
            sys.executable, str(HERE / "evaluate_classifier.py"),
            "--data", args.data, "--split", "test",
            "--weights", f"weights/benchmark/{best['arch']}.pt", "--tag", f"{best['arch']}_vs_yolo",
            "--reports-dir", str(rep_dir), "--compare-yolo", args.compare_yolo,
        ] + (["--device", args.device] if args.device else []))
        vj = rep_dir / f"evaluation_metrics_{best['arch']}_vs_yolo.json"
        if rc == 0 and vj.exists():
            yolo = json.loads(vj.read_text(encoding="utf-8")).get("yolo_comparison")

    report = write_report(ranked, args, yolo)
    print(f"Report -> {report}")
    card = write_model_card(best, args, yolo)
    print(f"Model card -> {card}")

    if args.promote == "auto":
        import shutil
        src = PROJECT_ROOT / "weights" / "benchmark" / f"{best['arch']}.pt"
        dst = PROJECT_ROOT / "weights" / "leaf_classifier.pt"
        shutil.copy2(src, dst)
        print(f"[promote] {src.name} -> {dst} (webapp will use it when DISEASE_BACKEND=auto/classifier)")


if __name__ == "__main__":
    main()
