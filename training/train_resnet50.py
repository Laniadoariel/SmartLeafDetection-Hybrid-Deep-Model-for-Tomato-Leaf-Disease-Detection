"""Fine-tune ResNet50 on PlantVillage dataset for tomato disease classification.

Downloads the PlantVillage tomato subset and fine-tunes a pretrained ResNet50
to classify 10 tomato leaf disease classes.

The 10 classes (matching PipelineConfig defaults):
  Bacterial_spot, Early_blight, Late_blight, Leaf_Mold,
  Septoria_leaf_spot, Spider_mites, Target_Spot,
  Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus, healthy

Dataset structure expected:
  training/plantvillage/
    Bacterial_spot/
      img001.jpg
      ...
    Early_blight/
      ...
    healthy/
      ...

Usage:
    # Download PlantVillage tomato images first (see --download flag)
    python training/train_resnet50.py --download --data-dir training/plantvillage

    # Or if you already have the data:
    python training/train_resnet50.py --data-dir training/plantvillage --epochs 25
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# The 10 tomato disease classes matching PipelineConfig.class_names
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
]


def download_plantvillage(data_dir: Path) -> None:
    """Download PlantVillage tomato subset from Kaggle or a mirror.

    This creates the expected directory structure with one folder per class.
    Requires the 'kaggle' package and API credentials, or falls back to
    instructions for manual download.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
    if len(existing) >= 5:
        print(f"Dataset already exists at {data_dir} with {len(existing)} classes. Skipping download.")
        return

    print("Attempting to download PlantVillage dataset...")
    try:
        import kaggle  # type: ignore[import-untyped]
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "emmarex/plantdisease",
            path=str(data_dir.parent),
            unzip=True,
        )
        # The dataset extracts with a nested structure — reorganize
        _reorganize_plantvillage(data_dir)
        print(f"Download complete! Data at: {data_dir}")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("  2. Download and extract the dataset")
        print(f"  3. Copy the Tomato___* folders to: {data_dir}/")
        print("  4. Rename folders to match class names (remove 'Tomato___' prefix)")
        print(f"\nExpected structure:")
        for name in CLASS_NAMES:
            print(f"  {data_dir}/{name}/")


def _reorganize_plantvillage(data_dir: Path) -> None:
    """Reorganize downloaded PlantVillage data into clean class folders.

    The Kaggle dataset uses names like 'Tomato___Bacterial_spot' — we
    strip the 'Tomato___' prefix to match our CLASS_NAMES.
    """
    # Map from Kaggle folder names to our class names
    kaggle_to_class = {
        "Tomato___Bacterial_spot": "Bacterial_spot",
        "Tomato___Early_blight": "Early_blight",
        "Tomato___Late_blight": "Late_blight",
        "Tomato___Leaf_Mold": "Leaf_Mold",
        "Tomato___Septoria_leaf_spot": "Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Spider_mites",
        "Tomato___Target_Spot": "Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato_mosaic_virus",
        "Tomato___healthy": "healthy",
    }

    # Search for Kaggle-style folders in parent directory
    search_dirs = [data_dir.parent, data_dir.parent / "PlantVillage"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for kaggle_name, class_name in kaggle_to_class.items():
            src = search_dir / kaggle_name
            if src.exists():
                dst = data_dir / class_name
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                    print(f"  Moved {kaggle_name} -> {class_name}")


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms.

    Training uses augmentation; validation uses only resize + normalize.
    Both use ImageNet normalization to match the pretrained ResNet50.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Build ResNet50 with custom classification head.

    Args:
        num_classes: Number of output classes (10 for tomato diseases).
        freeze_backbone: If True, freeze all layers except the final FC.
            Recommended for small datasets / initial fine-tuning.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on PlantVillage")
    parser.add_argument("--data-dir", default="training/plantvillage", help="Dataset directory")
    parser.add_argument("--download", action="store_true", help="Download PlantVillage dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze ResNet backbone")
    parser.add_argument("--unfreeze-epoch", type=int, default=10, help="Epoch to unfreeze backbone")
    parser.add_argument("--device", default="", help="Device: '' for auto, 'cpu', 'cuda'")
    parser.add_argument("--output", default="resnet50_tomato.pt", help="Output weights path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Download if requested
    if args.download:
        download_plantvillage(data_dir)

    if not data_dir.exists():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        print("Use --download to fetch the PlantVillage dataset, or provide --data-dir.")
        return

    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loading
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)

    # Verify classes match expected
    found_classes = full_dataset.classes
    print(f"Found {len(found_classes)} classes: {found_classes}")

    # Train/val split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Override val transform (random_split doesn't change transforms)
    val_dataset.dataset = datasets.ImageFolder(str(data_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # Model
    num_classes = len(found_classes)
    model = build_model(num_classes, freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Unfreeze backbone at specified epoch
        if epoch == args.unfreeze_epoch and args.freeze_backbone:
            print(f"\n--- Unfreezing backbone at epoch {epoch} ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to: {args.output}")


if __name__ == "__main__":
    main()
