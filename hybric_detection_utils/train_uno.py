#!/usr/bin/env python3
"""
UNO Card Symbol CNN — Training & Inference
==========================================

Trains a CNN to classify the 15 UNO symbols from 80x80 RGB corner crops:
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +2, +4, wild, skip, reverse

The model uses ~4.8M parameters (well under the 12M limit). Only images
marked usable=1 in labels.csv are used for training.

Usage:
    # Train
    python train_uno.py train --image_dir ./cards --labels labels.csv

    # Run on a single image
    python train_uno.py predict --model uno_symbol_cnn.pt --image card.png

    # Run on every image in a folder (writes predictions.csv)
    python train_uno.py predict --model uno_symbol_cnn.pt --image_dir ./new_cards

Dependencies:
    pip install torch torchvision pillow numpy
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+2",
    "+4",
    "wild",
    "skip",
    "reverse",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# ImageNet normalization — a sensible default for RGB photos.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ============================================================
# Model
# ============================================================


class UnoSymbolCNN(nn.Module):
    """
    Four-block VGG-style CNN with batch norm and global average pooling.
    Input:  80x80 RGB
    Output: logits over 15 classes
    Params: ~4.8M  (well under the 12M cap)
    """

    def __init__(self, num_classes=15, dropout=0.4):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 64),  # 80 -> 40
            block(64, 128),  # 40 -> 20
            block(128, 256),  # 20 -> 10
            block(256, 512),  # 10 -> 5
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Data
# ============================================================


class UnoDataset(Dataset):
    def __init__(self, image_dir, filenames_with_labels, transform=None):
        self.image_dir = Path(image_dir)
        self.samples = filenames_with_labels  # list of (filename, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img = Image.open(self.image_dir / filename).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_usable_labels(labels_csv):
    """Return {label: [filename, ...]} for usable rows only."""
    by_class = defaultdict(list)
    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["usable"] != "1":
                continue
            if row["label"] not in CLASS_TO_IDX:
                continue
            by_class[row["label"]].append(row["filename"])
    return by_class


def stratified_split(by_class, val_frac, rng):
    """Split per-class so each class is represented in both train and val."""
    train, val = [], []
    for cls, files in by_class.items():
        files = files.copy()
        rng.shuffle(files)
        idx = CLASS_TO_IDX[cls]
        if len(files) <= 1:
            train.extend((f, idx) for f in files)
            continue
        n_val = max(1, int(round(len(files) * val_frac)))
        n_val = min(n_val, len(files) - 1)  # keep at least 1 in train
        val.extend((f, idx) for f in files[:n_val])
        train.extend((f, idx) for f in files[n_val:])
    return train, val


def build_transforms():
    """
    Augmentations chosen for UNO symbols:
      - Small affine (rotation/translate/scale) — cards aren't perfectly aligned.
      - Color jitter — symbol detection should be color-invariant.
      - NO horizontal/vertical flip — would confuse skip/reverse arrows
        and could turn 6 into 9 (the underline is the only difference).
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.92, 1.08)
            ),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.3, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    return train_tf, val_tf


# ============================================================
# Train / Eval loops
# ============================================================


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_main(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    by_class = load_usable_labels(args.labels)
    total_usable = sum(len(v) for v in by_class.values())
    if total_usable == 0:
        print(
            "No usable labeled images found. Label some first with label_uno.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Total usable images: {total_usable}")
    print("Class distribution:")
    for cls in CLASSES:
        print(f"  {cls:>8s}: {len(by_class.get(cls, []))}")

    train_samples, val_samples = stratified_split(by_class, args.val_split, rng)
    print(f"\nTrain: {len(train_samples)}  |  Val: {len(val_samples)}")

    train_tf, val_tf = build_transforms()
    train_ds = UnoDataset(args.image_dir, train_samples, train_tf)
    val_ds = UnoDataset(args.image_dir, val_samples, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = UnoSymbolCNN(num_classes=len(CLASSES), dropout=args.dropout).to(device)
    n = count_params(model)
    print(f"\nModel parameters: {n:,}  ({n / 1e6:.2f}M)")
    assert n < 12_000_000, f"Param count {n} exceeds 12M limit."

    # Class weights to counteract imbalance.
    counts = np.array([len(by_class.get(c, [])) for c in CLASSES], dtype=np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    patience_left = args.patience
    print()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improved = va_acc > best_val_acc
        marker = "  *new best*" if improved else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}{marker}"
        )

        if improved:
            best_val_acc = va_acc
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "val_acc": va_acc,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                print(f"\nEarly stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")


# ============================================================
# Inference
# ============================================================


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = UnoSymbolCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes


def predict_image(model, classes, image_path, device, tf):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax().item())
    return classes[idx], float(probs[idx].item()), probs.cpu().numpy()


def predict_main(args):
    device = torch.device(args.device)
    model, classes = load_model(args.model, device)
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )

    if args.image:
        label, conf, _ = predict_image(model, classes, args.image, device, tf)
        print(f"{args.image}  ->  {label}  (confidence {conf:.3f})")
        return

    image_dir = Path(args.image_dir)
    files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    out_path = Path(args.predictions)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "confidence"])
        for p in files:
            try:
                label, conf, _ = predict_image(model, classes, p, device, tf)
            except Exception as e:
                print(f"[warn] {p.name}: {e}", file=sys.stderr)
                continue
            writer.writerow([p.name, label, f"{conf:.4f}"])
            print(f"{p.name}  ->  {label}  ({conf:.3f})")
    print(f"\nPredictions written to: {out_path}")


# ============================================================
# CLI
# ============================================================


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train the CNN")
    tr.add_argument("--image_dir", required=True)
    tr.add_argument("--labels", default="labels.csv")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--dropout", type=float, default=0.4)
    tr.add_argument("--val_split", type=float, default=0.15)
    tr.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stop after N epochs without val improvement. 0 disables.",
    )
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--output", default="uno_symbol_cnn.pt")
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    pr = sub.add_parser("predict", help="Run the trained model on image(s)")
    pr.add_argument("--model", default="uno_symbol_cnn.pt")
    pr.add_argument("--image", help="Path to a single image to classify")
    pr.add_argument("--image_dir", help="Folder of images to classify (batch mode)")
    pr.add_argument(
        "--predictions",
        default="predictions.csv",
        help="Output CSV when using --image_dir",
    )
    pr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    if args.command == "train":
        train_main(args)
    else:
        if not args.image and not args.image_dir:
            p.error("Provide either --image or --image_dir for prediction.")
        predict_main(args)


if __name__ == "__main__":
    main()
