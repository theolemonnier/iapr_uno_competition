"""Extract 80x80 symbol crops from synthetic YOLO labels for CNN training."""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_maps import FULL_CLASSES, SYMBOL_CLASSES, to_symbol
from config import (
    SYNTHETIC_IMAGES,
    SYNTHETIC_LABELS,
    SYNTHETIC_CLASSES_TXT,
    SYMBOL_CROPS_DIR,
    SYMBOL_CROPS_LABELS,
)

CROP_SIZE = 80
PAD_FRAC = 0.30


def _bbox_to_pixel(cx, cy, w, h, img_w, img_h, pad_frac):
    bw = w * img_w * (1 + pad_frac)
    bh = h * img_h * (1 + pad_frac)
    side = max(bw, bh)
    px = cx * img_w
    py = cy * img_h
    x0 = int(round(px - side / 2))
    y0 = int(round(py - side / 2))
    x1 = int(round(px + side / 2))
    y1 = int(round(py + side / 2))
    return x0, y0, x1, y1


def _safe_crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bot = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bot:
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REPLICATE,
        )
        x0 += pad_left; x1 += pad_left
        y0 += pad_top; y1 += pad_top
    return img[y0:y1, x0:x1]


def main() -> int:
    disk_classes = SYNTHETIC_CLASSES_TXT.read_text().splitlines()
    assert disk_classes == FULL_CLASSES, "synthetic/classes.txt drift"

    SYMBOL_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for s in SYMBOL_CLASSES:
        (SYMBOL_CROPS_DIR / s.replace("+", "plus_")).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    rows: list[tuple[str, str, int]] = []
    label_files = sorted(SYNTHETIC_LABELS.glob("*.txt"))
    for lf in label_files:
        img_path = SYNTHETIC_IMAGES / f"{lf.stem}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for k, line in enumerate(lf.read_text().splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
            full = FULL_CLASSES[cls_id]
            symbol = to_symbol(full)
            x0, y0, x1, y1 = _bbox_to_pixel(cx, cy, w, h, iw, ih, PAD_FRAC)
            crop = _safe_crop(img, x0, y0, x1, y1)
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
            sym_dir = SYMBOL_CROPS_DIR / symbol.replace("+", "plus_")
            out_name = f"{lf.stem}_{k:02d}.jpg"
            out_path = sym_dir / out_name
            cv2.imwrite(str(out_path), crop)
            rows.append((f"{symbol.replace('+', 'plus_')}/{out_name}", symbol, 1))
            counts[symbol] += 1

    SYMBOL_CROPS_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMBOL_CROPS_LABELS, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "label", "usable"])
        for r in rows:
            wr.writerow(r)

    print("Crops written:", sum(counts.values()))
    for s in SYMBOL_CLASSES:
        n = counts.get(s, 0)
        marker = "" if n >= 50 else "  WARN below 50"
        print(f"  {s:>8s}: {n}{marker}")
    missing = [s for s in SYMBOL_CLASSES if counts.get(s, 0) == 0]
    if missing:
        print(f"FATAL: missing classes {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
