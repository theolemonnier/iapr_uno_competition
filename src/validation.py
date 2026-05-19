"""Score backends against synthetic/game_state.csv using the official metric."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Callable, Any

import cv2
import pandas as pd

from src.detection import CardDetection
from src.orchestrator import process_image_with_detections
from src.player_zones import center_roi_frac, load_zone_fracs
from config import SYNTHETIC_GAME_STATE, SYNTHETIC_IMAGES

DetectFn = Callable[[Any], list[CardDetection]]


def _hand_to_counter(s: str) -> Counter[str]:
    if not s or s == "EMPTY":
        return Counter()
    return Counter(s.split(";"))


def _hand_f1(pred: str, gt: str) -> tuple[int, int, int]:
    p = _hand_to_counter(pred)
    g = _hand_to_counter(gt)
    tp = sum((p & g).values())
    fp = sum((p - g).values())
    fn = sum((g - p).values())
    return tp, fp, fn


def score_row(pred: dict, gt: dict) -> dict[str, float]:
    center_acc = float(pred["center_card"] == gt["center_card"])
    active_acc = float(pred["active_player"] == gt["active_player"])
    tp = fp = fn = 0
    for i in (1, 2, 3, 4):
        a, b, c = _hand_f1(pred[f"player_{i}_cards"], gt[f"player_{i}_cards"])
        tp += a; fp += b; fn += c
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * f1
    return {"center_acc": center_acc, "active_acc": active_acc, "f1": f1, "score": score}


def evaluate_backend(
    *,
    detect_fn: DetectFn,
    cnn,
    classes: list[str],
    device: str = "cpu",
    val_frac: float = 0.20,
    game_state_csv: Path = SYNTHETIC_GAME_STATE,
    images_dir: Path = SYNTHETIC_IMAGES,
) -> dict:
    df = pd.read_csv(game_state_csv).fillna("")
    n = len(df)
    n_val = max(1, int(round(n * val_frac)))
    val = df.iloc[-n_val:]
    zone_fracs = load_zone_fracs()
    center_roi = center_roi_frac(zone_fracs)

    rows: list[dict] = []
    for _, gt in val.iterrows():
        img_path = images_dir / f"{gt['image_id']}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        dets = detect_fn(image)
        pred = process_image_with_detections(
            image_id=gt["image_id"], image_bgr=image, detections=dets,
            cnn=cnn, classes=classes, zone_fracs=zone_fracs,
            center_roi=center_roi, device=device,
        )
        rows.append(score_row(pred, gt.to_dict()))

    agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]} if rows else {}
    agg["n"] = len(rows)
    return agg
