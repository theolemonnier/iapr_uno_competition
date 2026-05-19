"""Per-image orchestration and full-test-set submission writer."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from src.card_labeler import fmt_hand, predict_card
from src.detection import CardDetection
from src.player_zones import (
    assign_hands,
    center_roi_frac,
    detect_center,
    load_zone_fracs,
)
from src.token_detection import detect_active_player_label

Backend = Literal["classical", "yolo"]
DetectFn = Callable[[np.ndarray], list[CardDetection]]

_SUBMISSION_COLUMNS = [
    "image_id", "center_card", "active_player",
    "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards",
]


def process_image_with_detections(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    detections: list[CardDetection],
    cnn,
    classes: list[str],
    zone_fracs: dict[int, tuple[float, float, float, float]],
    center_roi: tuple[float, float, float, float],
    device: str,
) -> dict:
    center = detect_center(detections, image_bgr.shape, center_roi)
    hands = assign_hands(
        [d for d in detections if d is not center],
        image_bgr.shape, zone_fracs,
    )
    active = detect_active_player_label(image_bgr) or "p1"

    center_label = ""
    if center is not None:
        c = predict_card(center, image_bgr, cnn, device, classes)
        center_label = c or ""

    row = {"image_id": image_id, "center_card": center_label, "active_player": active}
    for p in (1, 2, 3, 4):
        labels = [
            lbl for d in hands[p]
            if (lbl := predict_card(d, image_bgr, cnn, device, classes)) is not None
        ]
        row[f"player_{p}_cards"] = fmt_hand(labels)
    return row


def process_image(
    *,
    image_path: Path,
    detect_fn: DetectFn,
    cnn,
    classes: list[str],
    zone_fracs: dict,
    center_roi: tuple,
    device: str,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    detections = detect_fn(image)
    return process_image_with_detections(
        image_id=image_path.stem,
        image_bgr=image,
        detections=detections,
        cnn=cnn,
        classes=classes,
        zone_fracs=zone_fracs,
        center_roi=center_roi,
        device=device,
    )


def run_submission(
    *,
    image_dir: Path,
    out_csv: Path,
    detect_fn: DetectFn,
    cnn,
    classes: list[str],
    device: str = "cpu",
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    zone_fracs = load_zone_fracs()
    center_roi = center_roi_frac(zone_fracs)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUBMISSION_COLUMNS)
        w.writeheader()
        for p in paths:
            try:
                row = process_image(
                    image_path=p, detect_fn=detect_fn, cnn=cnn,
                    classes=classes, zone_fracs=zone_fracs,
                    center_roi=center_roi, device=device,
                )
            except Exception as exc:
                print(f"[warn] {p.name}: {exc}")
                row = {c: "" for c in _SUBMISSION_COLUMNS}
                row["image_id"] = p.stem
                row["active_player"] = "p1"
                for i in (1, 2, 3, 4):
                    row[f"player_{i}_cards"] = "EMPTY"
            w.writerow(row)
    print(f"Wrote {out_csv} ({len(paths)} rows)")
