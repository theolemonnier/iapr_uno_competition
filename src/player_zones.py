"""Player-zone geometry derived from data/jetons_et_backgrounds/backgrounds/bbox_player_position.csv."""
from __future__ import annotations
from collections import defaultdict
from typing import Optional
import logging

import pandas as pd

from src.detection import CardDetection
from config import BBOX_CSV

log = logging.getLogger(__name__)

ZoneRect = tuple[float, float, float, float]

FALLBACK_CENTER_ROI: ZoneRect = (0.40, 0.40, 0.60, 0.60)


def load_zone_fracs(csv_path=BBOX_CSV) -> dict[int, ZoneRect]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("player")
    ref_w = float((df["posx"] + df["width"]).max())
    ref_h = float((df["posy"] + df["height"]).max())
    out: dict[int, ZoneRect] = {}
    for p in (1, 2, 3, 4):
        row = df.loc[p]
        x0 = float(row["posx"]) / ref_w
        y0 = float(row["posy"]) / ref_h
        x1 = float(row["posx"] + row["width"]) / ref_w
        y1 = float(row["posy"] + row["height"]) / ref_h
        out[p] = (x0, y0, x1, y1)
    return out


def center_roi_frac(zone_fracs: dict[int, ZoneRect]) -> ZoneRect:
    try:
        x0 = zone_fracs[4][2]
        x1 = zone_fracs[2][0]
        y0 = zone_fracs[3][3]
        y1 = zone_fracs[1][1]
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("computed center ROI is invalid")
        return (x0, y0, x1, y1)
    except Exception as exc:
        log.warning("center_roi_frac fallback to %s (%s)", FALLBACK_CENTER_ROI, exc)
        return FALLBACK_CENTER_ROI


def _in_rect(cx_frac: float, cy_frac: float, rect: ZoneRect) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= cx_frac <= x1 and y0 <= cy_frac <= y1


def assign_player(
    centroid_xy: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    zone_fracs: dict[int, ZoneRect],
) -> Optional[int]:
    h, w = image_shape[:2]
    cx, cy = centroid_xy
    cx_f, cy_f = cx / w, cy / h
    for p, rect in zone_fracs.items():
        if _in_rect(cx_f, cy_f, rect):
            return p
    return None


def detect_center(
    detections: list[CardDetection],
    image_shape: tuple[int, int, int] | tuple[int, int],
    center_roi: ZoneRect,
) -> Optional[CardDetection]:
    h, w = image_shape[:2]
    inside = [
        d for d in detections
        if _in_rect(d.centroid[0] / w, d.centroid[1] / h, center_roi)
    ]
    if len(inside) == 1:
        return inside[0]
    return None


def assign_hands(
    detections: list[CardDetection],
    image_shape: tuple[int, int, int] | tuple[int, int],
    zone_fracs: dict[int, ZoneRect],
) -> dict[int, list[CardDetection]]:
    hands: dict[int, list[CardDetection]] = defaultdict(list)
    for d in detections:
        p = assign_player(d.centroid, image_shape, zone_fracs)
        if p is not None:
            hands[p].append(d)
    return {1: hands.get(1, []), 2: hands.get(2, []), 3: hands.get(3, []), 4: hands.get(4, [])}
