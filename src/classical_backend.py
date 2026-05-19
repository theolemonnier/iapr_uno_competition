"""Classical CV detection backend (section D)."""
from __future__ import annotations
import numpy as np

from src.detection import CardDetection
from src.segment_cards import (
    segment_cards,
    quad_centroid,
    quad_to_body_mask,
    extract_corner_crops,
)
from src.token_detection import detect_background


def detect_cards_classical(image_bgr: np.ndarray) -> list[CardDetection]:
    bg = detect_background(image_bgr)
    kind = "white" if bg == "white" else "noisy"
    polys = segment_cards(image_bgr, background_kind=kind)
    detections: list[CardDetection] = []
    for poly in polys:
        if len(poly) != 4:
            continue
        quad = poly.reshape(-1, 2).astype(np.int32)
        crops = extract_corner_crops(image_bgr, quad)
        if not crops:
            continue
        mask = quad_to_body_mask(image_bgr.shape, quad)
        detections.append(CardDetection(
            quad=quad,
            corner_crops=crops,
            body_mask=mask,
            centroid=quad_centroid(quad),
            source="classical",
        ))
    return detections
