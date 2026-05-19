"""Classical card segmentation primitives."""
from __future__ import annotations
from typing import Literal

import cv2
import numpy as np


def segment_cards(
    image_bgr: np.ndarray,
    background_kind: Literal["white", "noisy"] = "noisy",
    min_area: int = 40000,
) -> list[np.ndarray]:
    """Segment card-shaped polygons from a scene via HLS saturation."""
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    _, _, s = cv2.split(hls)
    blurred = cv2.GaussianBlur(s, (5, 5), 0)

    threshold_min = 15 if background_kind == "white" else 30
    threshold_max = 255

    mask = cv2.inRange(blurred, threshold_min, threshold_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) >= 3:
            polygons.append(approx)
    return polygons


def quad_centroid(quad: np.ndarray) -> tuple[int, int]:
    pts = quad.reshape(-1, 2)
    return (int(pts[:, 0].mean()), int(pts[:, 1].mean()))


def quad_to_body_mask(image_shape, quad: np.ndarray) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.reshape(-1, 2).astype(np.int32)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def extract_corner_crops(image_bgr: np.ndarray, quad: np.ndarray, crop_size: int = 80) -> list[np.ndarray]:
    pts = quad.reshape(-1, 2).astype(np.float32)
    if len(pts) != 4:
        return []
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]

    canon_w, canon_h = 200, 310
    dst = np.array([[0, 0], [canon_w - 1, 0], [canon_w - 1, canon_h - 1], [0, canon_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image_bgr, M, (canon_w, canon_h))

    cw, ch = 40, 60
    hl = warped[5:5 + ch, 5:5 + cw]
    lr = warped[canon_h - 5 - ch:canon_h - 5, canon_w - 5 - cw:canon_w - 5]
    crops = []
    for c in (hl, lr):
        if c.size == 0:
            continue
        resized = cv2.resize(c, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return crops
