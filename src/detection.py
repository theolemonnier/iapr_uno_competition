"""Detection dataclass shared by both backends."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class CardDetection:
    """
    A localized card in a scene image.

    Attributes
    ----------
    quad : np.ndarray, shape (4, 2), dtype int
        Corner points in image (x, y) coords. For the YOLO backend this is the
        corner-bbox rectangle (no quad reconstruction).
    corner_crops : list[np.ndarray]
        80x80 RGB crops of the card's corner symbol(s). 1 or 2 elements.
    body_mask : np.ndarray, shape (H, W), dtype bool
        True where the card body color should be sampled.
    centroid : tuple[int, int]
        (x, y) in image coords. Used by player-zone assignment.
    source : str
        "classical" or "yolo" — for logging / debugging.
    confidence : Optional[float]
        YOLO confidence for that backend; None for classical.
    """
    quad: np.ndarray
    corner_crops: list[np.ndarray]
    body_mask: np.ndarray
    centroid: tuple[int, int]
    source: str
    confidence: Optional[float] = None
