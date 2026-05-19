"""HSV-based card-body color classifier.

Reuses extract_color_masks from src/detect_reference.py for the tuned ranges.
"""
from __future__ import annotations
import numpy as np

from src.detect_reference import extract_color_masks

_LABEL_OF = {"Yellow": "y", "Red": "r", "Blue": "b", "Green": "g"}


def classify_color(image_bgr: np.ndarray, body_mask: np.ndarray) -> str:
    """Return one of {'r','g','b','y'}."""
    masks = extract_color_masks(image_bgr, plot=False)
    bm = body_mask.astype(bool)
    counts = {
        label: int(np.count_nonzero((masks[name] > 0) & bm))
        for name, label in _LABEL_OF.items()
    }
    return max(counts, key=counts.get)
