import numpy as np
import pytest
from src.color_classifier import classify_color


def _solid(color_bgr, size=(100, 100)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


@pytest.mark.unit
@pytest.mark.parametrize("bgr,expected", [
    ((0, 0, 255), "r"),
    ((255, 0, 0), "b"),
    ((0, 255, 0), "g"),
    ((0, 255, 255), "y"),
])
def test_classify_color_solid(bgr, expected):
    img = _solid(bgr)
    mask = np.ones(img.shape[:2], dtype=bool)
    assert classify_color(img, mask) == expected


@pytest.mark.unit
def test_classify_color_uses_only_masked_pixels():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = (0, 0, 255)
    img[50:, :] = (255, 0, 0)
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :] = True
    assert classify_color(img, mask) == "r"
