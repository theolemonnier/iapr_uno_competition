import numpy as np
import pytest
from unittest.mock import MagicMock

from src.card_labeler import fmt_hand, predict_card
from src.detection import CardDetection


def _make_det(crops):
    return CardDetection(
        quad=np.zeros((4, 2), dtype=int),
        corner_crops=crops,
        body_mask=np.ones((10, 10), dtype=bool),
        centroid=(5, 5),
        source="test",
    )


@pytest.mark.unit
def test_fmt_hand_empty():
    assert fmt_hand([]) == "EMPTY"


@pytest.mark.unit
def test_fmt_hand_multi():
    out = fmt_hand(["b_0", "r_skip"])
    assert out == "b_0;r_skip"


@pytest.mark.unit
def test_predict_card_wild(monkeypatch):
    fake_cnn = MagicMock()
    monkeypatch.setattr("src.card_labeler._run_cnn", lambda model, crop, device: ("wild", 0.99))
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    det = _make_det([np.zeros((80, 80, 3), dtype=np.uint8)])
    assert predict_card(det, img, fake_cnn, device="cpu") == "wild"


@pytest.mark.unit
def test_predict_card_color_symbol(monkeypatch):
    fake_cnn = MagicMock()
    monkeypatch.setattr("src.card_labeler._run_cnn", lambda model, crop, device: ("5", 0.9))
    monkeypatch.setattr("src.card_labeler.classify_color", lambda img, mask: "r")
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    det = _make_det([np.zeros((80, 80, 3), dtype=np.uint8)])
    assert predict_card(det, img, fake_cnn, device="cpu") == "r_5"


@pytest.mark.unit
def test_predict_card_no_crops(monkeypatch):
    fake_cnn = MagicMock()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    det = _make_det([])
    assert predict_card(det, img, fake_cnn, device="cpu") is None
