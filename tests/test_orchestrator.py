import numpy as np
import pytest
from unittest.mock import MagicMock

from src.orchestrator import process_image_with_detections
from src.detection import CardDetection
from src.player_zones import load_zone_fracs, center_roi_frac


def _make_det(centroid):
    return CardDetection(
        quad=np.zeros((4, 2), dtype=int),
        corner_crops=[np.zeros((80, 80, 3), dtype=np.uint8)],
        body_mask=np.ones((1000, 1000), dtype=bool),
        centroid=centroid,
        source="test",
    )


@pytest.fixture
def fracs():
    return load_zone_fracs()


@pytest.fixture
def roi(fracs):
    return center_roi_frac(fracs)


@pytest.mark.unit
def test_process_image_with_detections_full_row(fracs, roi, monkeypatch):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    dets = [
        _make_det((500, 500)),
        _make_det((500, 950)),
        _make_det((950, 500)),
    ]
    monkeypatch.setattr(
        "src.orchestrator.predict_card",
        lambda d, img, cnn, device, classes: f"card_at_{d.centroid[0]}_{d.centroid[1]}",
    )
    monkeypatch.setattr("src.orchestrator.detect_active_player_label", lambda img: "p2")

    row = process_image_with_detections(
        image_id="game_test",
        image_bgr=image,
        detections=dets,
        cnn=MagicMock(),
        classes=["dummy"],
        zone_fracs=fracs,
        center_roi=roi,
        device="cpu",
    )
    assert row["image_id"] == "game_test"
    assert row["center_card"] == "card_at_500_500"
    assert row["active_player"] == "p2"
    assert row["player_1_cards"] == "card_at_500_950"
    assert row["player_2_cards"] == "card_at_950_500"
    assert row["player_3_cards"] == "EMPTY"
    assert row["player_4_cards"] == "EMPTY"


@pytest.mark.unit
def test_process_image_active_fallback(fracs, roi, monkeypatch):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    monkeypatch.setattr("src.orchestrator.predict_card", lambda *a, **kw: "x")
    monkeypatch.setattr("src.orchestrator.detect_active_player_label", lambda img: None)
    row = process_image_with_detections(
        image_id="x", image_bgr=image, detections=[], cnn=MagicMock(),
        classes=["dummy"], zone_fracs=fracs, center_roi=roi, device="cpu",
    )
    assert row["active_player"] == "p1"
    assert row["center_card"] == ""
    assert all(row[f"player_{i}_cards"] == "EMPTY" for i in (1, 2, 3, 4))
