import pytest
import numpy as np
from src.player_zones import (
    load_zone_fracs,
    center_roi_frac,
    assign_player,
    detect_center,
    assign_hands,
    FALLBACK_CENTER_ROI,
)
from src.detection import CardDetection


@pytest.fixture
def zone_fracs():
    return load_zone_fracs()


@pytest.mark.unit
def test_zone_fracs_keys(zone_fracs):
    assert set(zone_fracs.keys()) == {1, 2, 3, 4}


@pytest.mark.unit
def test_zone_fracs_in_unit_square(zone_fracs):
    for p, (x0, y0, x1, y1) in zone_fracs.items():
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


@pytest.mark.unit
def test_center_roi_from_csv(zone_fracs):
    x0, y0, x1, y1 = center_roi_frac(zone_fracs)
    assert 0.22 < x0 < 0.25
    assert 0.74 < x1 < 0.77
    assert 0.28 < y0 < 0.31
    assert 0.69 < y1 < 0.72


@pytest.mark.unit
def test_center_roi_fallback_constant():
    assert FALLBACK_CENTER_ROI == (0.40, 0.40, 0.60, 0.60)


@pytest.mark.unit
@pytest.mark.parametrize("frac_xy,expected", [
    ((0.50, 0.95), 1),
    ((0.90, 0.50), 2),
    ((0.50, 0.05), 3),
    ((0.05, 0.50), 4),
    ((0.50, 0.50), None),
])
def test_assign_player(zone_fracs, frac_xy, expected):
    image_shape = (1000, 1000)
    cx, cy = int(frac_xy[0] * 1000), int(frac_xy[1] * 1000)
    assert assign_player((cx, cy), image_shape, zone_fracs) == expected


def _make_det(centroid):
    return CardDetection(
        quad=np.zeros((4, 2), dtype=int),
        corner_crops=[],
        body_mask=np.zeros((1, 1), dtype=bool),
        centroid=centroid,
        source="test",
    )


@pytest.mark.unit
def test_detect_center_unique(zone_fracs):
    image_shape = (1000, 1000)
    dets = [_make_det((500, 500)), _make_det((50, 50))]
    roi = center_roi_frac(zone_fracs)
    c = detect_center(dets, image_shape, roi)
    assert c is dets[0]


@pytest.mark.unit
def test_detect_center_none_when_multiple(zone_fracs):
    image_shape = (1000, 1000)
    dets = [_make_det((480, 500)), _make_det((520, 500))]
    roi = center_roi_frac(zone_fracs)
    assert detect_center(dets, image_shape, roi) is None


@pytest.mark.unit
def test_detect_center_none_when_empty(zone_fracs):
    image_shape = (1000, 1000)
    roi = center_roi_frac(zone_fracs)
    assert detect_center([_make_det((50, 50))], image_shape, roi) is None


@pytest.mark.unit
def test_assign_hands(zone_fracs):
    image_shape = (1000, 1000)
    dets = [
        _make_det((500, 950)),
        _make_det((950, 500)),
        _make_det((500, 50)),
        _make_det((50, 500)),
        _make_det((10, 10)),
    ]
    hands = assign_hands(dets, image_shape, zone_fracs)
    assert len(hands[1]) == 1
    assert len(hands[2]) == 1
    assert len(hands[3]) == 1
    assert len(hands[4]) == 1
    assert sum(len(v) for v in hands.values()) == 4
