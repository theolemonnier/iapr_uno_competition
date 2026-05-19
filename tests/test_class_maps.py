import pytest
from src.class_maps import (
    SYMBOL_CLASSES,
    FULL_CLASSES,
    to_symbol,
    to_color,
    submission_label,
)


@pytest.mark.unit
def test_symbol_classes_count():
    assert len(SYMBOL_CLASSES) == 15


@pytest.mark.unit
def test_full_classes_count():
    assert len(FULL_CLASSES) == 54


@pytest.mark.unit
@pytest.mark.parametrize("full,sym", [
    ("b_0", "0"),
    ("r_9", "9"),
    ("y_skip", "skip"),
    ("g_reverse", "reverse"),
    ("b_draw_2", "+2"),
    ("wild", "wild"),
    ("draw_4", "+4"),
])
def test_to_symbol(full, sym):
    assert to_symbol(full) == sym


@pytest.mark.unit
@pytest.mark.parametrize("full,color", [
    ("b_0", "b"),
    ("r_9", "r"),
    ("y_skip", "y"),
    ("g_reverse", "g"),
    ("wild", None),
    ("draw_4", None),
])
def test_to_color(full, color):
    assert to_color(full) == color


@pytest.mark.unit
@pytest.mark.parametrize("color,symbol,expected", [
    ("b", "0", "b_0"),
    ("y", "skip", "y_skip"),
    ("r", "+2", "r_draw_2"),
    (None, "wild", "wild"),
    (None, "+4", "draw_4"),
])
def test_submission_label(color, symbol, expected):
    assert submission_label(color, symbol) == expected
