"""Class vocabularies and label conversion for UNO cards."""
from __future__ import annotations
from typing import Optional

SYMBOL_CLASSES: list[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+2", "+4", "wild", "skip", "reverse",
]

FULL_CLASSES: list[str] = [
    # blue
    "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "b_6", "b_7", "b_8", "b_9",
    "b_draw_2", "b_reverse", "b_skip",
    # colorless
    "draw_4",
    # green
    "g_0", "g_1", "g_2", "g_3", "g_4", "g_5", "g_6", "g_7", "g_8", "g_9",
    "g_draw_2", "g_reverse", "g_skip",
    # red
    "r_0", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8", "r_9",
    "r_draw_2", "r_reverse", "r_skip",
    # colorless
    "wild",
    # yellow
    "y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6", "y_7", "y_8", "y_9",
    "y_draw_2", "y_reverse", "y_skip",
]

_COLORLESS = {"wild", "draw_4"}
_SYMBOL_TO_SUBMISSION = {"+2": "draw_2", "+4": "draw_4"}


def to_symbol(full_label: str) -> str:
    """Map a 54-class label to one of the 15 CNN symbol classes."""
    if full_label == "wild":
        return "wild"
    if full_label == "draw_4":
        return "+4"
    _, _, rest = full_label.partition("_")
    if rest == "draw_2":
        return "+2"
    return rest


def to_color(full_label: str) -> Optional[str]:
    """Return the color prefix ('r','g','b','y') or None for wild/draw_4."""
    if full_label in _COLORLESS:
        return None
    return full_label.split("_", 1)[0]


def submission_label(color: Optional[str], symbol: str) -> str:
    """Combine color + symbol into the Kaggle submission alphabet."""
    if symbol == "wild":
        return "wild"
    if symbol == "+4":
        return "draw_4"
    sub_sym = _SYMBOL_TO_SUBMISSION.get(symbol, symbol)
    if color is None:
        raise ValueError(f"non-colorless symbol {symbol!r} requires a color")
    return f"{color}_{sub_sym}"
