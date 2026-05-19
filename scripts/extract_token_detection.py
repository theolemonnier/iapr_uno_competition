"""One-shot: extract code cells from notebooks/token_detection.ipynb
and prepend a module header + wrapper. Re-runnable; idempotent."""
from pathlib import Path
import json

NB = Path("notebooks/token_detection.ipynb")
OUT = Path("src/token_detection.py")

HEADER = '''"""Active-player token detection.

Promoted from notebooks/token_detection.ipynb (kept intact for reference).
Regenerate with: .venv/bin/python scripts/extract_token_detection.py
"""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

'''

FOOTER = '''

def detect_active_player_label(image: np.ndarray) -> Optional[str]:
    """Wrapper: return 'p1'..'p4' or None if no token found."""
    try:
        p = detect_active_player(image)
    except Exception:
        return None
    if p is None or p == 0:
        return None
    try:
        n = int(p)
    except (TypeError, ValueError):
        return None
    if 1 <= n <= 4:
        return f"p{n}"
    return None
'''


def main() -> int:
    nb = json.loads(NB.read_text())
    parts = [HEADER]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def " not in src:
            continue
        parts.append(src.rstrip() + "\n\n\n")
    parts.append(FOOTER)
    OUT.write_text("".join(parts))
    print(f"Wrote {OUT} ({len(parts)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
