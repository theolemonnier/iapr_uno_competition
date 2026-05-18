"""Tune player-zone geometry visually.

Edit the constants below, then run:
    .venv/bin/python preview_zones.py

Writes ~/Desktop/uno_zones_preview.jpg comparing the zones over a synthetic
white canvas, a noisy background, and two real training images.
"""
import cv2
import numpy as np
import random
from pathlib import Path

# ====== TWEAK THESE ======
SCENE_W, SCENE_H = 1080, 720   # canvas size (aspect 1.5 = 3:2 matches train images)
MARGIN = 30                    # outer margin in px

# Zone fractions (0..1 of canvas):
P1_TOP_FRAC    = 0.65   # P1 bottom zone starts at this y-fraction
P3_BOTTOM_FRAC = 0.35   # P3 top zone ends at this y-fraction

P1_3_RIGHT_FRAC  = 0.30   # P1 and P3 left zone ends at this x-fraction
P1_3_LEFT_FRAC   = 0.7   # P1 and P3 right zone starts at this x-fraction

P2_LEFT_FRAC   = 0.65   # P2 right zone starts at this x-fraction
P4_RIGHT_FRAC  = 0.35   # P4 left zone ends at this x-fraction

P2_4_TOP_FRAC   = 0.75   # P2 and 4  zone starts at this y-fraction
P2_4_BOTTOM_FRAC   = 0.25   # P2 and 4  zone ends at this y-fraction



CENTER_XMIN, CENTER_XMAX = 0.35, 0.65
CENTER_YMIN, CENTER_YMAX = 0.35, 0.65
# =========================

M = MARGIN
# Player zones now form a "plus" around the center: each zone is constrained on
# both axes (bottom/top zones limited to the middle horizontally, left/right
# zones limited to the middle vertically). Corners are unused.
zones = {
    'P1 bottom': (int(P1_3_RIGHT_FRAC*SCENE_W), int(P1_TOP_FRAC*SCENE_H),    int(P1_3_LEFT_FRAC*SCENE_W),  SCENE_H - M),
    'P3 top'   : (int(P1_3_RIGHT_FRAC*SCENE_W), M,                           int(P1_3_LEFT_FRAC*SCENE_W),  int(P3_BOTTOM_FRAC*SCENE_H)),
    'P2 right' : (int(P2_LEFT_FRAC*SCENE_W),    int(P2_4_BOTTOM_FRAC*SCENE_H), SCENE_W - M,                int(P2_4_TOP_FRAC*SCENE_H)),
    'P4 left'  : (M,                            int(P2_4_BOTTOM_FRAC*SCENE_H), int(P4_RIGHT_FRAC*SCENE_W), int(P2_4_TOP_FRAC*SCENE_H)),
    'Center'   : (int(CENTER_XMIN*SCENE_W),     int(CENTER_YMIN*SCENE_H),    int(CENTER_XMAX*SCENE_W),     int(CENTER_YMAX*SCENE_H)),
}
colors = {
    'P1 bottom': (0, 255, 0),
    'P2 right' : (0, 200, 255),
    'P3 top'   : (255, 100, 100),
    'P4 left'  : (255, 0, 255),
    'Center'   : (0, 255, 255),
}


def draw_zones(img, label=''):
    out = img.copy()
    overlay = img.copy()
    for name, (x0, y0, x1, y1) in zones.items():
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colors[name], -1)
    out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)
    for name, (x0, y0, x1, y1) in zones.items():
        cv2.rectangle(out, (x0, y0), (x1, y1), colors[name], 3)
        cv2.putText(out, name, (x0 + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colors[name], 2, cv2.LINE_AA)
    cv2.putText(out, label, (10, SCENE_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    root = Path(__file__).parent
    bg_files = [p for p in sorted((root / 'data' / 'backgrounds').glob('*'))
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    random.seed(0)

    white = np.full((SCENE_H, SCENE_W, 3), 255, np.uint8)
    panels = [draw_zones(white, f'WHITE BG {SCENE_W}x{SCENE_H}')]

    if bg_files:
        bg = cv2.imread(str(random.choice(bg_files)))
        bg = cv2.resize(bg, (SCENE_W, SCENE_H))
        panels.append(draw_zones(bg, f'NOISY BG {SCENE_W}x{SCENE_H}'))
    else:
        panels.append(draw_zones(white.copy(), 'NO BG FILES FOUND'))

    for name in ('L1000770.jpg', 'L1000772.jpg'):
        p = root / 'iapr-26-uno-vision-challenge' / 'train_images' / name
        if p.exists():
            real = cv2.imread(str(p))
            panels.append(draw_zones(cv2.resize(real, (SCENE_W, SCENE_H)), f'REAL {p.stem}'))

    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:4]) if len(panels) >= 4 else np.hstack(panels[2:] + [panels[0]])
    grid = np.vstack([top, bot])

    out_path = Path.home() / 'Desktop' / 'uno_zones_preview.jpg'
    cv2.imwrite(str(out_path), grid)
    print(f'Wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
