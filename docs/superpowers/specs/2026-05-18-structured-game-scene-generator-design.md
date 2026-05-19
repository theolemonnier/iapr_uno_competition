# Structured UNO Game-Scene Generator

**Date:** 2026-05-18
**Scope:** Add a new composer to `generate_data.ipynb` that synthesises scenes mirroring the real test-image layout (four player zones, optional center card, active-player token) in addition to the existing free-placement composers.

## Goal

The existing notebook generates random card scatter useful for training a generic
card detector. The Kaggle test images are structured: four players sit around a
table in fixed positions, an optional center card sits in the middle, and a
small physical token marks the active player. To improve training data
distribution we add a third composer, `compose_game_scene`, that produces
images closer to the real distribution while keeping the existing composers
available.

## Non-goals

- Replacing the existing composers (`compose_no_overlap`, `compose_occlusion`).
- Adding the active-player token as a YOLO class (token will be unlabeled and
  detected at inference via colour/shape thresholding).
- Building the inference pipeline. Only the data generator changes here.

## Inputs / outputs

**Inputs (unchanged):** `data/card_images/*.jpg`, `data/annotations.csv`,
`data/backgrounds/*`.

**New outputs (in addition to the existing YOLO files):**
- `synthetic/images/game_{idx:05d}.jpg`
- `synthetic/labels/game_{idx:05d}.txt` (YOLO format, one line per visible
  corner hull, same schema as existing labels)
- `synthetic/game_state.csv` — one row per generated game-scene image, columns:
  `image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards`
  using the Kaggle submission semantics (`EMPTY` for empty hands, `;` as
  intra-hand separator, `p1..p4` for active player).

## Scene geometry

Canvas size: `SCENE_W × SCENE_H = 1080 × 720` (aspect 1.5 = 3:2 to match the real
train images, which are 4000×2662). Margin `M = 30 px` (outer border).

**Plus-shaped layout:** Player zones are constrained on both axes so they form
a `+` around the center. Bottom/top zones (P1, P3) are limited to a middle
horizontal strip; left/right zones (P2, P4) are limited to a middle vertical
strip. The four corners are unused. The five rectangles tile the middle of
the canvas without overlap (zone edges touch but do not cross).

| Zone   | x range                                  | y range                                |
|--------|------------------------------------------|----------------------------------------|
| P1 bottom | `[0.30·W, 0.70·W]`                    | `[0.65·H, H - M]`                      |
| P3 top    | `[0.30·W, 0.70·W]`                    | `[M, 0.35·H]`                          |
| P2 right  | `[0.65·W, W - M]`                     | `[0.25·H, 0.75·H]`                     |
| P4 left   | `[M, 0.35·W]`                         | `[0.25·H, 0.75·H]`                     |
| Center    | `[0.35·W, 0.65·W]`                    | `[0.35·H, 0.65·H]`                     |

Implementation constants (kept in sync between `preview_zones.py` and the
notebook helpers cell):

```python
SCENE_W, SCENE_H = 1080, 720
MARGIN           = 30
P1_TOP_FRAC      = 0.65   # P1 bottom zone starts at this y-fraction
P3_BOTTOM_FRAC   = 0.35   # P3 top zone ends at this y-fraction
P1_3_RIGHT_FRAC  = 0.30   # P1 and P3 left x edge (zone x_min)
P1_3_LEFT_FRAC   = 0.70   # P1 and P3 right x edge (zone x_max)
P2_LEFT_FRAC     = 0.65   # P2 right zone starts at this x-fraction
P4_RIGHT_FRAC    = 0.35   # P4 left zone ends at this x-fraction
P2_4_BOTTOM_FRAC = 0.25   # P2 and P4 top y edge (zone y_min)
P2_4_TOP_FRAC    = 0.75   # P2 and P4 bottom y edge (zone y_max)
CENTER_XMIN, CENTER_XMAX = 0.35, 0.65
CENTER_YMIN, CENTER_YMAX = 0.35, 0.65
```

A card is "in" its zone when the card's centre point falls inside the zone
rectangle. Card sprites may visually spill outside the zone — that is desired
to mimic real images.

## Composition algorithm

```
compose_game_scene():
  # 1. Background
  bg_type = random.choice(['white', 'noisy'])
  if bg_type == 'white':
      scene = np.full((H, W, 3), 255, np.uint8)
      scene += small_gaussian_noise()              # ±5 grey levels
  else:
      scene = random_background()                   # existing helper

  placed_all = []         # for YOLO labels (every visible card)
  hands = {p: [] for p in ['p1','p2','p3','p4']}   # for game_state.csv

  # 2. Player hands
  for p in ['p1','p2','p3','p4']:
      if random.random() < 0.30:
          continue   # empty slot
      n_cards = random.randint(1, 7)
      placed_in_zone = []
      for _ in range(n_cards):
          card = random.choice(CARDS)
          sprite, hulls, quad = warp_card(card,
                                          scale=random.uniform(0.6, 0.9),
                                          angle=random.uniform(-180, 180))
          # place center at random (x,y) inside zone_p; reject if OOB after
          # accounting for sprite half-extents
          for _try in range(30):
              cx, cy = random_point_in_zone(p)
              if sprite_fits(sprite, cx, cy):
                  break
          else:
              continue   # could not fit, skip this card
          paste sprite onto scene
          placed_in_zone.append({'card': card, 'hulls': hulls_in_scene,
                                  'quad': quad_in_scene})

      # Hull-visibility post-pass (same logic as compose_occlusion):
      # for each card in this hand, drop hulls covered >5% by any
      # later-placed card in the SAME hand.
      visible = post_pass_visibility(placed_in_zone)
      placed_all.extend(visible)
      hands[p] = [pl['card']['name'] for pl in visible]

  # 3. Center card (80% chance)
  center_card_name = ''
  if random.random() < 0.80:
      for _try in range(30):
          cx, cy = random_point_in_zone('center')
          if (sprite_fits and not_overlapping_existing(scene-level quads, allow=False)):
              break
      if placed:
          paste; placed_all.append(...)
          center_card_name = card.name
  # else center stays as empty string (Kaggle CSV semantics for "no card" is
  # ambiguous; we leave empty and document — actual handling is consumer's
  # choice when reading game_state.csv).

  # 4. Active-player token
  non_empty = [p for p in ['p1','p2','p3','p4'] if hands[p]]
  if non_empty:
      active = random.choice(non_empty)
      token_xy = sample_point_near_hand(active)  # offset to one side of zone
      if bg_type == 'white':
          draw_black_square(scene, token_xy, size~30×50, angle ~U(-15,15))
      else:
          draw_yellow_disc(scene, token_xy, radius~15)
      # Token has no YOLO label.
  else:
      active = 'p1'   # fallback; image has no players
                      # (rare since per-player skip prob is 0.30, so all-empty ≈ 0.81%)

  # 5. Emit files
  write image, YOLO label file (using yolo_lines_from_placed(placed_all)),
  append row to game_state.csv.
```

## Helpers to add

- `player_zones() -> dict[str, tuple[int,int,int,int]]`
- `random_point_in_zone(zone_name) -> (int,int)`
- `sprite_fits(sprite, cx, cy) -> bool`
- `draw_black_square(scene, xy, ...)` and `draw_yellow_disc(scene, xy, ...)`
- `sample_point_near_hand(player) -> (int,int)` — picks a small region adjacent
  to the player's zone (e.g. for P3 top: x near hand centroid, y just below the
  top margin)
- `post_pass_visibility(placed_in_zone, coverage_thresh=0.05)` — reuses the
  hull-coverage logic already inside `compose_occlusion` (refactor into a
  shared helper).

## Integration with `generate_dataset`

Extend `generate_dataset(...)` signature:

```
def generate_dataset(n_no_overlap=300,
                    n_occlusion=300,
                    n_game_scene=600):
```

After writing the existing batches, append a third loop:

```
write_classes_file()   # unchanged
# ... existing loops ...
with open(OUT_DIR / 'game_state.csv', 'w') as f:
    f.write('image_id,center_card,active_player,'
            'player_1_cards,player_2_cards,player_3_cards,player_4_cards\n')
    for i in tqdm(range(n_game_scene), desc='game-scene'):
        result = compose_game_scene()
        if result is None:
            continue
        scene, placed, hands, center, active = result
        stem = f'game_{i:05d}'
        save_sample(stem, scene, placed)
        f.write(format_row(stem, center, active, hands) + '\n')
```

`format_row` converts hand lists to `;`-joined strings, mapping empty hands to
`EMPTY`.

## Parameter summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bg_type` distribution | 50/50 white vs noisy | matches the two scene categories in CLAUDE.md |
| Per-player empty probability | `0.30` | real images show 1–2 empty slots typically |
| Hand size | `randint(1, 7)` | user-specified; real images skew lower but generator over-covers |
| Card scale | `U(0.6, 0.9)` | smaller than free-placement composers so 7 cards fit a zone |
| Card angle | `U(-180°, 180°)` | fully random rotation per user choice |
| Center-card probability | `0.80` | 20% augmentation for "no center visible" robustness |
| Token (white bg) | black filled rectangle ~30×50 px, rotation U(-15°,15°) | matches L1000770 |
| Token (noisy bg) | yellow filled disc, radius ~15 px | matches L1000983 |
| Coverage threshold for hull visibility | `0.05` | matches existing `compose_occlusion` |
| Default `n_game_scene` | `600` | order of magnitude similar to existing composers |

## Edge cases

- **All players empty** (~0.81% per the 30% skip rate): no token drawn,
  `active='p1'` placeholder in CSV, hands all `EMPTY`. Consumer can filter
  these rows if undesired.
- **Center card placement fails** after 30 tries: scene proceeds without a
  centre card; `center_card` column stays empty string (caller decides how to
  interpret).
- **Card cannot fit zone** after 30 tries: that card is skipped silently; the
  hand simply ends up shorter than `n_cards`.
- **Token overlaps cards**: not specifically prevented. Sampling
  `sample_point_near_hand` uses a small offset region that is typically clear,
  but no rejection sampling. Acceptable noise.

## Out of scope / explicitly deferred

- Realistic per-player rotation bias (cards oriented toward each player). User
  preferred fully random rotation.
- Intra-hand overlap variant (geaxgx fan/stack inside one hand). Not selected.
- Token classification by YOLO. Token detection will be a separate inference-time
  module.
- Validation / metric harness using `game_state.csv`. Future work.

## Files changed

- `generate_data.ipynb`:
  - one new code cell defining `compose_game_scene` + helpers
  - existing `generate_dataset` cell updated to add the third loop and write
    `synthetic/game_state.csv`
- No other files modified.

## Verification

After the notebook runs:
1. `synthetic/images/game_*.jpg` and matching `.txt` label files exist.
2. `synthetic/game_state.csv` has the right header and one row per `game_*`
   image.
3. Visual preview cell renders ≥6 game scenes with bbox + zone overlays so a
   human can eyeball that hands land in their zones and tokens render in the
   right colour for the bg type.
4. Spot-check that YOLO labels match `game_state.csv`: for one sample, the set
   of class IDs in the `.txt` is a (multi-)subset of the union of cards listed
   across hands + center.
