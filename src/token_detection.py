"""Active-player token detection.

Promoted from notebooks/token_detection.ipynb (kept intact for reference).
Regenerate with: .venv/bin/python scripts/extract_token_detection.py
"""

from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_player_from_centroid_position(
    centroid: Tuple[int], image_shape: Tuple[int, int]
) -> int:
    dx = (centroid[0] - image_shape[1] / 2.0) / image_shape[1]
    dy = (centroid[1] - image_shape[0] / 2.0) / image_shape[0]
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4  # right or left
    return 1 if dy > 0 else 3  # bottom or top


def detect_active_player_yellow(image: np.ndarray) -> int:
    """
    Find the yellow active-player token in an Uno game image.

    Returns 1 if the token is on the bottom, 2 if right, 3 if top, 4 if left. Returns 0 if no token is found.
    """
    # Create an hsv mask to isolate strong candidates for the yellow token
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([18, 70, 235]), np.array([29, 255, 255]))

    # Clean the mask by removing small objects and filling small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the yellow blob whose contour shape and area is most close to the token's
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center, best_circ = None, 0.7

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (
            15000 < area < 50000
        ):  # Relatively big range to remove outliers without risking to eliminate the token
            continue
        circularity = 4 * np.pi * area / cv2.arcLength(cnt, True) ** 2
        if circularity < best_circ:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        fill = cv2.countNonZero(mask[y : y + h, x : x + w]) / area
        if (
            fill < 0.8
        ):  # Reject rings that are not filled enough (specifically meant to avoid "skip turn" cards)
            continue
        M = cv2.moments(cnt)
        best_center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        best_circ = circularity

    if best_center is None:
        return 0

    # Find the player from the center of the token's identified shape
    return get_player_from_centroid_position(best_center, image.shape[:2])


def detect_active_player_black(image: np.ndarray) -> int:
    """
    Find the black active-player token in an Uno game image.

    Returns 1 if the token is on the bottom, 2 if right, 3 if top, 4 if left. Returns 0 if no token is found.
    """
    # Create an hsv mask to isolate strong candidates for the black token
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 55, 150]))

    # Clean the mask by removing small objects and filling small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the black blob whose contour shape and area is most close to the token's
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center, best_score = None, -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # skip pure noise
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(cnt)
        if w == 0 or h == 0:
            continue
        rect_score = area / (w * h)  # 1.0 = perfect rectangle
        ratio = max(w, h) / min(w, h)

        # Define a score where 1 means we are at the approximate ideal value for each and keep the best
        area_fit = 1 - min(1, abs(area - 40000) / 40000)
        aspect_fit = 1 - min(1, abs(ratio - 1.6) / 1.6)
        score = area_fit * rect_score * aspect_fit

        if score > best_score:
            best_score = score
            best_center = (cx, cy)

    if best_center is None:
        return 0

    # Find the player from the center of the token's identified shape
    return get_player_from_centroid_position(best_center, image.shape[:2])


def detect_background(image: np.ndarray) -> str:
    """Return which image background type the image has."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return "leafy" if hsv[:, :, 1].mean() > 50 else "white"


def detect_active_player(image: np.ndarray) -> int:
    """Output the current active player based on the image."""
    if detect_background(image) == "leafy":
        active_player = detect_active_player_yellow(image)
    else:
        active_player = detect_active_player_black(image)
    if active_player == 0:  # If no token was found, assign it to player 1 as a default
        return 1

    return active_player


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


def generate_pipeline_figure(image_path: str):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    image_type = detect_background(img_rgb)

    if image_type == "leafy":
        mask = cv2.inRange(hsv, np.array([18, 70, 235]), np.array([29, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    else:
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 55, 150]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay = img_rgb.copy()

    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 5)

    best_cnt = None
    best_center = None

    if image_type == "leafy":
        best_circ = 0.7
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (15000 < area < 50000):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / perimeter**2
            if circularity < best_circ:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            fill = cv2.countNonZero(mask_close[y : y + h, x : x + w]) / area
            if fill < 0.8:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                best_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                best_circ = circularity
                best_cnt = cnt
    else:
        best_score = -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            (cx, cy), (w, h), _ = cv2.minAreaRect(cnt)
            if w == 0 or h == 0:
                continue
            rect_score = area / (w * h)
            ratio = max(w, h) / min(w, h)
            area_fit = 1 - min(1, abs(area - 40000) / 40000)
            aspect_fit = 1 - min(1, abs(ratio - 1.6) / 1.6)
            score = area_fit * rect_score * aspect_fit

            if score > best_score:
                best_score = score
                best_center = (int(cx), int(cy))
                best_cnt = cnt

    if best_cnt is not None:
        cv2.drawContours(overlay, [best_cnt], -1, (0, 255, 0), 8)
        cv2.circle(overlay, best_center, radius=15, color=(255, 255, 0), thickness=-1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Initial HSV Mask")
    axes[2].imshow(mask_close, cmap="gray")
    axes[2].set_title("3. Morphological Clean")
    axes[3].imshow(overlay)
    axes[3].set_title("4. Selected Token & Center")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
