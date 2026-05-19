"""YOLO detection backend (section E).

YOLO outputs corner-bbox detections. Body mask is a ring of pixels just
outside the corner bbox where the card color lives; centroid is bbox center.
"""
from __future__ import annotations
import numpy as np
import cv2

from src.detection import CardDetection
from config import YOLO_CHECKPOINT


_RING_OUTER_FRAC = 1.5
_RING_INNER_FRAC = 0.2


class _YoloRunner:
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from ultralytics import YOLO
            cls._model = YOLO(str(YOLO_CHECKPOINT))
        return cls._model


def _corner_crop(image_bgr, x0, y0, x1, y1, crop_size=80):
    h, w = image_bgr.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _ring_mask(image_shape, x0, y0, x1, y1, outer_frac=_RING_OUTER_FRAC, inner_frac=_RING_INNER_FRAC):
    h, w = image_shape[:2]
    bw = x1 - x0; bh = y1 - y0
    side = max(bw, bh)
    pad_out = int(round(side * outer_frac))
    pad_in = int(round(side * inner_frac))
    ox0 = max(0, x0 - pad_out); oy0 = max(0, y0 - pad_out)
    ox1 = min(w, x1 + pad_out); oy1 = min(h, y1 + pad_out)
    ix0 = max(0, x0 - pad_in); iy0 = max(0, y0 - pad_in)
    ix1 = min(w, x1 + pad_in); iy1 = min(h, y1 + pad_in)
    mask = np.zeros((h, w), dtype=bool)
    mask[oy0:oy1, ox0:ox1] = True
    mask[iy0:iy1, ix0:ix1] = False
    return mask


def detect_cards_yolo(image_bgr: np.ndarray, conf: float = 0.25) -> list[CardDetection]:
    model = _YoloRunner.get()
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    detections: list[CardDetection] = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for (x0, y0, x1, y1), c in zip(xyxy, confs):
        crop = _corner_crop(image_bgr, x0, y0, x1, y1)
        if crop is None:
            continue
        mask = _ring_mask(image_bgr.shape, x0, y0, x1, y1)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        detections.append(CardDetection(
            quad=quad,
            corner_crops=[crop],
            body_mask=mask,
            centroid=((x0 + x1) // 2, (y0 + y1) // 2),
            source="yolo",
            confidence=float(c),
        ))
    return detections
