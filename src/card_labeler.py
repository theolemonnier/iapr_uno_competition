"""Compose CNN symbol + HSV color into a Kaggle submission label."""
from __future__ import annotations
from typing import Optional
import numpy as np

from src.detection import CardDetection
from src.color_classifier import classify_color
from src.class_maps import submission_label

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def _get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


def load_cnn(checkpoint_path: str, device: str = "cpu"):
    """Load the trained UnoSymbolCNN. Attaches `model.classes` for inference."""
    import torch
    from hybric_detection_utils.train_uno import UnoSymbolCNN
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = UnoSymbolCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.classes = classes
    return model, classes


def _run_cnn(model, crop_rgb: np.ndarray, device: str) -> tuple[str, float]:
    """Single-crop inference. Returns (symbol_class, confidence)."""
    import torch
    from PIL import Image
    img = Image.fromarray(crop_rgb)
    x = _get_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax().item())
    return model.classes[idx], float(probs[idx].item())


def predict_card(
    det: CardDetection,
    image_bgr: np.ndarray,
    cnn,
    device: str = "cpu",
    classes: Optional[list[str]] = None,
) -> Optional[str]:
    if not det.corner_crops:
        return None
    best_sym: Optional[str] = None
    best_conf = -1.0
    for crop in det.corner_crops:
        sym, conf = _run_cnn(cnn, crop, device)
        if conf > best_conf:
            best_conf = conf
            best_sym = sym
    if best_sym is None:
        return None
    if best_sym == "wild" or best_sym == "+4":
        return submission_label(None, best_sym)
    color = classify_color(image_bgr, det.body_mask)
    return submission_label(color, best_sym)


def fmt_hand(labels: list[str]) -> str:
    if not labels:
        return "EMPTY"
    return ";".join(labels)
