"""Train YOLOv8n on synthetic UNO scenes for card localization.

Per project rules (CLAUDE.md): no pretrained models. `YOLO("yolov8n.yaml")`
instantiates the architecture without loading the COCO-pretrained .pt.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from config import SYNTHETIC_DATA_YAML, MODELS_DIR, YOLO_CHECKPOINT


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.yaml")
    results = model.train(
        data=str(SYNTHETIC_DATA_YAML),
        epochs=80,
        imgsz=640,
        batch=16,
        project=str(MODELS_DIR),
        name="yolo_uno_run",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    YOLO_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    YOLO_CHECKPOINT.write_bytes(best.read_bytes())
    print(f"Wrote {YOLO_CHECKPOINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
