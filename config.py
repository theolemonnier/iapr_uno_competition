"""Project paths, single source of truth."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Data
DATA_DIR = ROOT_DIR / "data"
CARD_IMAGES = DATA_DIR / "card_images"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
JETONS_BG_DIR = DATA_DIR / "jetons_et_backgrounds"
BBOX_CSV = JETONS_BG_DIR / "backgrounds" / "bbox_player_position.csv"
KAGGLE_DIR = DATA_DIR / "iapr-26-uno-vision-challenge"
TEST_IMAGES = KAGGLE_DIR / "test_images"
TRAIN_IMAGES = KAGGLE_DIR / "train_images"

# Synthetic
SYNTHETIC_DIR = ROOT_DIR / "synthetic"
SYNTHETIC_IMAGES = SYNTHETIC_DIR / "images"
SYNTHETIC_LABELS = SYNTHETIC_DIR / "labels"
SYNTHETIC_CLASSES_TXT = SYNTHETIC_DIR / "classes.txt"
SYNTHETIC_DATA_YAML = SYNTHETIC_DIR / "data.yaml"
SYNTHETIC_GAME_STATE = SYNTHETIC_DIR / "game_state.csv"

# Pipeline outputs
SYMBOL_CROPS_DIR = DATA_DIR / "symbol_crops"
SYMBOL_CROPS_LABELS = SYMBOL_CROPS_DIR / "labels.csv"
MODELS_DIR = ROOT_DIR / "models"
CNN_CHECKPOINT = MODELS_DIR / "uno_symbol_cnn.pt"
YOLO_CHECKPOINT = MODELS_DIR / "yolo_uno.pt"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
