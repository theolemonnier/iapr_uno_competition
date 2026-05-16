from pathlib import Path

# Root of project
ROOT_DIR = Path(__file__).resolve().parent

# Folders
REFERENCE_IMAGES = ROOT_DIR / "reference_images"
TEST_IMAGES = ROOT_DIR / "test_images"
TRAIN_IMAGES = ROOT_DIR / "train_images"
SRC_DIR = ROOT_DIR / "src"

# path to images 
IMG_PATHS = {"reference": REFERENCE_IMAGES / "L1000767.jpg",
            "train_white_bg": TRAIN_IMAGES / "L1000777.jpg", # white
            "train_colorful_bg": TRAIN_IMAGES / "L1000974.jpg", # colorful
            "train_white_overlapp": TRAIN_IMAGES / "L1000851.jpg", # white bg and 2 identical colors 
            "my_reference": REFERENCE_IMAGES / "my_reference.png",
            "cards_contours_reference": REFERENCE_IMAGES / "contours_binary.png"
            }