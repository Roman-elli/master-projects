# Path variables
from pathlib import Path

ROOT_DIR = Path().cwd() / ".."

IMG_DIR = ROOT_DIR / "assets" / "images"
TRAIN_IMG_DIR = IMG_DIR / "train"

LABELS_DIR = ROOT_DIR / "assets" / "labels"
LABELS_PATH = LABELS_DIR / "labels.csv"

# Training variables
BATCH_SIZE = 32
IMAGE_SIZE = 64