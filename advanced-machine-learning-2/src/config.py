# Path variables
from pathlib import Path

ROOT_DIR = Path().cwd() / ".."

IMG_DIR = ROOT_DIR / "assets" / "images"
TRAIN_IMG_DIR = IMG_DIR / "train"

LABELS_DIR = ROOT_DIR / "assets" / "labels"
LABELS_PATH = LABELS_DIR / "labels.csv"

RESULTS_DIR = ROOT_DIR / "results"
GRAPH_RESULTS_DIR = RESULTS_DIR / "graphs"

# CNN path variables
CNN_RESULTS_DIR = RESULTS_DIR / "cnn"
CNN_MODELS_DIR = CNN_RESULTS_DIR / "models"
CNN_PLOTS_DIR = CNN_RESULTS_DIR / "plots"

# VAE path variables
VAE_RESULTS_DIR = RESULTS_DIR / "vae"
VAE_MODELS_DIR = VAE_RESULTS_DIR / "models"
VAE_PLOTS_DIR = VAE_RESULTS_DIR / "plots"

# GAN path variables
GAN_RESULTS_DIR = RESULTS_DIR / "gan"
GAN_MODELS_DIR = GAN_RESULTS_DIR / "models"
GAN_PLOTS_DIR = GAN_RESULTS_DIR / "plots"

# Diffusion model path variables
DF_RESULTS_DIR = RESULTS_DIR / "diffusion_model"
DF_MODELS_DIR = DF_RESULTS_DIR / "models"
DF_PLOTS_DIR = DF_RESULTS_DIR / "plots"

# Dataset variables
TEST_SIZE = 0.15 # 15% para teste
VAL_SIZE = 0.15  # 15% para validação

# Training variables
BATCH_SIZE = 32
IMAGE_SIZE = 64

RANDOM_SEED = 42
N_EPOCHS = 50