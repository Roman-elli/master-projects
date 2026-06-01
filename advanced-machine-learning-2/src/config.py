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
CNN_RESULTS_DIR = RESULTS_DIR / "models" # "cnn_seed_42" "cnn_seed_100"
CNN_MODELS_DIR = CNN_RESULTS_DIR
CNN_PLOTS_DIR = CNN_RESULTS_DIR / "plots"

TARGET_CLASSES_PATH = CNN_RESULTS_DIR / 'classes_para_augmentar.csv' 

CNN_VAE_RESULTS_DIR = RESULTS_DIR / "augmented_dataset" / "cnn_vae"
CNN_VAE_MODELS_DIR = CNN_VAE_RESULTS_DIR / "models"
CNN_VAE_PLOTS_DIR = CNN_VAE_RESULTS_DIR / "plots"

CNN_GAN_RESULTS_DIR = RESULTS_DIR / "augmented_dataset" / "cnn_gan"
CNN_GAN_MODELS_DIR = CNN_GAN_RESULTS_DIR / "models"
CNN_GAN_PLOTS_DIR = CNN_GAN_RESULTS_DIR / "plots"

CNN_DIFF_RESULTS_DIR = RESULTS_DIR / "augmented_dataset" / "cnn_diff"
CNN_DIFF_MODELS_DIR = CNN_DIFF_RESULTS_DIR / "models"
CNN_DIFF_PLOTS_DIR = CNN_DIFF_RESULTS_DIR / "plots"

# VAE path variables
VAE_RESULTS_DIR = RESULTS_DIR / "models" / "vae"
VAE_MODELS_DIR = VAE_RESULTS_DIR / "models"
VAE_PLOTS_DIR = VAE_RESULTS_DIR / "plots"
VAE_AUGMENTED_DIR = IMG_DIR / "augmented_cvae"
VAE_LABELS = LABELS_DIR / "augmented_cvae_labels.csv"

# GAN path variables
GAN_RESULTS_DIR = RESULTS_DIR / "models" / "gan"
GAN_MODELS_DIR = GAN_RESULTS_DIR / "models"
GAN_PLOTS_DIR = GAN_RESULTS_DIR / "plots"
GAN_AUGMENTED_DIR = IMG_DIR / "augmented_dcgan"
GAN_LABELS = LABELS_DIR / "augmented_dcgan_labels.csv"

# Diffusion model path variables
DIFF_RESULTS_DIR = RESULTS_DIR / "models" / "diffusion_model"
DIFF_MODELS_DIR = DIFF_RESULTS_DIR / "models"
DIFF_PLOTS_DIR = DIFF_RESULTS_DIR / "plots"
DIFF_AUGMENTED_DIR = IMG_DIR / "augmented_diff"
DIFF_LABELS = LABELS_DIR / "augmented_diff_labels.csv"

# Dataset variables
TEST_SIZE = 0.15 # 15% para teste
VAL_SIZE = 0.15  # 15% para validação

# Training variables
BATCH_SIZE = 32
IMAGE_SIZE = 64

RANDOM_SEED = 42
N_EPOCHS = 50
SAMPLES_PER_CLASS = 40