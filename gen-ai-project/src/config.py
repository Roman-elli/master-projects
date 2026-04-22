from pathlib import Path

# Caminhos dos Dados
PROJECT_ROOT = Path().cwd() / ".."

KAGGLE_ROOT = PROJECT_ROOT / "data" / "archive"
EXPORT_ROOT  = PROJECT_ROOT / "data"
CSV_PATH_20 = PROJECT_ROOT / "assets" / "training_20_percent.csv"

# Dataset setup
INDEX_COLUMN = 'train_id_original' #warning if using colab kernel on vscode you need to put the files on your google drive and link this notebook to it.
TRAIN_FRACTION = 1.0  # Example: 0.5 means half of train split
ONLY_20 = True
IMAGE_SIZE = 32
NUM_WORKERS = 2

# Hiperparâmetros gerais
SEED = 42 # 42 -- 123
BATCH_SIZE = 64

# MODEL VARIABLES
N_EPOCHS = 1
WEIGHT_DECAY = 1e-4

# VAE
VAE_BETA = 0.7
VAE_LR = 1e-3
VAE_LATENT_DIM = 16 # 16 -- 128
VAE_OPTIM = "ADAM" # ADAM -- ADAMW

# GAN
GAN_LATENT_DIM = 100  # 16 -- 100
GAN_LR = 2e-4
GAN_BETA1 = 0.5       # 0.5 -- 0.9

# PixelUNit
DIFF_CHANNELS = 64     # 32, 64, 128
DIFF_TIMESTEPS = 1000  # 500, 1000
DIFF_LR = 2e-4         # 1e-4, 2e-4

GAN_BETA_TEXT = "5" if GAN_BETA1 == 0.5 else "9"

# Pastas específicas por arquitetura
RESULTS_BASE_DIR = PROJECT_ROOT / ("results" if ONLY_20 else "results_best")
PASTA_REAIS_FID = PROJECT_ROOT / "data" / "real_fid_samples"

VAE_RESULTS_DIR = RESULTS_BASE_DIR / "vae"
VAE_RESULT_PATH = VAE_RESULTS_DIR / f"run_opt{VAE_OPTIM}_Latent{VAE_LATENT_DIM}_SEED{SEED}"
VAE_PASTA_FID = VAE_RESULT_PATH / "fid_samples"

GAN_RESULTS_DIR = RESULTS_BASE_DIR / "gan"
GAN_RESULT_PATH = GAN_RESULTS_DIR / f"run_beta_{GAN_BETA_TEXT}_Latent{GAN_LATENT_DIM}_SEED{SEED}"
GAN_PASTA_FID = GAN_RESULT_PATH / "fid_samples"

DIFFUSION_RESULTS_DIR = RESULTS_BASE_DIR / "diffusion"
DIFFUSION_RESULT_PATH = DIFFUSION_RESULTS_DIR / f"run_beta_X_Latent_X_SEED{SEED}"
DIFFUSION_PASTA_FID = DIFFUSION_RESULT_PATH / "fid_samples"

CAMINHO_VAE_FID_JSON = VAE_RESULT_PATH / "fid_metrics.json"

CAMINHO_GAN_FID_JSON = GAN_RESULT_PATH / "fid_metrics.json"

CAMINHO_DIFFUSION_FID_JSON = DIFFUSION_RESULT_PATH / "fid_metrics.json"