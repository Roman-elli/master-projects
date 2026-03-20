# Path variables
from pathlib import Path

# Dataset variables
IMAGE_SIZE = 64
TEST_SIZE = 0.2
DATA_AUG_SIZE = 5500
AUGMENT_DATA = True

# Model variables
BATCH_SIZE = 32
N_EPOCHS = 50
HIDDEN_LAYER_SIZES = [1024, 512]
LR = 0.001
#LR = 0.01 # para SGD

# Opções de Loss: "CrossEntropy" | "MultiMarginLoss" | "CrossEntropy_LabelSmoothing"
LOSS_FUNCTION = "CrossEntropy_LabelSmoothing" 

LABEL_SMOOTHING = 0.1 # Utilizado com a loss function CrossEntropy_LabelSmoothing

# Opções de Otimizador: "ADAM" | "RMSprop" | "ADAMW" | "SGD"
OPTIM = "ADAM"

WEIGHT_DECAY = 0.01 # Utilizado com o otimizador ADAMW
MOMENTUM = 0.9 # Utilizado com o otimizador SGD

# Path Variables
images_path = Path().cwd() / ".." / "assets" / "images"
labels_path = Path().cwd() / ".." / "assets" / "labels"
TRAIN_IMG_DIR = images_path / 'train'
TRAIN_LABELS_PATH = labels_path / 'train_labels.csv'

mlp_results_folder_name = f"MLP_{LOSS_FUNCTION}_{OPTIM}"
mlp_results_path = Path().cwd() / ".." / "results" / "mlp" / "no_augmentation" / mlp_results_folder_name
mlp_results_path_augmented = Path().cwd() / ".." / "results" / "mlp" / "augmented" / mlp_results_folder_name

cnn_results_folder_name = f"CNN_{LOSS_FUNCTION}_{OPTIM}"
cnn_results_path = Path().cwd() / ".." / "results" / "cnn" / "no_augmentation" / cnn_results_folder_name
cnn_results_path_augmented = Path().cwd() / ".." / "results" / "cnn" / "augmented" / cnn_results_folder_name