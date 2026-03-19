# Path variables
from pathlib import Path
import os

# Dataset variables
BATCH_SIZE = 32
IMAGE_SIZE = 64
TEST_SIZE = 0.2
DATA_AUG_SIZE = 5500

# Model variables

## MLP ##
MLP_EPOCHS = 50
MLP_HIDDEN_LAYER_SIZES = [1024, 512]
MLP_LR = 0.001

# Opções de Loss: "CrossEntropy" | "MultiMarginLoss" | "CrossEntropy_LabelSmoothing"
MLP_LOSS_FUNCTION = "CrossEntropy" 

MLP_LABEL_SMOOTHING = 0.1 # Utilizado com a loss function CrossEntropy_LabelSmoothing

# Opções de Otimizador: "ADAM" | "RMSprop" | "ADAMW" | "SGD"
MLP_OPTIM = "ADAM"

MLP_WEIGHT_DECAY = 0.01 # Utilizado com o otimizador ADAMW
MLP_MOMENTUM = 0.9 # Utilizado com o otimizador SGD


## CNN ##
CNN_EPOCHS = 50
CNN_HIDDEN_LAYER_SIZES = [1024, 512]
CNN_LR = 0.001
#CNN_LR = 0.01 para SGD

# Opções de Loss: "CrossEntropy" | "MultiMarginLoss" | "CrossEntropy_LabelSmoothing"
CNN_LOSS_FUNCTION = "CrossEntropy" 

CNN_LABEL_SMOOTHING = 0.1 # Utilizado com a loss function CrossEntropy_LabelSmoothing

# Opções de Otimizador: "ADAM" | "RMSprop" | "ADAMW" | "SGD"
CNN_OPTIM = "SGD"

CNN_WEIGHT_DECAY = 0.01 # Utilizado com o otimizador ADAMW
CNN_MOMENTUM = 0.9 # Utilizado com o otimizador SGD


## ResNet ##







# Path Variables
images_path = Path().cwd() / ".." / "assets" / "images"
labels_path = Path().cwd() / ".." / "assets" / "labels"

mlp_results_folder_name = f"MLP_Ep{MLP_EPOCHS}_{MLP_LOSS_FUNCTION}_{MLP_OPTIM}"
mlp_results_path = Path().cwd() / ".." / "results" / "MLP" / mlp_results_folder_name

cnn_results_folder_name = f"CNN_Ep{CNN_EPOCHS}_{CNN_LOSS_FUNCTION}_{CNN_OPTIM}"
cnn_results_path = Path().cwd() / ".." / "results" / "CNN" / cnn_results_folder_name

TRAIN_IMG_DIR = images_path / 'train'
TRAIN_LABELS_PATH = labels_path / 'train_labels.csv'