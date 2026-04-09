# Path variables
from pathlib import Path

# Dataset variables
IMAGE_SIZE = 224
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
LOSS_FUNCTION = "CrossEntropy" 

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

rnet_results_folder_name = f"RNET_{LOSS_FUNCTION}_{OPTIM}"
rnet_results_path = Path().cwd() / ".." / "results" / "resnet" / "no_augmentation" / rnet_results_folder_name
rnet_results_path_augmented = Path().cwd() / ".." / "results" / "resnet" / "augmented" / rnet_results_folder_name

# Generate submissions variables
TEST_IMG_DIR = images_path / 'test' 
MODEL_SUBMISSION_PATH = Path().cwd() / ".." / "results" / "resnet_improved" / "augmented" / "RNET_CrossEntropy_SGD" 

# Caminho do ficheiro .pt
MODEL_TO_LOAD_PATH = MODEL_SUBMISSION_PATH / "best.pt" 

# Onde guardar o CSV final para o Kaggle
SUBMISSION_CSV_PATH = MODEL_SUBMISSION_PATH / "submission.csv"