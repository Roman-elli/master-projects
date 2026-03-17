# Path variables
from pathlib import Path
import os

images_path = Path().cwd()  / "../assets/images"
labels_path = Path().cwd()  / "../assets/labels"

TRAIN_IMG_DIR = os.path.join(images_path, 'train')
TRAIN_LABELS_PATH = os.path.join(labels_path, 'train_labels.csv')


# Dataset variables
BATCH_SIZE = 32
IMAGE_SIZE = 64

# Model variables
## MLP ##
MLP_EPOCHS = 50
MLP_HIDDEN_LAYER_SIZES = [1024, 512]
MLP_LR = 0.001
MLP_LOSS_FUNCTION = "MultiMarginLoss" #  "CrossEntropy"
MLP_OPTIM = "RMSprop" # "ADAM"

## CNN ##

## ResNet ##
