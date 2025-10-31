# Files & folders path
ASSETS_FOLDERS_PATH = "assets/"
BASE_DATA_PATH = "data/features"

PCA_THRESHOLD = 0.75
LABELS_COUNTER = 16

SENSORS = ['acc', 'gyro']
ACTIVITIES = range(1, 17)
SENSOR_COLS = {
    'acc': [1, 2, 3],
    'gyro': [4, 5, 6],
    'mag': [7, 8, 9]
}


BODY_PARTS_PATH = ["Left_Wrist", "Right_Wrist", "Chest", "Right_Upper_Leg", "Left_Lower_Leg"]
BODY_PARTS = ["Left Wrist", "Right Wrist", "Chest", "Right Upper Leg", "Left Lower Leg"]

# Configurações de janela
WINDOW_SIZE = 2000
OVERLAP = 0.5
FS = 51.2
