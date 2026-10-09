from pathlib import Path

# Caminhos dos Dados
PROJECT_ROOT = Path().cwd()
RAW_DATA_PATH = PROJECT_ROOT / "assets" / "appointments_dataset.csv"
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = RESULTS_PATH / "models"
MATRICES_PATH = RESULTS_PATH / "confusion_matrices"


# Variável de controlo do pipeline
RUN_TRANSFORMATION = True

# Caminho para o ficheiro processado
PROCESSED_FILE_PATH = DATA_PATH / "processed_appointments.csv"

IMPORTANT_COLUMNS = [
    'PatientId',
    #'AppointmentID',
    'Gender',
    'ScheduledDay', # 2016-04-29T16:08:27Z (podemos extrair mes, dia, hora,...)
    'AppointmentDay', 
    'Age',
    'Neighbourhood',
    'Scholarship',
    'Hipertension',
    'Diabetes',
    'Alcoholism',
    'Handcap',
    'SMS_received',
    'No-show'
    ]

# Training model variables
modelos = [
    'LogisticRegression', 
    'KNN', 
    'RandomForest', 
    'XGBoost', 
    'LightGBM'
]

TEST_SIZE = 0.2

NUMBER_OF_SEEDS = 10
OPTUNA_TRAIN_STEPS = 30
