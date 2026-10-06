from pathlib import Path

# Caminhos dos Dados
PROJECT_ROOT = Path().cwd()
RAW_DATA_PATH = PROJECT_ROOT / "assets" / "appointments_dataset.csv"
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"

IMPORTANT_COLUMNS = [
    'PatientId',
    'AppointmentID',
    'Gender',
    'ScheduledDay',
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
'''
Informações importantes que podem ser relevantes serem extraídas:
- 
'''