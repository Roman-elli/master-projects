# src/config.py
from pathlib import Path

# Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent 

ROOT_DIR = CURRENT_DIR.parent

# pasta de assets
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
