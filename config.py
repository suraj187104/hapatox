"""
Configuration for Hepatotoxicity Prediction Web Application
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Model paths (local models for teacher demo)
MODELS_DIR = BASE_DIR / "models_local"
GAHT_MODEL_PATH = None  # Disabled (requires torch)
RF_MODEL_PATH = MODELS_DIR / "rf_fold_0.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_fold_0.pkl"

# Data paths (full dataset)
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "combined_tox21_hepatotoxicity.csv"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "submission_workspace" / "results"
METRICS_PATH = RESULTS_DIR / "metrics" / "final_results_summary.json"
STATS_PATH = RESULTS_DIR / "metrics" / "statistical_tests.json"

# Web application settings
APP_NAME = "HepatoTox Predictor"
APP_VERSION = "1.0.0"
SECRET_KEY = "hepatotox-research-2025-secure-key"  # Change in production

# Model settings
USE_ENSEMBLE = False  # Set True to use all 5 folds (slower but more accurate)
DEVICE = "cpu"  # Change to "cuda" if GPU available

# Feature extraction settings
ECFP_RADIUS = 2
ECFP_BITS = 2048
MAX_CONFORMERS = 1

# UI settings
MOLECULES_PER_PAGE = 20
MAX_BATCH_SIZE = 100

# Flask settings
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000
