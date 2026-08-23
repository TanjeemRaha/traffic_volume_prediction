"""Project path definitions and directory initialization."""

# Code by Tanjeem Farhana Raha

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_GZIP_PATH = RAW_DATA_DIR / "Metro_Interstate_Traffic_Volume.csv.gz"
RAW_CSV_PATH = RAW_DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "i94_traffic_cleaned.csv"
QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_traffic_model.joblib"


def create_project_directories() -> None:
    """Create required project directories."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
