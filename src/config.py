# paths + settings shared by every sprint script

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
# small files the dashboard needs. these DO go in the repo so the live app
# doesn't have to download 1GB+ of complaints every time it wakes up
APP_DIR = ROOT / "app_data"

ARCHIVE_DIR = RAW_DIR / "cfpb_archive"
SAMPLE_FILE = PROCESSED_DIR / "complaints_sample.parquet"
MONTHLY_FILE = PROCESSED_DIR / "monthly_counts.csv"
ENRICHED_FILE = PROCESSED_DIR / "complaints_enriched.parquet"

# the CFPB stopped publishing complaint narratives in august 2026, so the
# official download has no text anymore. this is a parquet copy made before
# that, with the narratives still in it (~1GB, 67 files)
HF_ARCHIVE = "Mouwiya/cfpb-consumer-complaints"

# two full recent years. product names on the CFPB site shift over time,
# so a fixed window keeps the categories stable
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# credit reporting is ~88% of all complaints. capping each product keeps
# the classifier from just learning "always guess credit reporting"
MAX_PER_PRODUCT = 5000

RANDOM_STATE = 42

for folder in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, APP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
