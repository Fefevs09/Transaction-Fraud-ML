"""Centralized constants: URLs/paths, feature-column lists, and pipeline seeds shared across the project."""

from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2

TRAIN_DATA_URL = "https://raw.githubusercontent.com/Fefevs09/Transaction-Fraud-ML/main/data/treino.csv"
HOLDOUT_DATA_URL = "https://raw.githubusercontent.com/Fefevs09/Transaction-Fraud-ML/main/data/teste.csv"

CSV_SEPARATOR = "|"
CSV_ENCODING = "utf-8"

TARGET_COLUMN = "is_fraud"

# PII and high-cardinality columns that add no modeling signal.
PII_AND_HIGH_CARDINALITY_COLUMNS = [
    "ssn",
    "cc_num",
    "first",
    "last",
    "street",
    "zip",
    "unix_time",
    "lat",
    "long",
    "acct_num",
    "trans_num",
    "merchant",
    "merch_lat",
    "merch_long",
]

CATEGORICAL_COLUMNS = ["gender", "job", "profile", "category", "city", "state"]
UNKNOWN_CATEGORY_VALUE = -1

# Anchor output paths to this package's location so they resolve correctly
# regardless of the caller's working directory (script vs. notebook kernel).
REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_OUTPUT_PATH = REPO_ROOT / "reports" / "metrics.json"
SUBMISSION_OUTPUT_PATH = REPO_ROOT / "data" / "submissao.csv"
