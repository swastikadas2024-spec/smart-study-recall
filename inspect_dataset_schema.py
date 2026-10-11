
import pandas as pd
from pathlib import Path

# Resolve the dataset path from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "settles.acl16.learning_traces.13m.csv.gz"
)

print("Dataset path:", DATASET_PATH)
print("Dataset exists:", DATASET_PATH.exists())

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Check that the CSV.gz file is inside "
        "ml/data/raw."
    )

print("\nLoading first 10 rows...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=10)

print("\n1. DATASET COLUMNS")
print("=" * 60)
for column in df.columns:
    print(f"{column}: {df[column].dtype}")

print("\n2. FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string(index=False))

print("\n3. IMPORTANT COLUMN CHECK")
print("=" * 60)
important_columns = [
    "user_id", "lexeme_id", "timestamp", "delta", "p_recall",
    "history_seen", "history_correct", "session_seen", "session_correct",
]

for column in important_columns:
    print(f"{column}: {'FOUND' if column in df.columns else 'MISSING'}")

print("\nInspection complete. No files were modified.")
