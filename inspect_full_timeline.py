
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "settles.acl16.learning_traces.13m.csv.gz"
)

print("Reading the full dataset timestamp column...")
df = pd.read_csv(
    DATASET_PATH,
    compression="gzip",
    usecols=["timestamp"],
)

timestamps = pd.to_datetime(df["timestamp"], unit="s")

print("\nFULL DATASET TIMELINE")
print("=" * 55)
print("Total rows:", len(df))
print("Earliest timestamp:", timestamps.min())
print("Latest timestamp:", timestamps.max())
print("Unique timestamps:", timestamps.nunique())
print("Timeline span:", timestamps.max() - timestamps.min())

print("\nYEARLY ROW COUNTS")
print("=" * 55)
print(timestamps.dt.year.value_counts().sort_index().to_string())

print("\nMONTHLY ROW COUNTS")
print("=" * 55)
print(
    timestamps.dt.to_period("M")
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTimeline inspection complete. No files were modified.")
