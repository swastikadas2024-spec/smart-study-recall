
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

print("Loading first 200,000 rows...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=200_000)
df["source_order"] = range(len(df))

df = df.sort_values(
    ["user_id", "lexeme_id", "timestamp", "source_order"]
).reset_index(drop=True)

grouped = df.groupby(["user_id", "lexeme_id"], sort=False)
df["next_timestamp"] = grouped["timestamp"].shift(-1)
df["gap_seconds"] = df["next_timestamp"] - df["timestamp"]

gaps = df[
    df["next_timestamp"].notna()
    & (df["gap_seconds"] > 0)
].copy()

gaps["gap_hours"] = gaps["gap_seconds"] / 3600
gaps["gap_days"] = gaps["gap_seconds"] / 86400

print("\nPOSITIVE GAP DISTRIBUTION")
print("=" * 55)
print(
    gaps["gap_hours"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    .round(3)
    .to_string()
)

print("\nGAP INTERVAL COUNTS")
print("=" * 55)

bins = [0, 1/60, 1, 24, 72, 168, float("inf")]
labels = [
    "Under 1 minute",
    "1 minute to 1 hour",
    "1 to 24 hours",
    "1 to 3 days",
    "3 to 7 days",
    "7+ days",
]

gaps["gap_group"] = pd.cut(
    gaps["gap_hours"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

print(gaps["gap_group"].value_counts().reindex(labels).to_string())

print("\nGAP INTERVAL PERCENTAGES")
print("=" * 55)
print(
    (gaps["gap_group"].value_counts(normalize=True).reindex(labels) * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

print("\nGap inspection complete. No files were modified.")
