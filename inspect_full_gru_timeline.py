
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT
    / "ml"
    / "data"
    / "raw"
    / "settles.acl16.learning_traces.13m.csv.gz"
)

CHUNK_SIZE = 500_000
total_rows = 0
earliest = None
latest = None
daily_counts = {}

print("Scanning the full dataset in chunks...")

for chunk in pd.read_csv(
    DATA_PATH,
    compression="gzip",
    usecols=["timestamp"],
    chunksize=CHUNK_SIZE,
):
    timestamps = pd.to_numeric(
        chunk["timestamp"], errors="coerce"
    ).dropna()

    if timestamps.empty:
        continue

    total_rows += len(timestamps)

    chunk_min = timestamps.min()
    chunk_max = timestamps.max()

    earliest = chunk_min if earliest is None else min(
        earliest, chunk_min
    )
    latest = chunk_max if latest is None else max(
        latest, chunk_max
    )

    dates = pd.to_datetime(
        timestamps, unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")

    counts = dates.value_counts()
    for date, count in counts.items():
        daily_counts[date] = (
            daily_counts.get(date, 0) + int(count)
        )

print("\nFULL DATASET TIMELINE")
print("=" * 50)
print("Valid timestamp rows:", total_rows)
print(
    "Earliest event:",
    pd.to_datetime(earliest, unit="s", utc=True),
)
print(
    "Latest event:",
    pd.to_datetime(latest, unit="s", utc=True),
)
print("\nRows per date:")

for date, count in sorted(daily_counts.items()):
    print(f"{date}: {count:,}")
