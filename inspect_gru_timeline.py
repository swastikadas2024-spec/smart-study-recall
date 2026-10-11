
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

df = pd.read_csv(
    DATA_PATH,
    compression="gzip",
    nrows=200_000,
    usecols=["timestamp"],
)

df["timestamp"] = pd.to_numeric(
    df["timestamp"], errors="coerce"
)
df = df.dropna(subset=["timestamp"])

df["date"] = pd.to_datetime(
    df["timestamp"], unit="s", utc=True
).dt.date

print("Rows:", len(df))
print("Earliest event:", pd.to_datetime(
    df["timestamp"].min(), unit="s", utc=True
))
print("Latest event:", pd.to_datetime(
    df["timestamp"].max(), unit="s", utc=True
))
print("\nRows per date:")
print(df["date"].value_counts().sort_index())
