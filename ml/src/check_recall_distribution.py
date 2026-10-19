
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
NROWS = 1_000_000

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=NROWS)

df["target_recall"] = (
    df["session_correct"] / df["session_seen"].replace(0, np.nan)
)

df = df[
    (df["delta"] > 0)
    & (df["history_seen"] > 0)
    & (df["session_seen"] > 0)
    & (df["history_correct"] >= 0)
    & (df["history_correct"] <= df["history_seen"])
    & (df["session_correct"] >= 0)
    & (df["session_correct"] <= df["session_seen"])
].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train = df[df["timestamp"] <= split1].copy()
test = df[df["timestamp"] > split2].copy()

bins = [-0.001, 0.4, 0.7, 1.001]
labels = ["Low (0-40%)", "Medium (40-70%)", "High (70-100%)"]

for name, part in [("TRAIN", train), ("TEST", test)]:
    part["recall_range"] = pd.cut(
        part["target_recall"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    print(f"\n{name} DISTRIBUTION")
    print("-" * 45)
    print("Rows:", len(part))
    print("Mean recall:", round(part["target_recall"].mean(), 4))
    print("Median recall:", round(part["target_recall"].median(), 4))
    print("Recall standard deviation:", round(part["target_recall"].std(), 4))
    print("\nRecall ranges:")
    print(part["recall_range"].value_counts().sort_index())
    print("\nRecall range percentages:")
    print(
        (part["recall_range"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .astype(str) + "%"
    )

print("\nDistribution check complete.")
