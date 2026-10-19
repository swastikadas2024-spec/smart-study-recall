
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
NROWS = 1_000_000

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=NROWS)

df["log_history_seen"] = np.log1p(df["history_seen"])
df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"].replace(0, np.nan)
)
df["delta_days"] = df["delta"] / 86400
df["log_delta_days"] = np.log1p(df["delta_days"])
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
cutoff = df["timestamp"].quantile(0.85)
test = df[df["timestamp"] > cutoff].copy()

features = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

test["recall_group"] = np.where(
    test["target_recall"] <= 0.4,
    "Low recall",
    np.where(test["target_recall"] <= 0.7, "Medium recall", "High recall")
)

for label in ["Low recall", "Medium recall", "High recall"]:
    group = test[test["recall_group"] == label]

    print(f"\n{'=' * 50}")
    print(f"{label} — {len(group)} rows")
    print("=" * 50)

    print(
        group[features + ["history_seen", "history_correct",
                          "session_seen", "session_correct"]]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(3)
        .to_string()
    )

print("\nFeature distribution inspection complete.")
