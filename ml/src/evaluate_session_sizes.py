
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
MODEL_PATH = Path("../models/recall_rf1.joblib")

print("Loading dataset and saved RF-1...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=1_000_000)

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

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

test["prediction"] = np.clip(
    model.predict(test[features].to_numpy(dtype=float)), 0, 1
)

test["session_group"] = np.where(
    test["session_seen"] == 1,
    "One-item sessions",
    "Multiple-item sessions",
)

print("\nRF-1 performance by session size")
print("=" * 50)

for label in ["One-item sessions", "Multiple-item sessions"]:
    group = test[test["session_group"] == label]
    if group.empty:
        continue

    actual = group["target_recall"]
    predicted = group["prediction"]

    print(f"\n{label}")
    print("Rows:", len(group))
    print("Mean actual recall:", round(actual.mean(), 4))
    print("Mean predicted recall:", round(predicted.mean(), 4))
    print("MAE:", round(mean_absolute_error(actual, predicted), 6))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 6))
    print("Actual target values:")
    print(actual.value_counts(normalize=True).sort_index().round(4).to_string())

print("\nSession-size evaluation complete.")
