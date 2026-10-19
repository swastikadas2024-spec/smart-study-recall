
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
MODEL_PATH = Path("../models/recall_rf1.joblib")
NROWS = 1_000_000

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=NROWS)

df["delta_days"] = df["delta"] / 86400
df["log_delta_days"] = np.log1p(df["delta_days"])
df["log_history_seen"] = np.log1p(df["history_seen"])
df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"].replace(0, np.nan)
)
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

# Use the final 15% of chronological observations as the test set.
split = df["timestamp"].quantile(0.85)
test = df[df["timestamp"] > split].copy()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

X_test = test[features].to_numpy(dtype=float)
y_test = test["target_recall"].to_numpy(dtype=float)

predictions = np.clip(model.predict(X_test), 0, 1)

test["actual_recall"] = y_test
test["predicted_recall"] = predictions

print("\nSaved RF-1 evaluation by actual recall range")
print("=" * 65)

test["recall_range"] = pd.cut(
    test["actual_recall"],
    bins=[-0.001, 0.4, 0.7, 1.001],
    labels=["Low (0-40%)", "Medium (40-70%)", "High (70-100%)"],
    include_lowest=True,
)

for label, group in test.groupby("recall_range", observed=False):
    if len(group) == 0:
        continue

    mae = mean_absolute_error(
        group["actual_recall"], group["predicted_recall"]
    )
    rmse = np.sqrt(mean_squared_error(
        group["actual_recall"], group["predicted_recall"]
    ))

    bias = (
        group["predicted_recall"] - group["actual_recall"]
    ).mean()

    print(f"\n{label}")
    print("Rows:", len(group))
    print("Actual mean:", round(group["actual_recall"].mean(), 4))
    print("Predicted mean:", round(group["predicted_recall"].mean(), 4))
    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("Prediction bias:", round(bias, 6))

print("\nEvaluation complete.")
print("Note: this evaluates session-level recall, not individual")
print("future-memory probability. It uses a chronological holdout.")
