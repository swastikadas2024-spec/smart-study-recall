
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
MODEL_PATH = Path("../models/recall_rf1.joblib")
NROWS = 1_000_000

print("Loading data and saved model...")
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

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

test["prediction"] = np.clip(
    model.predict(test[features].to_numpy(dtype=float)), 0, 1
)

print("\nFEATURE IMPORTANCE")
print("-" * 45)

if hasattr(model, "feature_importances_"):
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(importance.round(4).to_string())
else:
    print("Feature importance is not available.")

print("\nLOW-RECALL VS HIGH-RECALL INPUTS")
print("-" * 45)

test["actual_range"] = np.where(
    test["target_recall"] <= 0.4, "Low recall",
    np.where(test["target_recall"] <= 0.7, "Medium recall", "High recall")
)

columns = features + ["target_recall", "prediction"]

for label in ["Low recall", "Medium recall", "High recall"]:
    group = test[test["actual_range"] == label]
    print(f"\n{label} (n={len(group)})")
    print(group[columns].mean().round(4).to_string())

print("\nDiagnostic complete.")
