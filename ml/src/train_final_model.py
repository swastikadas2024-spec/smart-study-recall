
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

DATASET_PATH = Path("../data/raw/settles.acl16.learning_traces.13m.csv.gz")
MODEL_DIR = Path("../models")
MODEL_PATH = MODEL_DIR / "recall_rf1.joblib"
METADATA_PATH = MODEL_DIR / "recall_rf1_metadata.json"

NROWS = 1_000_000

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    compression="gzip",
    nrows=NROWS,
)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df["delta_days"] = df["delta"] / 86400
df["log_delta_days"] = np.log1p(df["delta_days"])
df["log_history_seen"] = np.log1p(df["history_seen"])

df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"]
)

df["target_recall"] = (
    df["session_correct"] / df["session_seen"]
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

# Reserve the final chronological 15% from this training run.
# The model is trained on the first 85%.
cutoff = df["timestamp"].quantile(0.85)
train_df = df[df["timestamp"] <= cutoff].copy()

X_train = train_df[FEATURES].to_numpy(dtype=float)
y_train = train_df["target_recall"].to_numpy(dtype=float)

print("Training rows:", len(train_df))
print("Training RF-1...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Store both the fitted model and its required feature order.
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "target": "session_correct / session_seen",
        "target_type": "session-level recall rate",
    },
    MODEL_PATH,
)

metadata = {
    "model_name": "Random Forest RF-1",
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 10,
    "random_state": 42,
    "training_rows": len(train_df),
    "dataset_rows_loaded": NROWS,
    "training_cutoff": str(cutoff),
    "features": FEATURES,
    "target": "session_correct / session_seen",
    "target_type": "session-level recall rate",
    "calibrated": False,
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved:", MODEL_PATH.resolve())
print("Metadata saved:", METADATA_PATH.resolve())
print("Feature importance:")
print(
    pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False).to_string()
)
