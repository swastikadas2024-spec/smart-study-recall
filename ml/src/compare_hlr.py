import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from hlr import predict_hlr


DATASET_PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"
MODEL_PATH = "../models/hlr_parameters.json"

NROWS = 500000


# Load model
with open(MODEL_PATH, "r") as f:
    model_data = json.load(f)

params = np.array(
    model_data["parameters"],
    dtype=float
)


# Load data
print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    compression="gzip",
    nrows=NROWS
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s"
)

df["delta_days"] = df["delta"] / (60 * 60 * 24)

df["log_history_seen"] = np.log1p(
    df["history_seen"]
)

df["historical_accuracy"] = (
    df["history_correct"] /
    df["history_seen"]
)

df["target_recall"] = (
    df["session_correct"] /
    df["session_seen"]
)


# Remove invalid rows
df = df[
    (df["delta"] > 0) &
    (df["history_seen"] > 0) &
    (df["session_seen"] > 0)
].copy()


# Chronological ordering
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# Same split as previous experiment
split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

test_df = df[
    df["timestamp"] > split2
].copy()


# HLR features
X_test = test_df[
    ["log_history_seen", "historical_accuracy"]
].to_numpy(dtype=float)

t_test = test_df[
    "delta_days"
].to_numpy(dtype=float)

actual = test_df[
    "target_recall"
].to_numpy(dtype=float)


# HLR prediction
predictions = predict_hlr(
    params,
    X_test,
    t_test
)

predictions = np.clip(
    predictions,
    0.0,
    1.0
)


# Metrics
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


print()
print("HLR Evaluation")
print("--------------")
print("Test rows:", len(test_df))
print("MAE :", mae)
print("RMSE:", rmse)


# Prediction distribution
print()
print("Prediction distribution")
print("-----------------------")
print(
    pd.Series(predictions).describe()
)

print()
print("Actual target distribution")
print("--------------------------")
print(
    pd.Series(actual).describe()
)