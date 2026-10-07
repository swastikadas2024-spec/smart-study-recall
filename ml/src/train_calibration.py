import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from calibrate_recall import (
    train_calibrator,
    get_hlr_predictions,
)


# -----------------------------
# 1. Load existing HLR model
# -----------------------------

model_path = "../models/hlr_parameters.json"

with open(model_path, "r") as f:
    model_data = json.load(f)

params = np.array(
    model_data["parameters"],
    dtype=float
)


# -----------------------------
# 2. Load dataset
# -----------------------------

dataset_path = (
    "../data/raw/"
    "settles.acl16.learning_traces.13m.csv.gz"
)

df = pd.read_csv(
    dataset_path,
    compression="gzip",
    nrows=500000
)


# -----------------------------
# 3. Prepare data
# -----------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s"
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

df["delta_days"] = (
    df["delta"] / (60 * 60 * 24)
)

df = df[
    (df["delta"] > 0) &
    (df["history_seen"] > 0) &
    (df["p_recall"] >= 0) &
    (df["p_recall"] <= 1)
].copy()


# -----------------------------
# 4. Temporal split
# -----------------------------

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train_df = df[
    df["timestamp"] <= split1
].copy()

validation_df = df[
    (df["timestamp"] > split1) &
    (df["timestamp"] <= split2)
].copy()

test_df = df[
    df["timestamp"] > split2
].copy()


print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))


# -----------------------------
# 5. Get raw HLR predictions
# -----------------------------

validation_predictions = get_hlr_predictions(
    params,
    validation_df
)


# -----------------------------
# 6. Train calibration layer
# -----------------------------

calibrator = train_calibrator(
    params,
    validation_df
)


# -----------------------------
# 7. Evaluate on unseen test data
# -----------------------------

test_predictions = get_hlr_predictions(
    params,
    test_df
)

calibrated_predictions = calibrator.predict(
    test_predictions
)

actual = test_df[
    "p_recall"
].to_numpy(dtype=float)


raw_mae = mean_absolute_error(
    actual,
    test_predictions
)

raw_rmse = np.sqrt(
    mean_squared_error(
        actual,
        test_predictions
    )
)

calibrated_mae = mean_absolute_error(
    actual,
    calibrated_predictions
)

calibrated_rmse = np.sqrt(
    mean_squared_error(
        actual,
        calibrated_predictions
    )
)


print()
print("HLR before calibration")
print("----------------------")
print("MAE :", raw_mae)
print("RMSE:", raw_rmse)

print()
print("HLR after calibration")
print("---------------------")
print("MAE :", calibrated_mae)
print("RMSE:", calibrated_rmse)