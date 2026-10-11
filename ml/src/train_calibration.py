import json
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from caliberate_recall import (
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
# Calibration quality for continuous recall targets
raw_calibration_error = np.mean(
    np.abs(test_predictions - actual)
)

calibrated_calibration_error = np.mean(
    np.abs(calibrated_predictions - actual)
)

print()
print("Calibration quality")
print("--------------------")
print("Raw HLR calibration error       :", raw_calibration_error)
print("Calibrated HLR calibration error:", calibrated_calibration_error)

# Calibration quality for continuous recall target
raw_calibration_error = np.mean(
    np.abs(test_predictions - actual)
)

calibrated_calibration_error = np.mean(
    np.abs(calibrated_predictions - actual)
)

print()
print("Calibration quality")
print("--------------------")
print("Raw HLR calibration error       :", raw_calibration_error)
print("Calibrated HLR calibration error:", calibrated_calibration_error)

# Continuous reliability analysis
bins = np.linspace(0, 1, 11)

print()
print("Reliability analysis")
print("--------------------")
print("Bin        Raw Predicted   Raw Actual   Calibrated Predicted   Calibrated Actual")

for i in range(len(bins) - 1):

    lower = bins[i]
    upper = bins[i + 1]

    raw_mask = (
        (test_predictions >= lower) &
        (test_predictions < upper)
    )

    cal_mask = (
        (calibrated_predictions >= lower) &
        (calibrated_predictions < upper)
    )

    if raw_mask.sum() > 0:
        raw_pred_mean = test_predictions[raw_mask].mean()
        raw_actual_mean = actual[raw_mask].mean()
    else:
        raw_pred_mean = np.nan
        raw_actual_mean = np.nan

    if cal_mask.sum() > 0:
        cal_pred_mean = calibrated_predictions[cal_mask].mean()
        cal_actual_mean = actual[cal_mask].mean()
    else:
        cal_pred_mean = np.nan
        cal_actual_mean = np.nan

    print(
        f"{lower:.1f}-{upper:.1f}   "
        f"{raw_pred_mean:.4f}         "
        f"{raw_actual_mean:.4f}       "
        f"{cal_pred_mean:.4f}                 "
        f"{cal_actual_mean:.4f}"
    )