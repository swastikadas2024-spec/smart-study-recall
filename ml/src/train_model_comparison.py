import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"

# Start with 500,000 rows for experimentation.
# We can scale this up after the pipeline works.
NROWS = 500000


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    compression="gzip",
    nrows=NROWS
)

print("Rows loaded:", len(df))


# ============================================================
# Prepare data
# ============================================================

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

df["log_delta_days"] = np.log1p(
    df["delta_days"]
)


# Target:
# current-session recall rate
df["target_recall"] = (
    df["session_correct"] /
    df["session_seen"]
)


# ============================================================
# Remove invalid rows
# ============================================================

df = df[
    (df["delta"] > 0) &
    (df["history_seen"] > 0) &
    (df["session_seen"] > 0)
].copy()


# ============================================================
# Sort chronologically
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# Chronological train / validation / test split
# ============================================================

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


print()
print("Dataset split")
print("-------------")
print("Training rows  :", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows      :", len(test_df))


# ============================================================
# Features
# ============================================================

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]


X_train = train_df[FEATURES].to_numpy(dtype=float)
y_train = train_df["target_recall"].to_numpy(dtype=float)

X_validation = validation_df[FEATURES].to_numpy(dtype=float)
y_validation = validation_df["target_recall"].to_numpy(dtype=float)

X_test = test_df[FEATURES].to_numpy(dtype=float)
y_test = test_df["target_recall"].to_numpy(dtype=float)


# ============================================================
# Helper function
# ============================================================

def evaluate_model(name, model, X, y):
    predictions = model.predict(X)

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    print()
    print(name)
    print("-" * len(name))
    print("MAE :", mae)
    print("RMSE:", rmse)

    return predictions, mae, rmse


# ============================================================
# 1. Linear Regression baseline
# ============================================================

print()
print("Training Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions, linear_mae, linear_rmse = evaluate_model(
    "Linear Regression",
    linear_model,
    X_test,
    y_test
)


# ============================================================
# 2. Random Forest
# ============================================================

print()
print("Training Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions, rf_mae, rf_rmse = evaluate_model(
    "Random Forest",
    rf_model,
    X_test,
    y_test
)


# ============================================================
# 3. Baseline predictor
# ============================================================

baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

print()
print("Mean Baseline")
print("-------------")
print("MAE :", baseline_mae)
print("RMSE:", baseline_rmse)


# ============================================================
# Final comparison
# ============================================================

print()
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    f"Mean Baseline       MAE={baseline_mae:.6f} "
    f"RMSE={baseline_rmse:.6f}"
)

print(
    f"Linear Regression   MAE={linear_mae:.6f} "
    f"RMSE={linear_rmse:.6f}"
)

print(
    f"Random Forest       MAE={rf_mae:.6f} "
    f"RMSE={rf_rmse:.6f}"
)