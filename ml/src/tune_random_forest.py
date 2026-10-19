import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATASET_PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"

NROWS = 500000


# ============================================================
# Load data
# ============================================================

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

df["log_delta_days"] = np.log1p(
    df["delta_days"]
)

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


# ============================================================
# Clean
# ============================================================

df = df[
    (df["delta"] > 0) &
    (df["history_seen"] > 0) &
    (df["session_seen"] > 0)
].copy()


# ============================================================
# Chronological split
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train_df = df[
    df["timestamp"] <= split1
]

test_df = df[
    df["timestamp"] > split2
]


# ============================================================
# Features
# ============================================================

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

X_train = train_df[FEATURES].to_numpy()
y_train = train_df["target_recall"].to_numpy()

X_test = test_df[FEATURES].to_numpy()
y_test = test_df["target_recall"].to_numpy()


# ============================================================
# Models to compare
# ============================================================

configs = [
    {
        "name": "RF-1",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 10,
    },
    {
        "name": "RF-2",
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 10,
    },
    {
        "name": "RF-3",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
    },
    {
        "name": "RF-4",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
    },
]


# ============================================================
# Train and evaluate
# ============================================================

results = []

for config in configs:

    print()
    print("Training", config["name"])

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    results.append(
        (
            config["name"],
            mae,
            rmse
        )
    )

    print("MAE :", mae)
    print("RMSE:", rmse)


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("RANDOM FOREST TUNING RESULTS")
print("=" * 60)

for name, mae, rmse in results:

    print(
        f"{name:<8} "
        f"MAE={mae:.6f} "
        f"RMSE={rmse:.6f}"
    )