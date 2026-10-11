
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ml" / "data" / "processed" / "next_review_candidates.csv"
MODEL_DIR = ROOT / "ml" / "models"
OUTPUT_PATH = MODEL_DIR / "next_review_rf_experiment.joblib"

# Keep this experiment manageable; sample across the entire timeline.
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. Choose a new output filename."
    )

print("Loading candidate data...")
df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "timestamp",
        "next_timestamp",
        "elapsed_days",
        "history_seen",
        "history_correct",
        "session_seen",
        "session_correct",
        "next_recall",
        "next_gap_days",
    ],
)

# Create features using only information available after the current session.
df["updated_history_seen"] = df["history_seen"] + df["session_seen"]
df["updated_history_correct"] = (
    df["history_correct"] + df["session_correct"]
)
df["historical_accuracy"] = (
    df["updated_history_correct"] / df["updated_history_seen"]
)
df["log_history_seen"] = np.log1p(df["updated_history_seen"])
df["log_elapsed_days"] = np.log1p(df["elapsed_days"])
df["log_next_gap_days"] = np.log1p(df["next_gap_days"])

df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "timestamp",
        "next_timestamp",
        "historical_accuracy",
        "log_history_seen",
        "elapsed_days",
        "log_elapsed_days",
        "next_gap_days",
        "log_next_gap_days",
        "next_recall",
    ]
)

# Sample across the full timeline, not just the first rows of the file.
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

df = df.sort_values("next_timestamp").reset_index(drop=True)

# Chronological 70/15/15 split.
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

FEATURES = [
    "historical_accuracy",
    "log_history_seen",
    "elapsed_days",
    "log_elapsed_days",
    "next_gap_days",
    "log_next_gap_days",
]

X_train = train[FEATURES].to_numpy(dtype=float)
y_train = train["next_recall"].to_numpy(dtype=float)
X_test = test[FEATURES].to_numpy(dtype=float)
y_test = test["next_recall"].to_numpy(dtype=float)

print("\nDATA SPLIT")
print("=" * 55)
print("Sample rows:", n)
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
print("Train target mean:", round(y_train.mean(), 4))
print("Test target mean:", round(y_test.mean(), 4))

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

results = []

# Mean baseline learned from training data only.
baseline_predictions = np.full(len(y_test), y_train.mean())
results.append((
    "Mean baseline",
    mean_absolute_error(y_test, baseline_predictions),
    np.sqrt(mean_squared_error(y_test, baseline_predictions)),
))

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_test), 0, 1)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results.append((name, mae, rmse))

    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))

    if name == "Random Forest":
        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "target": "next session recall rate",
                "target_type": "next-observation session-level recall",
                "uses_planned_review_gap": True,
                "calibrated": False,
            },
            OUTPUT_PATH,
        )
        print("Experimental model saved to:", OUTPUT_PATH)

print("\n" + "=" * 60)
print("NEXT-REVIEW HOLDOUT COMPARISON")
print("=" * 60)

for name, mae, rmse in results:
    print(f"{name:<22} MAE={mae:.6f} RMSE={rmse:.6f}")

print("\nThis is a preliminary experiment, not external validation.")
print("The saved experimental model does not replace RF-1.")
