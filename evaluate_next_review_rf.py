
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "ml" / "data" / "processed" / "next_review_candidates.csv"
MODEL_PATH = ROOT / "ml" / "models" / "next_review_rf_experiment.joblib"

SAMPLE_SIZE = 200_000
RANDOM_STATE = 42

print("Loading experimental model...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

print("Loading a sample of candidate rows...")
df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "history_seen",
        "history_correct",
        "session_seen",
        "session_correct",
        "elapsed_days",
        "next_gap_days",
        "next_recall",
    ],
)

# Build features using information available after the current session.
df["updated_seen"] = df["history_seen"] + df["session_seen"]
df["updated_correct"] = df["history_correct"] + df["session_correct"]

df["historical_accuracy"] = (
    df["updated_correct"] / df["updated_seen"]
)
df["log_history_seen"] = np.log1p(df["updated_seen"])
df["log_elapsed_days"] = np.log1p(df["elapsed_days"])
df["log_next_gap_days"] = np.log1p(df["next_gap_days"])

df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=features + ["next_recall", "next_gap_days"]
)

df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE,
)

X = df[features].to_numpy(dtype=float)
y = df["next_recall"].to_numpy(dtype=float)
pred = np.clip(model.predict(X), 0, 1)

df["prediction"] = pred
df["absolute_error"] = np.abs(y - pred)

print("\nOVERALL SAMPLE EVALUATION")
print("=" * 55)
print("Rows:", len(df))
print("MAE:", round(mean_absolute_error(y, pred), 6))
print("RMSE:", round(np.sqrt(mean_squared_error(y, pred)), 6))
print("Actual mean:", round(float(y.mean()), 4))
print("Predicted mean:", round(float(pred.mean()), 4))

df["gap_group"] = pd.cut(
    df["next_gap_days"],
    bins=[0, 1/24, 1, 3, 7, 14, np.inf],
    labels=[
        "Under 1 hour",
        "1 hour to 1 day",
        "1 to 3 days",
        "3 to 7 days",
        "7 to 14 days",
        "Over 14 days",
    ],
    include_lowest=True,
)

print("\nPERFORMANCE BY OBSERVED NEXT-SESSION GAP")
print("=" * 75)

for group, part in df.groupby("gap_group", observed=True):
    if len(part) == 0:
        continue

    mae = part["absolute_error"].mean()
    rmse = np.sqrt(
        np.mean((part["next_recall"] - part["prediction"]) ** 2)
    )

    print(
        f"{str(group):<20} "
        f"n={len(part):>7} "
        f"actual={part['next_recall'].mean():.4f} "
        f"predicted={part['prediction'].mean():.4f} "
        f"MAE={mae:.6f} "
        f"RMSE={rmse:.6f}"
    )

print("\nDiagnostic complete. No model files were changed.")
print("This random sample is diagnostic, not a new chronological holdout.")
