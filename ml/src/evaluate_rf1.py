
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"

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
    nrows=500000
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
].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train_df = df[df["timestamp"] <= split1]
validation_df = df[
    (df["timestamp"] > split1)
    & (df["timestamp"] <= split2)
]

X_train = train_df[FEATURES].to_numpy(dtype=float)
y_train = train_df["target_recall"].to_numpy(dtype=float)

X_val = validation_df[FEATURES].to_numpy(dtype=float)
y_val = validation_df["target_recall"].to_numpy(dtype=float)

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

print("\nTraining RF-1...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = np.clip(model.predict(X_val), 0, 1)

print("\nValidation metrics")
print("------------------")
print("MAE :", mean_absolute_error(y_val, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_val, predictions)))
print("Mean prediction:", predictions.mean())
print("Mean actual     :", y_val.mean())

print("\nPrediction distribution")
print("-----------------------")
print(pd.Series(predictions).describe())

print("\nReliability analysis")
print("--------------------")
print("Range       Count   Mean predicted   Mean actual")

edges = np.linspace(0, 1, 11)

for i in range(len(edges) - 1):
    lower = edges[i]
    upper = edges[i + 1]

    if i == len(edges) - 2:
        mask = (predictions >= lower) & (predictions <= upper)
    else:
        mask = (predictions >= lower) & (predictions < upper)

    if mask.sum() > 0:
        print(
            f"{lower:.1f}-{upper:.1f}   "
            f"{mask.sum():5d}   "
            f"{predictions[mask].mean():.4f}           "
            f"{y_val[mask].mean():.4f}"
        )

print("\nFeature importance")
print("------------------")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance.to_string())
