
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"
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

print("Rows loaded:", len(df))

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

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train = df[df["timestamp"] <= split1]
validation = df[
    (df["timestamp"] > split1)
    & (df["timestamp"] <= split2)
]
test = df[df["timestamp"] > split2]

print("\nChronological split")
print("-------------------")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

X_train = train[FEATURES].to_numpy(dtype=float)
y_train = train["target_recall"].to_numpy(dtype=float)

X_test = test[FEATURES].to_numpy(dtype=float)
y_test = test["target_recall"].to_numpy(dtype=float)

models = {
    "Linear Regression": LinearRegression(),
    "RF-1": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

baseline = np.full(len(y_test), y_train.mean())

mae = mean_absolute_error(y_test, baseline)
rmse = np.sqrt(mean_squared_error(y_test, baseline))
results.append(("Mean baseline", mae, rmse))

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, 1)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append((name, mae, rmse))

    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))

    if name == "RF-1":
        print("Mean prediction:", round(predictions.mean(), 6))
        print("Mean actual:", round(y_test.mean(), 6))
        print("\nPrediction distribution:")
        print(pd.Series(predictions).describe())

print("\n" + "=" * 55)
print("FINAL HOLDOUT COMPARISON")
print("=" * 55)

for name, mae, rmse in results:
    print(f"{name:<20} MAE={mae:.6f} RMSE={rmse:.6f}")

print(
    "\nNote: this is a larger-sample holdout evaluation, "
    "not a completely independent external validation."
)
