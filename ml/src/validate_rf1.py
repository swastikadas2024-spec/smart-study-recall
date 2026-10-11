
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH = "../data/raw/settles.acl16.learning_traces.13m.csv.gz"

df = pd.read_csv(PATH, compression="gzip", nrows=500000)

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

train = df[df["timestamp"] <= split1]
val = df[
    (df["timestamp"] > split1)
    & (df["timestamp"] <= split2)
]

features = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

X_train = train[features].to_numpy()
y_train = train["target_recall"].to_numpy()
X_val = val[features].to_numpy()
y_val = val["target_recall"].to_numpy()

models = {
    "Mean baseline": None,
    "Linear Regression": LinearRegression(),
    "RF-1": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ),
}

print("Validation comparison")
print("---------------------")

for name, model in models.items():
    if model is None:
        predictions = np.full(len(y_val), y_train.mean())
    else:
        model.fit(X_train, y_train)
        predictions = np.clip(model.predict(X_val), 0, 1)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    print(f"{name:<20} MAE={mae:.6f} RMSE={rmse:.6f}")
