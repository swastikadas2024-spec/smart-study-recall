import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "log_delta_days",
]


def prepare_features(df):
    data = df.copy()

    data["log_history_seen"] = np.log1p(
        data["history_seen"]
    )

    data["historical_accuracy"] = (
        data["history_correct"]
        / data["history_seen"]
    )

    data["log_delta_days"] = np.log1p(
        data["delta_days"]
    )

    return data


def train_model(train_df):
    data = prepare_features(train_df)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["p_recall"].to_numpy(dtype=float)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def evaluate_model(model, test_df):
    data = prepare_features(test_df)

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["p_recall"].to_numpy(dtype=float)

    predictions = np.clip(
        model.predict(X),
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

    return mae, rmse


def save_model(model, mae, rmse):
    output_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "recall_regression.json"
    )

    model_data = {
        "features": FEATURES,
        "mae": float(mae),
        "rmse": float(rmse),
        "model_type": "RandomForestRegressor",
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_leaf": 20
    }

    with open(output_path, "w") as f:
        json.dump(
            model_data,
            f,
            indent=4
        )

    print(f"Model metadata saved to: {output_path}")


if __name__ == "__main__":
    print("Recall regression model module ready.")