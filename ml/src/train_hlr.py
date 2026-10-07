import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "log_history_seen",
    "historical_accuracy",
]


def predict_hlr(params, X, t):
    """Predict recall probability using Half-Life Regression."""

    log_H = np.clip(X @ params, -20, 20)
    H = np.exp(log_H)

    exponent = np.clip(-t / H, -50, 0)

    predictions = 2 ** exponent

    return np.clip(predictions, 0.0, 1.0)


def hlr_loss(params, X, t, y):
    """Mean squared error used to train HLR."""

    predictions = predict_hlr(params, X, t)

    return np.mean((predictions - y) ** 2)


def main():

    # Dataset path
    dataset_path = Path(
        "../data/raw/settles.acl16.learning_traces.13m.csv.gz"
    )

    # Load a development sample
    df = pd.read_csv(
        dataset_path,
        compression="gzip",
        nrows=50000
    )

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    # Sort learning history
    df = df.sort_values(
        ["user_id", "lexeme_id", "timestamp"]
    ).reset_index(drop=True)

    # Feature engineering
    df["delta_days"] = df["delta"] / (60 * 60 * 24)
    df["log_history_seen"] = np.log1p(df["history_seen"])

    df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"]
    )
    df["attempt_count"] = df["history_seen"]
    df["log_time_gap"] = np.log1p(df["delta_days"])
    # Remove invalid rows
    df = df[
        (df["delta"] > 0) &
        (df["history_seen"] > 0) &
        (df["session_seen"] > 0) &
        (df["p_recall"] >= 0) &
        (df["p_recall"] <= 1)
    ].copy()

    # Sort chronologically for temporal split
    df = df.sort_values("timestamp")

    # 70 / 15 / 15 chronological split
    train_end = df["timestamp"].quantile(0.70)
    val_end = df["timestamp"].quantile(0.85)

    train_df = df[
        df["timestamp"] <= train_end
    ]

    val_df = df[
        (df["timestamp"] > train_end) &
        (df["timestamp"] <= val_end)
    ]

    test_df = df[
        df["timestamp"] > val_end
    ]

    # Training data
    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df["p_recall"].to_numpy(dtype=float)
    t_train = train_df["delta_days"].to_numpy(dtype=float)

    # Train HLR
    initial_params = np.zeros(len(FEATURES))

    result = minimize(
        hlr_loss,
        initial_params,
        args=(X_train, t_train, y_train),
        method="L-BFGS-B"
    )

    print("Training successful:", result.success)
    print("Training loss:", result.fun)

    # Test
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df["p_recall"].to_numpy(dtype=float)
    t_test = test_df["delta_days"].to_numpy(dtype=float)

    predictions = predict_hlr(
        result.x,
        X_test,
        t_test
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

    print("HLR MAE:", mae)
    print("HLR RMSE:", rmse)

    # Save model
    model_path = Path(
        "../models/hlr_parameters.json"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {
        "features": FEATURES,
        "parameters": result.x.tolist(),
        "target_retention": 0.80,
        "mae": float(mae),
        "rmse": float(rmse)
    }

    with open(model_path, "w") as f:
        json.dump(
            model_data,
            f,
            indent=4
        )

    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()