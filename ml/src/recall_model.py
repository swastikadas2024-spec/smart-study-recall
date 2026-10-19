import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "log_delta_days",
]


def prepare_features(df):
    """
    Create features for recall prediction.
    """

    data = df.copy()

    # Avoid log(0)
    data["log_history_seen"] = np.log1p(
        data["history_seen"]
    )

    # Historical performance before the current recall
    data["historical_accuracy"] = (
        data["history_correct"]
        / data["history_seen"]
    )

    # Time since previous interaction
    data["log_delta_days"] = np.log1p(
        data["delta_days"]
    )

    return data


def train_recall_model(train_df):
    """
    Train a logistic regression recall predictor.
    """

    data = prepare_features(train_df)

    X = data[FEATURES].to_numpy(dtype=float)

    # Convert recall probability into binary recall outcome
    y = (data["p_recall"] >= 0.5).astype(int)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    return model


def evaluate_model(model, test_df):
    """
    Evaluate recall prediction performance.
    """

    data = prepare_features(test_df)

    X = data[FEATURES].to_numpy(dtype=float)

    y_actual = data["p_recall"].to_numpy(dtype=float)

    predictions = model.predict_proba(X)[:, 1]

    mae = mean_absolute_error(
        y_actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            predictions
        )
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def save_model(model, metrics):
    """
    Save model parameters and evaluation metrics.
    """

    output_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "recall_model.json"
    )

    model_data = {
        "features": FEATURES,
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }

    with open(output_path, "w") as f:
        json.dump(
            model_data,
            f,
            indent=4
        )

    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    print("Recall model module ready.")