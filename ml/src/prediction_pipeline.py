
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "recall_rf1.joblib"
)

# Load the fitted model once when this module is imported.
_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_FEATURES = _bundle["features"]


def predict_session_recall(
    history_seen: int,
    history_correct: int,
    elapsed_days: float,
) -> dict:
    """
    Estimate session-level recall rate from historical information.

    Important: this model predicts a session-level recall rate,
    not a validated individual future-recall probability.
    """

    if history_seen < 1:
        raise ValueError("history_seen must be at least 1.")

    if not 0 <= history_correct <= history_seen:
        raise ValueError(
            "history_correct must be between 0 and history_seen."
        )

    if not np.isfinite(elapsed_days) or elapsed_days <= 0:
        raise ValueError("elapsed_days must be greater than 0.")

    historical_accuracy = history_correct / history_seen

    features = {
        "log_history_seen": np.log1p(history_seen),
        "historical_accuracy": historical_accuracy,
        "delta_days": elapsed_days,
        "log_delta_days": np.log1p(elapsed_days),
    }


    X = np.array(
        [[features[name] for name in _FEATURES]],
        dtype=float,
    )


    prediction = float(
        np.clip(_model.predict(X)[0], 0.0, 1.0)
    )

    return {
        "predicted_session_recall_rate": prediction,
        "historical_accuracy": historical_accuracy,
        "elapsed_days": float(elapsed_days),
        "model": "RF-1",
        "target_type": "session-level recall rate",
        "calibrated": False,
    }
