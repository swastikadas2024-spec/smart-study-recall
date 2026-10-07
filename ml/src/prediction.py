import json
from pathlib import Path

import numpy as np

from hlr import predict_hlr, calculate_half_life
from revision_policy import get_revision_recommendation


TARGET_RETENTION = 0.80

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "hlr_parameters.json"
)


def load_model():
    with open(MODEL_PATH, "r") as f:
        model = json.load(f)

    params = np.array(model["parameters"], dtype=float)

    return params


def calculate_forgetting_risk(retention_probability):
    return 1.0 - retention_probability


def calculate_revision_days(half_life_days):
    revision_days = (
        -half_life_days
        * np.log(TARGET_RETENTION)
        / np.log(2)
    )

    return revision_days


def get_memory_prediction(features, elapsed_days):
    params = load_model()

    history_seen, history_correct = features

    log_history_seen = np.log1p(history_seen)

    historical_accuracy = (
        history_correct / history_seen
        if history_seen > 0
        else 0.0
    )

    X = np.asarray(
        [log_history_seen, historical_accuracy],
        dtype=float
    ).reshape(1, -1)

    t = np.asarray(
        [elapsed_days],
        dtype=float
    )

    retention = float(
        predict_hlr(params, X, t)[0]
    )

    half_life = float(
        calculate_half_life(params, X)[0]
    )

    forgetting_risk = calculate_forgetting_risk(
        retention
    )

    revision_days = calculate_revision_days(
        half_life
    )

    recommendation = get_revision_recommendation(
        retention
    )

    return {
        "retention_probability": retention,
        "forgetting_risk": forgetting_risk,
        "half_life_days": half_life,
        "recommended_revision_days": revision_days,
        "revision_priority": recommendation["priority"],
        "revision_message": recommendation["message"],
    }