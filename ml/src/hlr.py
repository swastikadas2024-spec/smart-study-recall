import numpy as np


FEATURES = [
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]


def predict_hlr(params, X, t):
    """
    Predict recall probability using Half-Life Regression.

    H = exp(X @ params)
    P(recall) = 2^(-t / H)
    """

    log_H = np.clip(X @ params, -20, 20)
    H = np.exp(log_H)

    exponent = np.clip(-t / H, -50, 0)

    predictions = 2 ** exponent

    return np.clip(predictions, 0.0, 1.0)


def calculate_half_life(params, X):
    """
    Calculate predicted memory half-life in days.
    """

    log_H = np.clip(X @ params, -20, 20)

    return np.exp(log_H)