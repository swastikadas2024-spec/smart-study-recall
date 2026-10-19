import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

from hlr import predict_hlr


FEATURES = [
    "log_history_seen",
    "historical_accuracy",
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

    return data


def get_hlr_predictions(model_params, df):
    data = prepare_features(df)

    X = data[FEATURES].to_numpy(dtype=float)

    t = data["delta_days"].to_numpy(dtype=float)

    predictions = predict_hlr(
        model_params,
        X,
        t
    )

    return predictions


def train_calibrator(
    model_params,
    validation_df
):
    predictions = get_hlr_predictions(
        model_params,
        validation_df
    )

    actual = validation_df[
        "p_recall"
    ].to_numpy(dtype=float)

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    calibrator.fit(
        predictions,
        actual
    )

    return calibrator


if __name__ == "__main__":
    print("Recall calibration module ready.")