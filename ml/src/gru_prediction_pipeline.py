
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "recall_gru_v3.pt"
)

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]


class RecallGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


# Load the checkpoint once.
_checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

_model = RecallGRU(
    input_size=_checkpoint["input_size"],
    hidden_size=_checkpoint["hidden_size"],
)

_model.load_state_dict(_checkpoint["state_dict"])
_model.eval()

_SEQUENCE_LENGTH = _checkpoint["sequence_length"]
_SCALER_MEAN = np.asarray(
    _checkpoint["scaler_mean"], dtype=np.float32
)
_SCALER_SCALE = np.asarray(
    _checkpoint["scaler_scale"], dtype=np.float32
)


def predict_gru_session_recall(
    history_seen: int,
    history_correct: int,
    elapsed_days: float,
    recent_history: list | None = None,
) -> dict:
    """
    Predict session-level recall using GRU v3.

    recent_history contains up to 9 earlier feature rows,
    ordered from oldest to newest. Each row must contain:
    history_seen, history_correct, elapsed_days.

    If omitted, the current feature row is repeated to fill
    the sequence. This fallback is convenient for API testing,
    but does not represent a genuine observed event history.
    """

    if isinstance(history_seen, bool) or not isinstance(
        history_seen, int
    ):
        raise ValueError("history_seen must be an integer.")

    if isinstance(history_correct, bool) or not isinstance(
        history_correct, int
    ):
        raise ValueError("history_correct must be an integer.")

    if history_seen < 1:
        raise ValueError("history_seen must be at least 1.")

    if not 0 <= history_correct <= history_seen:
        raise ValueError(
            "history_correct must be between 0 and history_seen."
        )

    if not np.isfinite(elapsed_days) or elapsed_days <= 0:
        raise ValueError("elapsed_days must be greater than 0.")

    def make_features(seen, correct, elapsed):
        if isinstance(seen, bool) or not isinstance(seen, int):
            raise ValueError("Each history_seen must be an integer.")
        if isinstance(correct, bool) or not isinstance(correct, int):
            raise ValueError("Each history_correct must be an integer.")
        if seen < 1 or not 0 <= correct <= seen:
            raise ValueError("Invalid historical counts.")
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Each elapsed_days must be greater than 0.")

        return [
            float(np.log1p(seen)),
            float(correct / seen),
            float(elapsed),
            float(np.log1p(elapsed)),
        ]

    current_row = make_features(
        history_seen, history_correct, elapsed_days
    )

    rows = []

    if recent_history is not None:
        if not isinstance(recent_history, list):
            raise ValueError("recent_history must be a list.")

        if len(recent_history) > _SEQUENCE_LENGTH - 1:
            raise ValueError(
                "Provide at most 9 earlier history rows."
            )

        for item in recent_history:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each recent_history item must be an object."
                )

            try:
                row = make_features(
                    item["history_seen"],
                    item["history_correct"],
                    item["elapsed_days"],
                )
            except KeyError as exc:
                raise ValueError(
                    "Each history row requires history_seen, "
                    "history_correct, and elapsed_days."
                ) from exc

            rows.append(row)

    # The model expects exactly 10 rows. When fewer are available,
    # repeat the oldest available row, matching the training script's
    # left-padding strategy.
    rows.append(current_row)

    if len(rows) < _SEQUENCE_LENGTH:
        padding = [rows[0]] * (_SEQUENCE_LENGTH - len(rows))
        rows = padding + rows

    sequence = np.asarray(rows, dtype=np.float32)

    scaled = (sequence - _SCALER_MEAN) / _SCALER_SCALE
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction = float(_model(tensor).item())

    return {
        "predicted_session_recall_rate": float(
            np.clip(prediction, 0.0, 1.0)
        ),
        "historical_accuracy": history_correct / history_seen,
        "elapsed_days": float(elapsed_days),
        "model": "GRU-v3",
        "target_type": "session-level recall rate",
        "calibrated": False,
        "sequence_length": _SEQUENCE_LENGTH,
        "used_observed_history_rows": (
    len(recent_history) if recent_history is not None else 0
),
"history_fallback_used": (
    recent_history is None or len(recent_history) == 0
),
    }
