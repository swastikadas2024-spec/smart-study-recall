"""Feature contract and inference helpers for next-session recall prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SEQUENCE_LENGTH = 10
MAX_CANDIDATE_GAP_DAYS = 30.0
TARGET_RECALL = 0.80
FEATURES = [
    "log_updated_seen",
    "updated_accuracy",
    "log_session_seen",
    "session_accuracy",
    "log_candidate_gap_days",
]
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def event_features(
    events: list[dict[str, Any]], candidate_gap_days: float
) -> np.ndarray:
    """Build ordered event rows; the last row uses the proposed gap.

    Each event has questions_seen, questions_correct and (for prior events)
    gap_days to the following observed event. The final event's saved gap is
    intentionally ignored and replaced by the candidate interval.
    """
    if not events:
        raise ValueError("At least one study event is required.")
    if not np.isfinite(candidate_gap_days) or not 0 < candidate_gap_days <= MAX_CANDIDATE_GAP_DAYS:
        raise ValueError(
            f"candidate_gap_days must be greater than 0 and at most {MAX_CANDIDATE_GAP_DAYS:g}."
        )

    cumulative_seen = 0
    cumulative_correct = 0
    rows: list[list[float]] = []
    for index, event in enumerate(events):
        seen = event.get("questions_seen")
        correct = event.get("questions_correct")
        if isinstance(seen, bool) or not isinstance(seen, int) or seen < 1:
            raise ValueError("Each event questions_seen must be a positive integer.")
        if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= seen:
            raise ValueError("Each event questions_correct must be between 0 and questions_seen.")
        gap = candidate_gap_days if index == len(events) - 1 else event.get("gap_days")
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not np.isfinite(gap) or gap <= 0:
            raise ValueError("Prior events must include a positive gap_days to the next event.")

        cumulative_seen += seen
        cumulative_correct += correct
        rows.append([
            float(np.log1p(cumulative_seen)),
            float(cumulative_correct / cumulative_seen),
            float(np.log1p(seen)),
            float(correct / seen),
            float(np.log1p(float(gap))),
        ])

    return np.asarray(rows[-SEQUENCE_LENGTH:], dtype=np.float32)


def sequence_with_padding(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2 or rows.shape[1] != len(FEATURES):
        raise ValueError("Feature rows do not match the next-session model contract.")
    if len(rows) >= SEQUENCE_LENGTH:
        return rows[-SEQUENCE_LENGTH:]
    padding = np.repeat(rows[:1], SEQUENCE_LENGTH - len(rows), axis=0)
    return np.concatenate([padding, rows], axis=0)


@dataclass
class NextSessionGRU:
    """Lazy PyTorch architecture definition used by training and serving."""
    input_size: int = len(FEATURES)
    hidden_size: int = 64

    def build(self):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
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
                outputs, _ = self.gru(x)
                return self.head(outputs[:, -1, :]).squeeze(-1)

        return Network(self.input_size, self.hidden_size)
