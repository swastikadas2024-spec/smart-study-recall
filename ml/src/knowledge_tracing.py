"""Causal sequence models for next-interaction knowledge tracing.

The target in this module is an observed interaction/session correctness
fraction. It is not a delayed-memory or forgetting-probability target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEQUENCE_LENGTH = 30
NUMERIC_FEATURES = (
    "log_history_seen",
    "history_accuracy",
    "log_gap_days",
    "previous_session_accuracy",
)


def build_sequences(
    frame,
    item_to_index: dict[str, int],
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create causal windows for predicting each event from prior history.

    Each target's token contains its known item ID and pre-event history fields.
    It never contains that target's session correctness. Previous answers enter
    subsequent tokens through ``previous_session_accuracy``.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    required = {
        "user_id", "lexeme_id", "timestamp", "history_seen", "history_correct",
        "session_seen", "session_correct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing interaction fields: {', '.join(sorted(missing))}")

    ordered = frame.sort_values(
        ["user_id", "timestamp", "lexeme_id"], kind="mergesort"
    ).reset_index(drop=True)
    users = ordered["user_id"].astype(str).to_numpy()
    item_ids = ordered["lexeme_id"].astype(str).map(item_to_index).fillna(1).to_numpy(dtype=np.int64)
    timestamps = ordered["timestamp"].to_numpy(dtype=np.float64)
    hist_seen = ordered["history_seen"].to_numpy(dtype=np.float32)
    hist_correct = ordered["history_correct"].to_numpy(dtype=np.float32)
    session_seen = ordered["session_seen"].to_numpy(dtype=np.float32)
    session_correct = ordered["session_correct"].to_numpy(dtype=np.float32)
    n = len(ordered)
    targets = session_correct / np.maximum(session_seen, 1.0)
    features = np.zeros((n, len(NUMERIC_FEATURES)), dtype=np.float32)
    previous_accuracy = np.zeros(n, dtype=np.float32)
    gaps = np.zeros(n, dtype=np.float64)

    # A timestamp can contain several learner-item records from one study
    # session. Treat the whole timestamp as one information boundary: the
    # target token may not see another target's answer from that same session.
    group_starts = np.flatnonzero(np.r_[True, (users[1:] != users[:-1]) | (timestamps[1:] != timestamps[:-1])])
    group_ends = np.r_[group_starts[1:], n]
    windows = np.zeros((n, sequence_length, 1 + len(NUMERIC_FEATURES)), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    window_indices: list[list[int] | None] = [None] * n
    prior_events: list[int] = []
    previous_group_accuracy = 0.0
    previous_group_timestamp = None
    previous_user = None
    for group_start, group_end in zip(group_starts, group_ends):
        user = users[group_start]
        if user != previous_user:
            prior_events = []
            previous_group_accuracy = 0.0
            previous_group_timestamp = None
            previous_user = user
        gap_days = (
            max(0.0, (timestamps[group_start] - previous_group_timestamp) / 86400.0)
            if previous_group_timestamp is not None else 0.0
        )
        group_accuracy = float(
            session_correct[group_start:group_end].sum()
            / max(session_seen[group_start:group_end].sum(), 1.0)
        )
        for index in range(group_start, group_end):
            gaps[index] = gap_days
            previous_accuracy[index] = previous_group_accuracy
            history = prior_events[-(sequence_length - 1):] if sequence_length > 1 else []
            indexes = history + [index]
            lengths[index] = len(indexes)
            window_indices[index] = indexes
            windows[index, :len(indexes), 0] = item_ids[indexes]
        prior_events.extend(range(group_start, group_end))
        if len(prior_events) > sequence_length - 1:
            del prior_events[:len(prior_events) - (sequence_length - 1)]
        previous_group_accuracy = group_accuracy
        previous_group_timestamp = timestamps[group_start]

    features[:, 0] = np.log1p(np.maximum(hist_seen, 0))
    features[:, 1] = np.divide(
        hist_correct,
        hist_seen,
        out=np.zeros_like(hist_correct),
        where=hist_seen > 0,
    )
    features[:, 2] = np.log1p(gaps)
    features[:, 3] = previous_accuracy

    for index, indexes in enumerate(window_indices):
        assert indexes is not None
        length = len(indexes)
        windows[index, :length, 1:] = features[indexes]

    return (
        windows,
        lengths.astype(np.int64),
        targets.astype(np.float32),
        session_seen.astype(np.float32),
        ordered["user_id"].to_numpy(),
    )


@dataclass
class SequenceKnowledgeTracer:
    """Factory for a tabular MLP, LSTM, or Transformer predictor."""

    item_count: int
    architecture: str = "lstm"
    embedding_size: int = 32
    hidden_size: int = 64
    dropout: float = 0.2
    sequence_length: int = SEQUENCE_LENGTH

    def build(self):
        import torch
        from torch import nn

        if self.architecture not in {"mlp", "lstm", "transformer"}:
            raise ValueError("architecture must be 'mlp', 'lstm', or 'transformer'")

        class Network(nn.Module):
            def __init__(inner_self):
                super().__init__()
                inner_self.item_embedding = nn.Embedding(self.item_count, self.embedding_size, padding_idx=0)
                inner_self.input_projection = nn.Sequential(
                    nn.Linear(self.embedding_size + len(NUMERIC_FEATURES), self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.GELU(),
                )
                if self.architecture == "mlp":
                    # Order-agnostic history summary is a feedforward baseline;
                    # the current event representation is kept separately.
                    inner_self.head = nn.Sequential(
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, 32),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(32, 1),
                    )
                elif self.architecture == "lstm":
                    inner_self.encoder = nn.LSTM(
                        self.hidden_size,
                        self.hidden_size,
                        num_layers=2,
                        batch_first=True,
                        dropout=self.dropout,
                    )
                else:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=4,
                        dim_feedforward=self.hidden_size * 2,
                        dropout=self.dropout,
                        batch_first=True,
                        norm_first=True,
                    )
                    inner_self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                    inner_self.position_embedding = nn.Embedding(self.sequence_length, self.hidden_size)
                inner_self.dropout = nn.Dropout(self.dropout)
                if self.architecture != "mlp":
                    inner_self.head = nn.Sequential(
                        nn.Linear(self.hidden_size, 32),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(32, 1),
                    )

            def forward(inner_self, x, lengths):
                item = x[:, :, 0].long()
                numeric = x[:, :, 1:]
                encoded = inner_self.input_projection(
                    torch.cat([inner_self.item_embedding(item), numeric], dim=-1)
                )
                positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                padding = positions >= lengths.unsqueeze(1)
                if self.architecture == "mlp":
                    valid = (~padding).unsqueeze(-1).to(encoded.dtype)
                    mean_context = (encoded * valid).sum(dim=1) / lengths.unsqueeze(1).clamp_min(1)
                    current = encoded[torch.arange(len(lengths), device=x.device), lengths - 1]
                    summary = torch.cat((current, mean_context), dim=-1)
                    return torch.sigmoid(inner_self.head(inner_self.dropout(summary)).squeeze(-1))
                if self.architecture == "transformer":
                    encoded = encoded + inner_self.position_embedding(positions)
                    causal = torch.triu(
                        torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    encoded = inner_self.encoder(
                        encoded,
                        mask=causal,
                        src_key_padding_mask=padding,
                    )
                else:
                    packed = nn.utils.rnn.pack_padded_sequence(
                        encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
                    )
                    packed_output, _ = inner_self.encoder(packed)
                    encoded, _ = nn.utils.rnn.pad_packed_sequence(
                        packed_output, batch_first=True, total_length=x.shape[1]
                    )
                last = encoded[torch.arange(len(lengths), device=x.device), lengths - 1]
                return torch.sigmoid(inner_self.head(inner_self.dropout(last)).squeeze(-1))

        return Network()
