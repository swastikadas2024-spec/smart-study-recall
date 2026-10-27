"""Inference for the experimental next-interaction knowledge tracer."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from knowledge_tracing import NUMERIC_FEATURES, SequenceKnowledgeTracer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=1)
def _load_model(architecture: str):
    if architecture not in {"mlp", "lstm", "transformer"}:
        raise ValueError("architecture must be 'mlp', 'lstm', or 'transformer'")
    path = MODEL_DIR / f"knowledge_tracer_{architecture}_forward_time.pt"
    if not path.exists():
        raise FileNotFoundError("Train ml/src/train_knowledge_tracer.py before requesting KT predictions.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SequenceKnowledgeTracer(
        item_count=int(checkpoint["item_count"]),
        architecture=checkpoint["architecture"],
        embedding_size=int(checkpoint["embedding_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        dropout=float(checkpoint["dropout"]),
        sequence_length=int(checkpoint["sequence_length"]),
    ).build()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, checkpoint


def predict_next_interaction(
    interactions: list[dict],
    target_item_id: str,
    target_history_seen: int,
    target_history_correct: int,
    elapsed_days: float,
    architecture: str = "lstm",
) -> dict:
    model, checkpoint = _load_model(architecture)
    sequence_length = int(checkpoint["sequence_length"])
    vocab = checkpoint["item_to_index"]
    history = interactions[-max(0, sequence_length - 1):]
    tokens: list[list[float]] = []
    previous_group_accuracy = 0.0
    previous_timestamp = None
    group_gap_days = 0.0
    previous_group_rows: list[dict] = []

    for index, row in enumerate(interactions):
        timestamp = float(row["timestamp"])
        if previous_timestamp is None or timestamp != previous_timestamp:
            group_gap_days = (
                max(0.0, (timestamp - previous_timestamp) / 86400.0)
                if previous_timestamp is not None else 0.0
            )
            if previous_group_rows:
                seen = sum(int(item["session_seen"]) for item in previous_group_rows)
                correct = sum(int(item["session_correct"]) for item in previous_group_rows)
                previous_group_accuracy = correct / max(seen, 1)
            previous_group_rows = []
            previous_timestamp = timestamp
        previous_group_rows.append(row)
        if index < len(interactions) - len(history):
            continue
        seen = int(row["history_seen"])
        correct = int(row["history_correct"])
        tokens.append([
            float(vocab.get(str(row["item_id"]), 1)),
            float(np.log1p(seen)),
            correct / seen if seen else 0.0,
            float(np.log1p(group_gap_days)),
            previous_group_accuracy,
        ])

    target_seen = int(target_history_seen)
    target_correct = int(target_history_correct)
    target_previous_accuracy = previous_group_accuracy
    tokens.append([
        float(vocab.get(str(target_item_id), 1)),
        float(np.log1p(target_seen)),
        target_correct / target_seen if target_seen else 0.0,
        float(np.log1p(elapsed_days)),
        target_previous_accuracy,
    ])
    tokens = tokens[-sequence_length:]
    sequence = np.zeros((1, sequence_length, len(NUMERIC_FEATURES) + 1), dtype=np.float32)
    sequence[0, :len(tokens)] = np.asarray(tokens, dtype=np.float32)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)

    active = sequence[0, :, 0] != 0
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    sequence[0, active, 1:] = (sequence[0, active, 1:] - mean) / scale
    with torch.no_grad():
        prediction = float(model(torch.from_numpy(sequence), lengths).item())

    known = str(target_item_id) in vocab
    return {
        "predicted_next_interaction_correct_fraction": float(np.clip(prediction, 0.0, 1.0)),
        "target_type": "next interaction session correct/seen fraction",
        "model": checkpoint["architecture"],
        "model_version": checkpoint["artifact_sha256"],
        "history_interactions_used": len(history),
        "cold_start": len(history) == 0,
        "trained_item_vocabulary_match": known,
        "experimental": True,
        "calibrated": False,
        "limitations": [
            "This is not a delayed-retention probability or a validated forgetting date.",
            "The model was trained on Duolingo lexeme interactions; course-topic transfer is unvalidated.",
            *([] if known else ["The target item was unseen during training and uses the OOV embedding."]),
        ],
    }
