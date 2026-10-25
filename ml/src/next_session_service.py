"""Load the validation-selected next-session model and predict a candidate gap."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import torch

from next_session_model import (
    FEATURES, MODEL_DIR, NextSessionGRU, event_features, sequence_with_padding,
)


@lru_cache(maxsize=1)
def _load_artifact():
    artifact = joblib.load(MODEL_DIR / "next_session_champion.joblib")
    metadata_path = MODEL_DIR.parent / "reports" / "next_session_evaluation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if artifact.get("kind") == "gru":
        checkpoint = torch.load(MODEL_DIR / artifact["model_path"], map_location="cpu", weights_only=True)
        model = NextSessionGRU(
            input_size=checkpoint["input_size"], hidden_size=checkpoint["hidden_size"]
        ).build()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        artifact["_model"] = model
        artifact["_scaler_mean"] = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        artifact["_scaler_scale"] = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    return artifact, metadata


def predict_next_session(events: list[dict], candidate_gap_days: float) -> dict:
    artifact, metadata = _load_artifact()
    rows = event_features(events, candidate_gap_days)
    kind = artifact["kind"]

    if kind == "gru":
        sequence = sequence_with_padding(rows)
        scaled = (sequence - artifact["_scaler_mean"]) / artifact["_scaler_scale"]
        tensor = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            score = float(artifact["_model"](tensor).item())
    elif kind == "random_forest":
        score = float(artifact["model"].predict(rows[-1:].astype(float))[0])
    elif kind == "linear_regression":
        score = float(artifact["model"].predict(rows[-1:].astype(float))[0])
    elif kind == "hlr":
        beta = np.asarray(artifact["parameters"], dtype=float)
        current = rows[-1]
        log_half_life = beta[0] + beta[1] * current[0] + beta[2] * current[1]
        half_life = np.exp(np.clip(log_half_life, -12, 12))
        score = float(np.exp(-np.log(2.0) * candidate_gap_days / half_life))
    elif kind == "mean":
        score = float(artifact["value"])
    else:
        raise RuntimeError(f"Unsupported trained next-session model kind: {kind}")

    return {
        "predicted_next_session_recall": float(np.clip(score, 0.0, 1.0)),
        "candidate_gap_days": float(candidate_gap_days),
        "model": kind,
        "target_type": "next-session correct/seen fraction",
        "calibrated": False,
        "experimental": True,
        "evaluation_protocols": list(metadata.get("evaluation_protocols", {}).keys()),
    }
