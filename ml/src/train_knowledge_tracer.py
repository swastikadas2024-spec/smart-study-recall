"""Train/evaluate an MLP, causal LSTM, or Transformer knowledge tracer.

Example:
  python ml/src/train_knowledge_tracer.py --architecture lstm --max-epochs 8
  python ml/src/train_knowledge_tracer.py --architecture mlp --optimizer rmsprop

The target is an interaction's observed session-correct fraction, not delayed
recall. The script uses sampled Duolingo events and writes separate artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_tracing import (  # noqa: E402
    NUMERIC_FEATURES,
    SEQUENCE_LENGTH,
    SequenceKnowledgeTracer,
    build_sequences,
)

DATA_PATH = ROOT / "ml/data/raw/settles.acl16.learning_traces.13m.csv.gz"
MODEL_DIR = ROOT / "ml/models"
REPORT_DIR = ROOT / "ml/reports"
USER_HASH_KEY = "smartstudyrecall"
RAW_COLUMNS = [
    "user_id", "lexeme_id", "timestamp", "history_seen", "history_correct",
    "session_seen", "session_correct",
]


def load_sample(sample_modulus: int, chunk_size: int = 250_000) -> tuple[pd.DataFrame, int]:
    if sample_modulus < 1:
        raise ValueError("sample_modulus must be at least 1")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    retained = []
    rows_read = 0
    for chunk in pd.read_csv(DATA_PATH, compression="gzip", usecols=RAW_COLUMNS, chunksize=chunk_size):
        rows_read += len(chunk)
        chunk["user_id"] = chunk["user_id"].astype(str)
        hashes = pd.util.hash_pandas_object(
            chunk["user_id"], index=False, hash_key=USER_HASH_KEY
        ).to_numpy(dtype=np.uint64)
        chunk = chunk[hashes % sample_modulus == 0].copy()
        if chunk.empty:
            continue
        for name in RAW_COLUMNS[2:]:
            chunk[name] = pd.to_numeric(chunk[name], errors="coerce")
        chunk = chunk.dropna(subset=RAW_COLUMNS[2:] + ["lexeme_id"])
        chunk = chunk[
            (chunk.timestamp >= 0)
            & (chunk.history_seen >= 0)
            & (chunk.history_correct >= 0)
            & (chunk.history_correct <= chunk.history_seen)
            & (chunk.session_seen > 0)
            & (chunk.session_correct >= 0)
            & (chunk.session_correct <= chunk.session_seen)
        ]
        if not chunk.empty:
            retained.append(chunk)
    if not retained:
        raise ValueError("No valid rows retained from the dataset sample.")
    frame = pd.concat(retained, ignore_index=True)
    frame["lexeme_id"] = frame["lexeme_id"].astype(str)
    # Remove only duplicate observations for the same learner-item at one
    # timestamp. Different items at the same timestamp are kept as a session
    # group; build_sequences prevents a target from seeing its peers' labels.
    ties = frame.duplicated(["user_id", "lexeme_id", "timestamp"], keep=False)
    frame = frame.loc[~ties].sort_values(
        ["user_id", "timestamp", "lexeme_id"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError("All sampled rows were removed as invalid or ambiguous ties.")
    return frame, rows_read


def make_splits(frame: pd.DataFrame, protocol: str, seed: int):
    if protocol == "forward_time":
        timestamps = frame.timestamp.to_numpy()
        order = np.argsort(timestamps, kind="stable")
        n = len(order)
        # Snap cutoffs to a timestamp boundary so simultaneous records cannot
        # straddle train/validation/test or leak their item IDs into train vocab.
        train_start_validation = timestamps[order[int(n * 0.70)]]
        validation_start_test = timestamps[order[int(n * 0.85)]]
        return (
            np.flatnonzero(timestamps < train_start_validation),
            np.flatnonzero(
                (timestamps >= train_start_validation)
                & (timestamps < validation_start_test)
            ),
            np.flatnonzero(timestamps >= validation_start_test),
        )
    users = frame.user_id.astype(str).unique().to_numpy(dtype=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_test = max(1, int(len(users) * 0.15))
    n_val = max(1, int(len(users) * 0.15))
    test_users = set(users[:n_test])
    val_users = set(users[n_test:n_test + n_val])
    user_ids = frame.user_id.astype(str).to_numpy()
    return (
        np.flatnonzero(~np.isin(user_ids, list(test_users | val_users))),
        np.flatnonzero(np.isin(user_ids, list(val_users))),
        np.flatnonzero(np.isin(user_ids, list(test_users))),
    )


def weighted_metrics(actual, predicted, seen, correct) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, 1 - 1e-6)
    weights = np.asarray(seen, dtype=float)
    denominator = max(float(weights.sum()), 1.0)
    mae = float(np.sum(weights * np.abs(actual - predicted)) / denominator)
    mse = float(np.sum(weights * (actual - predicted) ** 2) / denominator)
    log_loss = float(-np.sum(
        np.asarray(correct) * np.log(predicted)
        + (weights - np.asarray(correct)) * np.log1p(-predicted)
    ) / denominator)
    edges = np.linspace(0, 1, 11)
    bin_ids = np.minimum(np.digitize(predicted, edges[1:-1]), 9)
    ece = 0.0
    for index in range(10):
        mask = bin_ids == index
        if mask.any():
            bin_weight = weights[mask].sum()
            ece += float(bin_weight / denominator * abs(
                np.average(predicted[mask], weights=weights[mask])
                - np.average(actual[mask], weights=weights[mask])
            ))
    labels = np.r_[np.ones(len(actual)), np.zeros(len(actual))]
    scores = np.r_[predicted, predicted]
    auc_weights = np.r_[correct, weights - np.asarray(correct)]
    active = auc_weights > 0
    auc = None
    if len(np.unique(labels[active])) == 2:
        auc = float(roc_auc_score(labels[active], scores[active], sample_weight=auc_weights[active]))
    return {
        "weighted_mae": mae,
        "weighted_rmse": float(np.sqrt(mse)),
        "weighted_binomial_log_loss": log_loss,
        "weighted_brier_score": mse,
        "weighted_ece_10_bins": ece,
        "weighted_roc_auc": auc,
    }


def _feature_matrix(windows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    rows = np.arange(len(windows))
    return windows[rows, lengths - 1, 1:].astype(np.float32)


def train_deep(
    architecture: str,
    train_x: np.ndarray,
    train_lengths: np.ndarray,
    train_y: np.ndarray,
    train_seen: np.ndarray,
    val_x: np.ndarray,
    val_lengths: np.ndarray,
    val_y: np.ndarray,
    val_seen: np.ndarray,
    seed: int,
    max_epochs: int,
    batch_size: int,
    item_count: int,
    device,
    optimizer_name: str = "adamw",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observed_tokens = train_x[:, :, 0] != 0
    scaler = StandardScaler().fit(train_x[:, :, 1:][observed_tokens])

    def scale(values):
        result = values.copy()
        result[:, :, 1:] = scaler.transform(
            result[:, :, 1:].reshape(-1, len(NUMERIC_FEATURES))
        ).reshape(result[:, :, 1:].shape)
        result[:, :, 1:][result[:, :, 0] == 0] = 0.0
        return result.astype(np.float32)

    train_x = scale(train_x)
    val_x = scale(val_x)
    model = SequenceKnowledgeTracer(
        item_count, architecture=architecture, sequence_length=train_x.shape[1]
    ).build().to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_lengths),
            torch.from_numpy(train_y), torch.from_numpy(train_seen),
        ), batch_size=batch_size, shuffle=True,
    )
    vx = torch.from_numpy(val_x).to(device)
    vl = torch.from_numpy(val_lengths).to(device)
    vy = torch.from_numpy(val_y).to(device)
    vs = torch.from_numpy(val_seen).to(device)
    optimizer_types = {
        "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4),
        "rmsprop": lambda: torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4),
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
    }
    optimizer = optimizer_types[optimizer_name]()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    patience = 2
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for xb, lengths, yb, seen in loader:
            xb, lengths, yb, seen = xb.to(device), lengths.to(device), yb.to(device), seen.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, lengths).clamp(1e-6, 1 - 1e-6)
            loss = torch.nn.functional.binary_cross_entropy(pred, yb, reduction="none")
            loss = (loss * seen).sum() / seen.sum().clamp_min(1)
            batch_weight = float(seen.sum().item())
            train_loss_sum += float(loss.detach().item()) * batch_weight
            train_weight_sum += batch_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx, vl).clamp(1e-6, 1 - 1e-6)
            val_loss = float((torch.nn.functional.binary_cross_entropy(pred, vy, reduction="none") * vs).sum().div(vs.sum()).item())
        train_loss = train_loss_sum / max(train_weight_sum, 1.0)
        history.append({
            "epoch": epoch,
            "train_weighted_binomial_loss": train_loss,
            "validation_weighted_binomial_loss": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        print(
            f"epoch {epoch}/{max_epochs}: train weighted binomial loss={train_loss:.5f}; "
            f"validation weighted binomial loss={val_loss:.5f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_epoch, best_loss, history


def predict_deep(model, scaler, windows, lengths, device, batch_size=1024):
    values = windows.copy()
    flat = values[:, :, 1:].reshape(-1, len(NUMERIC_FEATURES))
    values[:, :, 1:] = scaler.transform(flat).reshape(values[:, :, 1:].shape)
    values[:, :, 1:][values[:, :, 0] == 0] = 0.0
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            xb = torch.from_numpy(values[start:start + batch_size]).to(device)
            lengths_batch = torch.from_numpy(lengths[start:start + batch_size]).to(device)
            predictions.extend(model(xb, lengths_batch).cpu().numpy())
    return np.asarray(predictions, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["mlp", "lstm", "transformer"], default="lstm")
    parser.add_argument("--optimizer", choices=["adamw", "adam", "rmsprop", "sgd"], default="adamw")
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--protocol", choices=["forward_time", "learner_held_out"], default="forward_time")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    frame, rows_read = load_sample(args.sample_modulus)
    print(
        f"Loaded {rows_read:,} source rows; retained {len(frame):,} examples "
        f"from {frame.user_id.nunique():,} learners in {time.time() - started:.1f}s.",
        flush=True,
    )
    split = make_splits(frame, args.protocol, args.seed)
    train_i, val_i, test_i = split
    split_boundaries = None
    if args.protocol == "forward_time":
        split_boundaries = {
            "validation_start_timestamp": float(frame.timestamp.iloc[val_i].min()),
            "test_start_timestamp": float(frame.timestamp.iloc[test_i].min()),
            "rule": "event-count quantile cutoffs snapped to complete timestamp groups",
        }
    item_to_index = {item: index + 2 for index, item in enumerate(sorted(frame.lexeme_id.iloc[train_i].astype(str).unique()))}
    windows, lengths, target, seen, users = build_sequences(
        frame, item_to_index, args.sequence_length
    )
    print(f"Built causal windows in {time.time() - started:.1f}s; fitting on {device}.", flush=True)
    train_x, val_x, test_x = windows[train_i], windows[val_i], windows[test_i]
    train_l, val_l, test_l = lengths[train_i], lengths[val_i], lengths[test_i]
    train_y, val_y, test_y = target[train_i], target[val_i], target[test_i]
    train_seen, val_seen, test_seen = seen[train_i], seen[val_i], seen[test_i]
    train_correct, val_correct, test_correct = (
        frame.session_correct.to_numpy(dtype=float)[part] for part in (train_i, val_i, test_i)
    )
    item_count = max(item_to_index.values(), default=1) + 1

    # Same-task, transparent baselines use only pre-response numeric features.
    train_features = _feature_matrix(train_x, train_l)
    val_features = _feature_matrix(val_x, val_l)
    test_features = _feature_matrix(test_x, test_l)
    train_mean = np.average(train_y, weights=train_seen)
    mean_predictions = np.full(len(test_i), train_mean)
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=20,
        random_state=args.seed, n_jobs=-1,
    ).fit(train_features, train_y, sample_weight=train_seen)
    print(f"Fitted same-task Random Forest baseline in {time.time() - started:.1f}s.", flush=True)
    rf_predictions = np.clip(rf.predict(test_features), 0, 1)

    model, scaler, best_epoch, best_val_loss, training_history = train_deep(
        args.architecture, train_x, train_l, train_y, train_seen,
        val_x, val_l, val_y, val_seen, args.seed, args.max_epochs,
        args.batch_size, item_count, device, args.optimizer,
    )
    val_predictions = predict_deep(model, scaler, val_x, val_l, device)
    test_predictions = predict_deep(model, scaler, test_x, test_l, device)
    metadata = {
        "architecture": args.architecture,
        "optimizer": args.optimizer,
        "target": "next interaction session_correct / session_seen",
        "target_is_delayed_recall": False,
        "source_dataset": "Duolingo Learning Traces",
        "sample_modulus": args.sample_modulus,
        "hash_key": USER_HASH_KEY,
        "rows_scanned": rows_read,
        "examples": len(frame),
        "learners": int(frame.user_id.nunique()),
        "items_in_train_vocabulary": len(item_to_index),
        "oov_item_index": 1,
        "padding_index": 0,
        "sequence_length": args.sequence_length,
        "numeric_features": list(NUMERIC_FEATURES),
        "protocol": args.protocol,
        "split_boundaries": split_boundaries,
        "split_sizes": {"train": len(train_i), "validation": len(val_i), "test": len(test_i)},
        "split_learners": {
            name: int(frame.user_id.iloc[index].nunique())
            for name, index in zip(("train", "validation", "test"), split)
        },
        "seed": args.seed,
        "requested_max_epochs": args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_binomial_loss": best_val_loss,
        "training_history": training_history,
        "device": str(device),
        "selection": "early stopping on validation weighted binomial log loss; report metrics never select the checkpoint",
        "metrics": {
            "mean_baseline_test": weighted_metrics(test_y, mean_predictions, test_seen, test_correct),
            "random_forest_test": weighted_metrics(test_y, rf_predictions, test_seen, test_correct),
            "deep_validation": weighted_metrics(val_y, val_predictions, val_seen, val_correct),
            "deep_test": weighted_metrics(test_y, test_predictions, test_seen, test_correct),
        },
        "limitations": [
            "This aggregated next-interaction target is not delayed retention.",
            "Duolingo lexeme IDs do not map to app course topics or uploaded materials.",
            "A single seed and deterministic learner sample are preliminary.",
        ],
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    optimizer_suffix = "" if args.optimizer == "adamw" else f"_{args.optimizer}"
    model_path = MODEL_DIR / f"knowledge_tracer_{args.architecture}{optimizer_suffix}_{args.protocol}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "item_count": item_count,
        "architecture": args.architecture,
        "embedding_size": 32,
        "hidden_size": 64,
        "dropout": 0.2,
        "sequence_length": args.sequence_length,
        "numeric_features": list(NUMERIC_FEATURES),
        "item_to_index": item_to_index,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "metadata": metadata,
    }, model_path)
    metadata["artifact"] = str(model_path.relative_to(ROOT)).replace("\\", "/")
    metadata["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"knowledge_tracing_{args.architecture}{optimizer_suffix}_{args.protocol}_evaluation.json"
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to {report_path.relative_to(ROOT)}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
