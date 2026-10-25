"""Train/evaluate models for predicting the next observed session score.

Task: given a learner-item's interactions through the current session and a
candidate gap from that session to the next one, predict the next session's
correct/seen fraction. This does not choose the candidate review gap.

Run from the repository root:
  python ml/src/train_next_session_model.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from next_session_model import (  # noqa: E402
    FEATURES, MAX_CANDIDATE_GAP_DAYS, NextSessionGRU, SEQUENCE_LENGTH,
)

DATA_PATH = ROOT / "ml/data/raw/settles.acl16.learning_traces.13m.csv.gz"
MODEL_DIR = ROOT / "ml/models"
REPORT_PATH = ROOT / "ml/reports/next_session_evaluation.json"
SEEDS = [42, 52, 62]
USER_HASH_KEY = "smartstudyrecall"
USER_SAMPLE_MODULUS = 100  # retain all events for a deterministic 1-in-100 learner sample
CHUNK_SIZE = 250_000
MAX_EPOCHS = 12
PATIENCE = 2
BATCH_SIZE = 512

RAW_COLUMNS = [
    "user_id", "lexeme_id", "timestamp", "history_seen", "history_correct",
    "session_seen", "session_correct",
]


def load_learner_sample() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    retained = []
    rows_read = 0
    for chunk in pd.read_csv(
        DATA_PATH, compression="gzip", usecols=RAW_COLUMNS, chunksize=CHUNK_SIZE
    ):
        rows_read += len(chunk)
        chunk["user_id"] = chunk["user_id"].astype(str)
        user_hash = pd.util.hash_pandas_object(
            chunk["user_id"], index=False, hash_key=USER_HASH_KEY
        ).to_numpy(dtype=np.uint64)
        chunk = chunk[user_hash % USER_SAMPLE_MODULUS == 0].copy()
        if chunk.empty:
            continue
        numeric = RAW_COLUMNS[2:]
        for column in numeric:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=numeric + ["lexeme_id"])
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
        if rows_read % (CHUNK_SIZE * 4) == 0:
            print(f"Scanned {rows_read:,} rows; retained {sum(map(len, retained)):,} sampled-learner rows.", flush=True)

    if not retained:
        raise ValueError("No valid events found in the sampled learners.")
    frame = pd.concat(retained, ignore_index=True)
    frame["lexeme_id"] = frame["lexeme_id"].astype(str)
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)

    # Simultaneous events for one learner-item cannot be ordered from this
    # dataset, so omit all events in each ambiguous tie group.
    tied = frame.duplicated(["user_id", "lexeme_id", "timestamp"], keep=False)
    frame = frame.loc[~tied].copy()
    frame.sort_values(
        ["user_id", "lexeme_id", "timestamp", "_source_order"],
        kind="mergesort", inplace=True,
    )

    groups = frame.groupby(["user_id", "lexeme_id"], sort=False)
    frame["next_timestamp"] = groups.timestamp.shift(-1)
    frame["next_session_seen"] = groups.session_seen.shift(-1)
    frame["next_session_correct"] = groups.session_correct.shift(-1)
    frame["candidate_gap_days"] = (
        frame.next_timestamp - frame.timestamp
    ) / 86400.0
    frame["target_next_recall"] = (
        frame.next_session_correct / frame.next_session_seen
    )

    frame["updated_seen"] = frame.history_seen + frame.session_seen
    frame["updated_correct"] = frame.history_correct + frame.session_correct
    frame["updated_accuracy"] = frame.updated_correct / frame.updated_seen
    frame["session_accuracy"] = frame.session_correct / frame.session_seen
    frame["log_updated_seen"] = np.log1p(frame.updated_seen)
    frame["log_session_seen"] = np.log1p(frame.session_seen)
    frame["log_candidate_gap_days"] = np.log1p(frame.candidate_gap_days)

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(
        subset=FEATURES + ["target_next_recall", "next_timestamp", "candidate_gap_days"],
        inplace=True,
    )
    frame = frame[
        (frame.candidate_gap_days > 0)
        & (frame.next_session_seen > 0)
        & (frame.target_next_recall >= 0)
        & (frame.target_next_recall <= 1)
    ].reset_index(drop=True)
    print(
        f"Loaded {rows_read:,} source rows; {len(frame):,} valid next-session pairs "
        f"from {frame.user_id.nunique():,} sampled learners after tie handling.",
        flush=True,
    )
    return frame


def build_sequences(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    sequences = np.zeros((len(frame), SEQUENCE_LENGTH, len(FEATURES)), dtype=np.float32)
    groups = frame.groupby(["user_id", "lexeme_id"], sort=False).indices
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        for position, row_index in enumerate(indices):
            history = indices[max(0, position - SEQUENCE_LENGTH + 1):position + 1]
            rows = values[history]
            if len(rows) < SEQUENCE_LENGTH:
                rows = np.concatenate(
                    [np.repeat(rows[:1], SEQUENCE_LENGTH - len(rows), axis=0), rows],
                    axis=0,
                )
            sequences[row_index] = rows
    return sequences


def make_splits(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    eligible = np.flatnonzero(
        (frame.candidate_gap_days.to_numpy() <= MAX_CANDIDATE_GAP_DAYS)
        & (frame.candidate_gap_days.to_numpy() > 0)
    )
    # Forward-in-time: earlier target events train, later events validate/test.
    time_order = eligible[np.argsort(frame.next_timestamp.to_numpy()[eligible], kind="stable")]
    n = len(time_order)
    time_split = (
        time_order[:int(n * 0.70)],
        time_order[int(n * 0.70):int(n * 0.85)],
        time_order[int(n * 0.85):],
    )

    # Learner-held-out: no learner appears in more than one partition.
    rng = np.random.default_rng(2026)
    users = frame.user_id.to_numpy()[eligible]
    unique_users = np.unique(users)
    rng.shuffle(unique_users)
    test_count = max(1, int(len(unique_users) * 0.15))
    validation_count = max(1, int(len(unique_users) * 0.15))
    test_users = set(unique_users[:test_count])
    validation_users = set(unique_users[test_count:test_count + validation_count])
    split_users = users
    group_split = (
        eligible[~np.isin(split_users, list(test_users | validation_users))],
        eligible[np.isin(split_users, list(validation_users))],
        eligible[np.isin(split_users, list(test_users))],
    )
    return {"forward_time": time_split, "learner_held_out": group_split}


def fit_hlr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # H = exp(b0 + b1*log(1+seen) + b2*updated_accuracy),
    # p(next recall) = 2^(-candidate_gap / H). Parameters use train data only.
    z = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
    gap = np.expm1(x[:, 4])
    ln2 = np.log(2.0)

    def objective(beta):
        log_half_life = np.clip(z @ beta, -12, 12)
        ratio = gap * np.exp(-log_half_life)
        pred = np.exp(-ln2 * ratio)
        residual = pred - y
        derivative = pred * ln2 * ratio
        gradient = 2 * (z.T @ (residual * derivative)) / len(y)
        return float(np.mean(residual ** 2)), gradient

    fit = minimize(
        objective, np.array([0.0, 0.25, 1.0]), jac=True, method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-8},
    )
    if not fit.success and not np.isfinite(fit.fun):
        raise RuntimeError(f"HLR baseline fitting failed: {fit.message}")
    return fit.x.astype(float)


def predict_hlr(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
    gap = np.expm1(x[:, 4])
    half_life = np.exp(np.clip(z @ beta, -12, 12))
    return np.clip(np.exp(-np.log(2.0) * gap / half_life), 0, 1)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    edges = np.linspace(0, 1, 11)
    bin_ids = np.minimum(np.digitize(pred, edges[1:-1]), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bin_ids == bin_index
        if mask.any():
            ece += float(mask.mean() * abs(float(pred[mask].mean() - y[mask].mean())))
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "ece_10_equal_width": ece,
    }


def cluster_bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, users: np.ndarray, repeats: int = 300
) -> dict[str, list[float]]:
    unique_users, inverse = np.unique(users, return_inverse=True)
    clusters = len(unique_users)
    counts = np.bincount(inverse, minlength=clusters).astype(float)
    absolute = np.bincount(inverse, weights=np.abs(y - pred), minlength=clusters)
    squared = np.bincount(inverse, weights=(y - pred) ** 2, minlength=clusters)
    bin_ids = np.minimum(np.digitize(pred, np.linspace(0, 1, 11)[1:-1]), 9)
    bin_counts = np.stack([
        np.bincount(inverse, weights=(bin_ids == j).astype(float), minlength=clusters)
        for j in range(10)
    ], axis=1)
    bin_pred = np.stack([
        np.bincount(inverse, weights=pred * (bin_ids == j), minlength=clusters)
        for j in range(10)
    ], axis=1)
    bin_actual = np.stack([
        np.bincount(inverse, weights=y * (bin_ids == j), minlength=clusters)
        for j in range(10)
    ], axis=1)

    rng = np.random.default_rng(20261024)
    samples = {"mae": [], "rmse": [], "ece_10_equal_width": []}
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, clusters, size=clusters), minlength=clusters)
        total = float(weights @ counts)
        samples["mae"].append(float(weights @ absolute / total))
        samples["rmse"].append(float(np.sqrt((weights @ squared) / total)))
        c = weights @ bin_counts
        p = weights @ bin_pred
        a = weights @ bin_actual
        nonzero = c > 0
        ece = np.sum((c[nonzero] / total) * np.abs(p[nonzero] / c[nonzero] - a[nonzero] / c[nonzero]))
        samples["ece_10_equal_width"].append(float(ece))
    return {key: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] for key, vals in samples.items()}


def fit_gru(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
    seed: int, epochs: int = MAX_EPOCHS,
) -> tuple[torch.nn.Module, StandardScaler, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train_x.reshape(-1, len(FEATURES)))

    def scale(x):
        return scaler.transform(x.reshape(-1, len(FEATURES))).reshape(x.shape).astype(np.float32)

    train_x = scale(train_x)
    val_x = scale(val_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextSessionGRU().build().to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(val_y.astype(np.float32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(val_tensor), val_y_tensor).item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_epoch


def predict_gru(model, scaler, sequences: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    scaled = scaler.transform(sequences.reshape(-1, len(FEATURES))).reshape(sequences.shape).astype(np.float32)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            predictions.extend(model(torch.from_numpy(scaled[start:start + batch_size]).to(device)).cpu().numpy())
    return np.clip(np.asarray(predictions, dtype=float), 0, 1)


def run_comparison(frame, sequences, split_name, partitions):
    train_i, val_i, test_i = partitions
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame.target_next_recall.to_numpy(dtype=np.float32)
    results = {}
    validation_runs = {}
    test_runs = {}
    gru_best_epochs = []

    for seed in SEEDS:
        print(f"{split_name}: fitting seed {seed} on {len(train_i):,} rows", flush=True)
        mean_pred = np.full(len(val_i), float(y[train_i].mean()))
        validation_runs.setdefault("mean", []).append((val_i, mean_pred))
        test_runs.setdefault("mean", []).append((test_i, np.full(len(test_i), float(y[train_i].mean()))))

        print(f"{split_name} seed {seed}: fitting HLR", flush=True)
        hlr = fit_hlr(x[train_i], y[train_i])
        validation_runs.setdefault("hlr", []).append((val_i, predict_hlr(x[val_i], hlr)))
        test_runs.setdefault("hlr", []).append((test_i, predict_hlr(x[test_i], hlr)))

        print(f"{split_name} seed {seed}: fitting linear regression", flush=True)
        linear = LinearRegression().fit(x[train_i], y[train_i])
        validation_runs.setdefault("linear_regression", []).append((val_i, np.clip(linear.predict(x[val_i]), 0, 1)))
        test_runs.setdefault("linear_regression", []).append((test_i, np.clip(linear.predict(x[test_i]), 0, 1)))

        print(f"{split_name} seed {seed}: fitting random forest", flush=True)
        forest = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            random_state=seed, n_jobs=-1,
        ).fit(x[train_i], y[train_i])
        validation_runs.setdefault("random_forest", []).append((val_i, np.clip(forest.predict(x[val_i]), 0, 1)))
        test_runs.setdefault("random_forest", []).append((test_i, np.clip(forest.predict(x[test_i]), 0, 1)))

        print(f"{split_name} seed {seed}: fitting GRU", flush=True)
        gru, scaler, best_epoch = fit_gru(
            sequences[train_i], y[train_i], sequences[val_i], y[val_i], seed
        )
        gru_best_epochs.append(best_epoch)
        validation_runs.setdefault("gru", []).append((val_i, predict_gru(gru, scaler, sequences[val_i])))
        test_runs.setdefault("gru", []).append((test_i, predict_gru(gru, scaler, sequences[test_i])))

    scores = {}
    for name in validation_runs:
        val_predictions = np.mean([pred for _, pred in validation_runs[name]], axis=0)
        test_predictions = np.mean([pred for _, pred in test_runs[name]], axis=0)
        scores[name] = {
            "validation_ensemble_mean": metrics(y[val_i], val_predictions),
            "test_ensemble_mean": metrics(y[test_i], test_predictions),
            "test_learner_cluster_bootstrap_95ci": cluster_bootstrap_ci(
                y[test_i], test_predictions, frame.user_id.to_numpy()[test_i]
            ),
            "test_per_seed": [
                metrics(y[test_i], pred) for _, pred in test_runs[name]
            ],
        }
    return scores, gru_best_epochs


def save_champion(frame, sequences, splits, results, gru_epochs):
    split_summary = {}
    all_users = frame.user_id.to_numpy()
    for split_name, (train_i, val_i, test_i) in splits.items():
        split_summary[split_name] = {
            "train_examples": int(len(train_i)),
            "validation_examples": int(len(val_i)),
            "test_examples": int(len(test_i)),
            "train_learners": int(frame.user_id.iloc[train_i].nunique()),
            "validation_learners": int(frame.user_id.iloc[val_i].nunique()),
            "test_learners": int(frame.user_id.iloc[test_i].nunique()),
            "learner_sets_disjoint": (
                not bool(set(all_users[train_i]) & set(all_users[val_i]))
                and not bool(set(all_users[train_i]) & set(all_users[test_i]))
                and not bool(set(all_users[val_i]) & set(all_users[test_i]))
            ) if split_name == "learner_held_out" else None,
        }
    names = ["mean", "hlr", "linear_regression", "random_forest", "gru"]
    validation_mae = {
        name: float(np.mean([
            results[split][name]["validation_ensemble_mean"]["mae"]
            for split in results
        ]))
        for name in names
    }
    champion = min(validation_mae, key=validation_mae.get)
    training_domain = frame.candidate_gap_days.to_numpy() <= MAX_CANDIDATE_GAP_DAYS
    domain_examples = int(training_domain.sum())
    total_examples = int(len(frame))
    frame = frame.loc[training_domain].reset_index(drop=True)
    sequences = sequences[training_domain]
    y = frame.target_next_recall.to_numpy(dtype=np.float32)
    x = frame[FEATURES].to_numpy(dtype=np.float32)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if champion == "mean":
        artifact = {"kind": champion, "value": float(y.mean()), "features": FEATURES}
    elif champion == "hlr":
        artifact = {"kind": champion, "parameters": fit_hlr(x, y).tolist(), "features": FEATURES}
    elif champion == "linear_regression":
        artifact = {"kind": champion, "model": LinearRegression().fit(x, y), "features": FEATURES}
    elif champion == "random_forest":
        artifact = {
            "kind": champion,
            "model": RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            ).fit(x, y),
            "features": FEATURES,
        }
    else:
        # Refit on all retained users only after evaluation/selection is done.
        scaler = StandardScaler().fit(sequences.reshape(-1, len(FEATURES)))
        scaled = scaler.transform(sequences.reshape(-1, len(FEATURES))).reshape(sequences.shape).astype(np.float32)
        model = NextSessionGRU().build()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(scaled), torch.from_numpy(y)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        final_epochs = max(1, int(np.median(gru_epochs)))
        for _ in range(final_epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()
        torch.save({
            "state_dict": model.state_dict(),
            "input_size": len(FEATURES),
            "hidden_size": 64,
            "sequence_length": SEQUENCE_LENGTH,
            "features": FEATURES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, MODEL_DIR / "next_session_champion.pt")
        artifact = {"kind": champion, "model_path": "next_session_champion.pt", "features": FEATURES}

    joblib.dump(artifact, MODEL_DIR / "next_session_champion.joblib")
    metadata = {
        "task": "predict the next observed session correct/seen fraction, conditional on a specified gap from the current session",
        "target": "next_session_correct / next_session_seen",
        "features": FEATURES,
        "maximum_candidate_gap_days": MAX_CANDIDATE_GAP_DAYS,
        "sequence_length": SEQUENCE_LENGTH,
        "source_dataset": "Duolingo Learning Traces; full source streamed, deterministic 1-in-N learner hash sample retained",
        "learner_hash_sample_modulus": USER_SAMPLE_MODULUS,
        "learners": int(frame.user_id.nunique()),
        "eligible_examples_gap_at_most_30_days": domain_examples,
        "total_examples_with_next_session": total_examples,
        "seeds": SEEDS,
        "evaluation_protocols": {
            "forward_time": "chronological 70/15/15 target-event split; estimates future events for the sampled population, with prior history available",
            "learner_held_out": "70/15/15 disjoint learner IDs; estimates performance for unseen learners who provide their own interaction history at inference",
        },
        "split_sizes": split_summary,
        "metrics": "MAE and RMSE for the continuous target; ECE with 10 equal-width bins; 95% percentile bootstrap intervals resampling learners as clusters (300 replicates) on ensemble-mean test predictions",
        "validation_mae_by_model": validation_mae,
        "selected_champion": champion,
        "selection_rule": "lowest mean validation MAE across both protocols; test metrics are not used to choose the deployed model",
        "results": results,
        "caveats": [
            "This target is a source dataset's next-session score, not a measured long-term memory probability.",
            "The source is a language-learning task; transfer to course topics and material quizzes is unvalidated.",
            "This is observational next-event prediction, not a causal estimate of the effect of changing review time.",
            f"A 1-in-{USER_SAMPLE_MODULUS} learner sample reduces runtime; report its deterministic sampling method and repeat evaluation with other hash buckets before broad claims.",
            "The schedule policy is a separate rule: choose the latest supported candidate gap whose predicted next-session score meets the chosen threshold; it is not trained as a schedule optimizer.",
        ],
    }
    (MODEL_DIR.parent / "reports").mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validation-selected champion: {champion} ({validation_mae[champion]:.5f} mean validation MAE)")
    print(f"Saved model to {MODEL_DIR / 'next_session_champion.joblib'}")
    print(f"Saved evaluation to {REPORT_PATH}")


def main():
    global USER_SAMPLE_MODULUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-sample-modulus", type=int, default=USER_SAMPLE_MODULUS,
                        help="Retain all rows for 1/N deterministically hashed learners; default 20.")
    args = parser.parse_args()
    if args.user_sample_modulus < 2:
        parser.error("--user-sample-modulus must be at least 2")
    USER_SAMPLE_MODULUS = args.user_sample_modulus

    started = time.time()
    frame = load_learner_sample()
    sequences = build_sequences(frame)
    partitions = make_splits(frame)
    results = {}
    epochs = []
    for split_name, split in partitions.items():
        results[split_name], split_epochs = run_comparison(frame, sequences, split_name, split)
        epochs.extend(split_epochs)
        print(f"{split_name} test metrics:", json.dumps({k: v["test_ensemble_mean"] for k, v in results[split_name].items()}, indent=2))
    save_champion(frame, sequences, partitions, results, epochs)
    print(f"Completed in {(time.time() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
