
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
ROWS_TO_LOAD = 200_000
SEQUENCE_LENGTH = 10
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ml" / "data" / "raw" / "settles.acl16.learning_traces.13m.csv.gz"
MODEL_DIR = ROOT / "ml" / "models"

GRU_PATH = MODEL_DIR / "recall_gru_v3.pt"
GRU_METADATA_PATH = MODEL_DIR / "recall_gru_v3_metadata.json"
RF_PATH = MODEL_DIR / "recall_rf_v3_baseline.joblib"

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# --------------------------------------------------
# Deep learning model
# --------------------------------------------------
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


# --------------------------------------------------
# Load and validate the dataset
# --------------------------------------------------
def load_data():
    print("Loading dataset sample...")

    columns = [
        "user_id",
        "lexeme_id",
        "timestamp",
        "delta",
        "history_seen",
        "history_correct",
        "session_seen",
        "session_correct",
    ]

    df = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        nrows=ROWS_TO_LOAD,
        usecols=columns,
    )

    df = df.dropna(subset=columns).copy()

    numeric_cols = [
        "timestamp", "delta", "history_seen", "history_correct",
        "session_seen", "session_correct",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    df = df[
        (df["history_seen"] >= 0)
        & (df["history_correct"] >= 0)
        & (df["history_correct"] <= df["history_seen"])
        & (df["session_seen"] > 0)
        & (df["session_correct"] >= 0)
        & (df["session_correct"] <= df["session_seen"])
        & (df["delta"] >= 0)
    ].copy()

    df["target"] = (
        df["session_correct"] / df["session_seen"]
    ).astype("float32")

    df["historical_accuracy"] = np.where(
        df["history_seen"] > 0,
        df["history_correct"] / df["history_seen"].clip(lower=1),
        0.5,
    )

    df["log_history_seen"] = np.log1p(df["history_seen"])
    df["delta_days"] = df["delta"] / 86400.0
    df["log_delta_days"] = np.log1p(df["delta_days"])

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES + ["target"]).copy()

    # Preserve original file order as a tie-breaker.
    df["_source_order"] = np.arange(len(df))

    # Exclude ambiguous learner-item event ordering.
    tie_counts = df.groupby(
        ["user_id", "lexeme_id", "timestamp"]
    )["timestamp"].transform("size")

    tied_rows = int((tie_counts > 1).sum())
    df = df[tie_counts == 1].copy()

    df = df.sort_values(
        ["timestamp", "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    print("Valid rows:", len(df))
    print("Rows excluded for tied learner-item timestamps:", tied_rows)
    print("Unique learners:", df["user_id"].nunique())
    return df


# --------------------------------------------------
# Build past-only sequences
# --------------------------------------------------
def build_sequences(df):
    print("Building sequences using only current and past events...")

    features_by_row = df[FEATURES].to_numpy(dtype=np.float32)
    targets = df["target"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)

    X = np.zeros(
        (len(df), SEQUENCE_LENGTH, len(FEATURES)),
        dtype=np.float32,
    )

    # Group indices by learner and concept. Each sequence ends
    # at its own row; no later event is included in that sequence.
    groups = df.groupby(
        ["user_id", "lexeme_id"], sort=False
    ).indices

    for indices in groups.values():
        indices = np.asarray(indices)
        indices = indices[
            np.argsort(timestamps[indices], kind="stable")
        ]

        for position, row_index in enumerate(indices):
            start_position = max(0, position - SEQUENCE_LENGTH + 1)
            history_indices = indices[start_position:position + 1]
            sequence = features_by_row[history_indices]

            if len(sequence) < SEQUENCE_LENGTH:
                padding = np.repeat(
                    sequence[:1],
                    SEQUENCE_LENGTH - len(sequence),
                    axis=0,
                )
                sequence = np.concatenate(
                    [padding, sequence], axis=0
                )

            X[row_index] = sequence

    return X, targets, timestamps, features_by_row


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Never overwrite existing model artifacts.
    for path in [GRU_PATH, GRU_METADATA_PATH, RF_PATH]:
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Choose new output filenames "
                "before running; existing files were not changed."
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, times, flat_features = build_sequences(df)

    # Chronological 70/15/15 split.
    order = np.argsort(times, kind="stable")
    X, y, times, flat_features = (
        X[order], y[order], times[order], flat_features[order]
    )

    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    flat_train = flat_features[:train_end]
    flat_test = flat_features[val_end:]

    time_train_end = float(times[train_end - 1])
    time_val_end = float(times[val_end - 1])

    print("\nCHRONOLOGICAL SPLIT")
    print("=" * 55)
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Train end timestamp:", time_train_end)
    print("Validation end timestamp:", time_val_end)

    # Fit scaler using training sequences only.
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, len(FEATURES)))

    def scale_sequences(arr):
        shape = arr.shape
        scaled = scaler.transform(
            arr.reshape(-1, len(FEATURES))
        )
        return scaled.reshape(shape).astype(np.float32)

    X_train = scale_sequences(X_train)
    X_val = scale_sequences(X_val)
    X_test = scale_sequences(X_test)

    device = torch.device("cpu")
    model = RecallGRU().to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = PATIENCE

    print("\nTRAINING GRU V3")
    print("=" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            predictions = model(xb.to(device))
            loss = loss_fn(predictions, yb.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                predictions = model(xb.to(device))
                loss = loss_fn(predictions, yb.to(device))
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_mse={train_loss:.6f} | "
            f"val_mse={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        gru_predictions = []
        for start in range(0, len(X_test), BATCH_SIZE):
            xb = torch.tensor(X_test[start:start + BATCH_SIZE])
            gru_predictions.extend(
                model(xb).cpu().numpy().tolist()
            )

    gru_predictions = np.clip(
        np.asarray(gru_predictions), 0, 1
    )

    # Train a fresh RF on exactly the same training rows.
    # This avoids comparing GRU test rows with an RF that may
    # already have trained on those events.
    print("\nTRAINING MATCHED RANDOM FOREST BASELINE")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        random_state=SEED,
        n_jobs=-1,
    )
    rf_model.fit(flat_train, y_train)
    rf_predictions = np.clip(rf_model.predict(flat_test), 0, 1)

    mean_predictions = np.full(len(y_test), float(y_train.mean()))

    gru_metrics = metrics(y_test, gru_predictions)
    rf_metrics = metrics(y_test, rf_predictions)
    baseline_metrics = metrics(y_test, mean_predictions)

    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    print("GRU v3:", gru_metrics)
    print("Matched Random Forest:", rf_metrics)
    print("Mean baseline:", baseline_metrics)
    print("Best GRU epoch:", best_epoch)

    # Save distinct artifacts.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(FEATURES),
            "hidden_size": 64,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_names": FEATURES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        GRU_PATH,
    )

    joblib.dump(
        {
            "model": rf_model,
            "features": FEATURES,
            "target": "session_correct / session_seen",
            "target_type": "session-level recall rate",
        },
        RF_PATH,
    )

    metadata = {
        "model_name": "GRU v3 learner-item sequence regressor",
        "framework": "PyTorch",
        "rows_loaded": ROWS_TO_LOAD,
        "valid_rows": len(df),
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
        "target": "session_correct / session_seen",
        "grouping": ["user_id", "lexeme_id"],
        "split": "70/15/15 chronological",
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "gru_test": gru_metrics,
        "matched_random_forest_test": rf_metrics,
        "mean_baseline_test": baseline_metrics,
        "limitations": [
            "Uses a 200,000-row source prefix rather than the full dataset.",
            "The test set is not user-independent.",
            "Target is observed session-level recall, not validated future retention probability.",
            "A prefix sample may not represent the entire dataset.",
        ],
    }

    with open(GRU_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved GRU:", GRU_PATH)
    print("Saved GRU metadata:", GRU_METADATA_PATH)
    print("Saved matched RF baseline:", RF_PATH)
    print("Existing GRU v2 and RF-1 artifacts were not modified.")


if __name__ == "__main__":
    main()
