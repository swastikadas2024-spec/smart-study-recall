
from pathlib import Path
import json
import random

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

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml/data/raw/settles.acl16.learning_traces.13m.csv.gz"
MODEL_DIR = ROOT / "ml/models"
MODEL_PATH = MODEL_DIR / "recall_gru_v2.pt"
METADATA_PATH = MODEL_DIR / "recall_gru_v2_metadata.json"

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def calculate_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# --------------------------------------------------
# GRU model
# --------------------------------------------------
class RecallGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64):
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
# Load and validate data
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

    df = df[
        (df["history_seen"] >= 0)
        & (df["history_correct"] >= 0)
        & (df["history_correct"] <= df["history_seen"])
        & (df["session_seen"] > 0)
        & (df["session_correct"] >= 0)
        & (df["session_correct"] <= df["session_seen"])
        & (df["delta"] >= 0)
    ].copy()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df.dropna(subset=["timestamp", "delta"]).copy()

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

    # Stable source order is only a tie-breaker, not proof of
    # the true order of simultaneous events.
    df["_source_order"] = np.arange(len(df))

    # Repeated timestamps for one learner-item pair cannot be
    # reliably ordered from timestamp alone. Exclude the entire
    # tied timestamp group from this experiment.
    tie_counts = df.groupby(
        ["user_id", "lexeme_id", "timestamp"]
    )["timestamp"].transform("size")

    tied_rows = int((tie_counts > 1).sum())
    df = df[tie_counts == 1].copy()

    df = df.sort_values(
        ["timestamp", "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    print("Valid rows after filtering:", len(df))
    print("Rows excluded due to tied learner-item timestamps:", tied_rows)
    print("Unique learners:", df["user_id"].nunique())
    print("Unique learner-item pairs:", len(
        df[["user_id", "lexeme_id"]].drop_duplicates()
    ))

    return df


# --------------------------------------------------
# Build sequences within learner-item pairs
# --------------------------------------------------
def build_sequences(df):
    print("Building learner-item sequences...")

    X_list = []
    y_list = []
    times = []
    baseline_features = []

    for _, group in df.groupby(
        ["user_id", "lexeme_id"], sort=False
    ):
        group = group.sort_values(
            ["timestamp", "_source_order"],
            kind="mergesort",
        )

        values = group[FEATURES].to_numpy(dtype=np.float32)
        targets = group["target"].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)

        for i in range(len(group)):
            start = max(0, i - SEQUENCE_LENGTH + 1)
            sequence = values[start:i + 1]

            if len(sequence) < SEQUENCE_LENGTH:
                padding = np.repeat(
                    sequence[:1],
                    SEQUENCE_LENGTH - len(sequence),
                    axis=0,
                )
                sequence = np.concatenate(
                    [padding, sequence], axis=0
                )

            X_list.append(sequence)
            y_list.append(targets[i])
            times.append(timestamps[i])
            baseline_features.append(values[i])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    baseline_features = np.asarray(
        baseline_features, dtype=np.float32
    )

    print("Sequences:", len(X))
    print("Sequence shape:", X.shape)

    return X, y, times, baseline_features


# --------------------------------------------------
# Main training and evaluation
# --------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, times, flat_features = build_sequences(df)

    # Chronological split by target event time.
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

    time_cut_train = float(times[train_end - 1])
    time_cut_val = float(times[val_end - 1])

    print("Train sequences:", len(X_train))
    print("Validation sequences:", len(X_val))
    print("Test sequences:", len(X_test))

    # Fit scaler only on training data.
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
    print("Training GRU v2 on", device)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = RecallGRU(input_size=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device))
                val_losses.append(
                    loss_fn(pred, yb.to(device)).item()
                )

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
            xb = torch.tensor(
                X_test[start:start + BATCH_SIZE]
            ).to(device)
            gru_predictions.extend(
                model(xb).cpu().numpy().tolist()
            )

    gru_predictions = np.asarray(gru_predictions)

    # RF-1 comparison on the same test target rows.
    rf_path = MODEL_DIR / "recall_rf1.joblib"
    if rf_path.exists():
        import joblib

        rf_bundle = joblib.load(rf_path)
        rf_model = rf_bundle["model"]
        rf_predictions = rf_model.predict(flat_test)
        rf_metrics = calculate_metrics(y_test, rf_predictions)
    else:
        rf_metrics = None

    mean_prediction = np.full(
        len(y_test), float(np.mean(y_train))
    )

    gru_metrics = calculate_metrics(y_test, gru_predictions)
    mean_metrics = calculate_metrics(y_test, mean_prediction)

    print("\n--- Test metrics ---")
    print("GRU v2:", gru_metrics)
    print("Mean baseline:", mean_metrics)
    print("RF-1:", rf_metrics)

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
        MODEL_PATH,
    )

    metadata = {
        "model_name": "GRU v2 learner-item sequence regressor",
        "framework": "PyTorch",
        "rows_loaded": ROWS_TO_LOAD,
        "valid_rows_after_filtering": len(df),
        "sequence_count": len(X),
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
        "target": "session_correct / session_seen",
        "grouping": ["user_id", "lexeme_id"],
        "tie_handling": (
            "All rows from learner-item groups with repeated "
            "timestamps were excluded because their event order "
            "could not be determined."
        ),
        "split": "70/15/15 chronological by target timestamp",
        "train_max_timestamp": time_cut_train,
        "validation_max_timestamp": time_cut_val,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "gru_v2_test": gru_metrics,
        "mean_baseline_test": mean_metrics,
        "rf1_test": rf_metrics,
        "limitations": [
            "Uses a 200,000-row dataset prefix, not the full dataset.",
            "Chronological test split is not user-independent.",
            "The target is observed session-level recall rate, not a validated future-retention probability.",
            "Historical features come from source-provided aggregates.",
            "A row-prefix sample may not represent the full dataset.",
        ],
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model:", MODEL_PATH)
    print("Saved metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()