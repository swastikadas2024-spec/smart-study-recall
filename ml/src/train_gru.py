
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "raw" / "settles.acl16.learning_traces.13m.csv.gz"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "recall_gru.pt"
METADATA_PATH = MODEL_DIR / "recall_gru_metadata.json"

NROWS = 200_000
SEQUENCE_LENGTH = 10
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3
SEED = 42

FEATURES = [
    "log_history_seen",
    "historical_accuracy",
    "delta_days",
    "log_delta_days",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

print("Loading dataset sample...")

df = pd.read_csv(
    DATA_PATH,
    compression="gzip",
    nrows=NROWS,
)

required = [
    "user_id",
    "timestamp",
    "delta",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Preserve original file order when timestamps tie.
df["_source_order"] = np.arange(len(df))

numeric_columns = [
    "timestamp",
    "delta",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required).copy()

df = df[
    (df["user_id"].notna())
    & (df["timestamp"] > 0)
    & (df["delta"] > 0)
    & (df["history_seen"] > 0)
    & (df["history_correct"] >= 0)
    & (df["history_correct"] <= df["history_seen"])
    & (df["session_seen"] > 0)
    & (df["session_correct"] >= 0)
    & (df["session_correct"] <= df["session_seen"])
].copy()

df["delta_days"] = df["delta"] / 86400.0
df["log_delta_days"] = np.log1p(df["delta_days"])
df["log_history_seen"] = np.log1p(df["history_seen"])
df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"]
)

# The target belongs to the current session.
df["target_recall"] = (
    df["session_correct"] / df["session_seen"]
)

# Stable ordering preserves source order for timestamp ties.
df = df.sort_values(
    ["user_id", "timestamp", "_source_order"],
    kind="mergesort",
).reset_index(drop=True)

print("Valid rows:", len(df))
print("Unique learners:", df["user_id"].nunique())


# --------------------------------------------------
# Build fixed-length learner sequences
# --------------------------------------------------

print("Building learner sequences...")

feature_values = df[FEATURES].to_numpy(dtype=np.float32)
targets = df["target_recall"].to_numpy(dtype=np.float32)
timestamps = df["timestamp"].to_numpy(dtype=np.float64)

sequences = []
labels = []
target_times = []
baseline_predictions = []

# Each example includes the target row's historical features,
# which are available before its session, plus up to nine
# preceding events for the same learner.
#
# No target-session correctness values enter the inputs.

for _, group in df.groupby("user_id", sort=False).indices.items():
    indices = np.asarray(group, dtype=np.int64)

    if len(indices) < 2:
        continue

    for position in range(1, len(indices)):
        target_index = indices[position]

        start = max(0, position - SEQUENCE_LENGTH + 1)
        window_indices = indices[start : position + 1]

        window = feature_values[window_indices]

        # Left-pad short histories by repeating the first
        # available feature row.
        if len(window) < SEQUENCE_LENGTH:
            padding = np.repeat(
                window[:1],
                SEQUENCE_LENGTH - len(window),
                axis=0,
            )
            window = np.concatenate([padding, window], axis=0)

        sequences.append(window)
        labels.append(targets[target_index])
        target_times.append(timestamps[target_index])

        # RF-1 baseline uses the same target row's
        # pre-session features.
        baseline_predictions.append(
            [
                feature_values[target_index, 0],
                feature_values[target_index, 1],
                feature_values[target_index, 2],
                feature_values[target_index, 3],
            ]
        )

if not sequences:
    raise ValueError(
        "No sequences were generated. Check the dataset and filters."
    )

X = np.asarray(sequences, dtype=np.float32)
y = np.asarray(labels, dtype=np.float32)
times = np.asarray(target_times, dtype=np.float64)
baseline_X = np.asarray(baseline_predictions, dtype=np.float32)

print("Sequences:", len(X))
print("Sequence shape:", X.shape)


# --------------------------------------------------
# Chronological train / validation / test split
# --------------------------------------------------

# Split by target-event time, not random row order.
order = np.argsort(times, kind="stable")
X = X[order]
y = y[order]
times = times[order]
baseline_X = baseline_X[order]

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = (
    X[:train_end],
    X[train_end:val_end],
    X[val_end:],
)

y_train, y_val, y_test = (
    y[:train_end],
    y[train_end:val_end],
    y[val_end:],
)

baseline_val = baseline_X[train_end:val_end]
baseline_test = baseline_X[val_end:]

print("Train sequences:", len(X_train))
print("Validation sequences:", len(X_val))
print("Test sequences:", len(X_test))

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("A chronological split is empty.")


# --------------------------------------------------
# Normalize using training data only
# --------------------------------------------------

# Fit statistics using training sequence timesteps only.
mean = X_train.reshape(-1, len(FEATURES)).mean(axis=0)
std = X_train.reshape(-1, len(FEATURES)).std(axis=0)
std[std < 1e-6] = 1.0

def normalize(values):
    return ((values - mean) / std).astype(np.float32)

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train.reshape(-1, 1)),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# --------------------------------------------------
# GRU deep neural network
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

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.regressor(last_hidden)


model = RecallGRU(input_size=len(FEATURES)).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


def predict(model, values):
    model.eval()
    results = []

    with torch.no_grad():
        for start in range(0, len(values), BATCH_SIZE):
            batch = torch.from_numpy(
                values[start : start + BATCH_SIZE]
            ).to(DEVICE)

            output = model(batch).squeeze(1)
            results.extend(output.cpu().numpy().tolist())

    return np.asarray(results, dtype=np.float64)


# --------------------------------------------------
# Training with validation-based early stopping
# --------------------------------------------------

best_val_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0

print("\nTraining GRU on", DEVICE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        batch_losses.append(loss.item())

    val_predictions = predict(model, X_val)
    val_loss = mean_squared_error(y_val, val_predictions)

    train_loss = float(np.mean(batch_losses))

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_mse={train_loss:.6f} | "
        f"val_mse={val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)


# --------------------------------------------------
# Final holdout evaluation
# --------------------------------------------------

test_predictions = predict(model, X_test)

gru_mae = mean_absolute_error(y_test, test_predictions)
gru_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("\nGRU test results")
print(f"MAE:  {gru_mae:.6f}")
print(f"RMSE: {gru_rmse:.6f}")

# Evaluate the existing RF-1 on exactly the same target rows.
import joblib

rf_path = MODEL_DIR / "recall_rf1.joblib"

if rf_path.exists():
    bundle = joblib.load(rf_path)
    rf_model = bundle["model"]

    rf_predictions = np.clip(
        rf_model.predict(baseline_test),
        0.0,
        1.0,
    )

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

    print("\nRF-1 baseline test results")
    print(f"MAE:  {rf_mae:.6f}")
    print(f"RMSE: {rf_rmse:.6f}")
else:
    rf_mae = None
    rf_rmse = None
    print("RF-1 model file not found; baseline comparison skipped.")

mean_baseline = np.full(len(y_test), y_train.mean())

mean_mae = mean_absolute_error(y_test, mean_baseline)
mean_rmse = np.sqrt(mean_squared_error(y_test, mean_baseline))

print("\nTraining-mean baseline test results")
print(f"MAE:  {mean_mae:.6f}")
print(f"RMSE: {mean_rmse:.6f}")


# --------------------------------------------------
# Save model and reproducibility metadata
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": len(FEATURES),
        "hidden_size": 64,
        "num_layers": 2,
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    },
    MODEL_PATH,
)

metadata = {
    "model_name": "Two-layer GRU recall regressor",
    "framework": "PyTorch",
    "dataset": DATA_PATH.name,
    "rows_loaded": NROWS,
    "valid_rows": int(len(df)),
    "sequences": int(n),
    "sequence_length": SEQUENCE_LENGTH,
    "features": FEATURES,
    "target": "session_correct / session_seen",
    "target_type": "session-level recall rate",
    "split": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
        "ordering": "chronological by target timestamp",
    },
    "best_epoch": best_epoch,
    "best_validation_mse": float(best_val_loss),
    "gru_test_mae": float(gru_mae),
    "gru_test_rmse": float(gru_rmse),
    "rf1_test_mae": None if rf_mae is None else float(rf_mae),
    "rf1_test_rmse": None if rf_rmse is None else float(rf_rmse),
    "mean_baseline_test_mae": float(mean_mae),
    "mean_baseline_test_rmse": float(mean_rmse),
    "seed": SEED,
    "calibrated": False,
    "limitations": [
        "Development experiment using a dataset prefix.",
        "Timestamp ties preserve source order; true event order may be unknown.",
        "Chronological splitting does not guarantee user-independent evaluation.",
        "Predictions estimate session-level recall rate, not a validated individual future-recall probability.",
    ],
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved GRU model:", MODEL_PATH)
print("Saved metadata:", METADATA_PATH)
print("Done.")
