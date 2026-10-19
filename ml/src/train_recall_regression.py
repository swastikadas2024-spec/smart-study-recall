import pandas as pd

from recall_regression import (
    train_model,
    evaluate_model,
    save_model,
)


# -----------------------------
# 1. Load dataset
# -----------------------------

dataset_path = (
    "../data/raw/"
    "settles.acl16.learning_traces.13m.csv.gz"
)

df = pd.read_csv(
    dataset_path,
    compression="gzip",
    nrows=500000
)


# -----------------------------
# 2. Prepare data
# -----------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s"
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

df["delta_days"] = (
    df["delta"] / (60 * 60 * 24)
)


# Remove invalid rows
df = df[
    (df["delta"] > 0) &
    (df["history_seen"] > 0) &
    (df["p_recall"] >= 0) &
    (df["p_recall"] <= 1)
].copy()


# -----------------------------
# 3. Temporal train/test split
# -----------------------------

split1 = df["timestamp"].quantile(0.70)
split2 = df["timestamp"].quantile(0.85)

train_df = df[
    df["timestamp"] <= split1
].copy()

test_df = df[
    df["timestamp"] > split2
].copy()


print("Training rows:", len(train_df))
print("Test rows:", len(test_df))


# -----------------------------
# 4. Train model
# -----------------------------

model = train_model(train_df)


# -----------------------------
# 5. Evaluate
# -----------------------------

mae, rmse = evaluate_model(
    model,
    test_df
)

print()
print("Random Forest Recall Model")
print("--------------------------")
print("MAE :", mae)
print("RMSE:", rmse)


# -----------------------------
# 6. Save model metadata
# -----------------------------

save_model(
    model,
    mae,
    rmse
)