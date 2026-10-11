
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "settles.acl16.learning_traces.13m.csv.gz"
)

NROWS = 200_000

print(f"Loading {NROWS:,} rows...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=NROWS)

# Preserve source order to make tied timestamps visible.
df["source_order"] = range(len(df))

df = df.sort_values(
    ["user_id", "lexeme_id", "timestamp", "source_order"]
).reset_index(drop=True)

# Current observation features.
df["historical_accuracy"] = (
    df["history_correct"] / df["history_seen"].replace(0, pd.NA)
)
df["elapsed_days"] = df["delta"] / 86400

# Next observation outcome for the same learner-concept pair.
grouped = df.groupby(["user_id", "lexeme_id"], sort=False)

df["next_timestamp"] = grouped["timestamp"].shift(-1)
df["next_session_seen"] = grouped["session_seen"].shift(-1)
df["next_session_correct"] = grouped["session_correct"].shift(-1)

df["next_recall"] = (
    df["next_session_correct"] / df["next_session_seen"]
)

# Only keep rows with a later observation and a valid next-session target.
examples = df[
    df["next_timestamp"].notna()
    & (df["next_timestamp"] > df["timestamp"])
    & (df["history_seen"] > 0)
    & (df["session_seen"] > 0)
    & (df["next_session_seen"] > 0)
    & df["historical_accuracy"].notna()
    & df["next_recall"].notna()
].copy()

print("\nNEXT-REVIEW SAMPLE SUMMARY")
print("=" * 55)
print("Original rows:", len(df))
print("Valid next-review examples:", len(examples))
print("Unique learners:", examples["user_id"].nunique())
print(
    "Unique learner-concept pairs:",
    examples[["user_id", "lexeme_id"]].drop_duplicates().shape[0],
)
print("Mean next recall:", round(examples["next_recall"].mean(), 4))

print("\nNext-review target distribution:")
print(examples["next_recall"].describe().round(4))

print("\nEXAMPLE SEQUENCES")
print("=" * 55)

columns = [
    "user_id",
    "lexeme_id",
    "timestamp",
    "elapsed_days",
    "history_seen",
    "history_correct",
    "next_timestamp",
    "next_session_seen",
    "next_session_correct",
    "next_recall",
]

print(examples[columns].head(10).to_string(index=False))

print("\nTarget-construction experiment complete.")
print("No model or dataset files were modified.")
