
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "ml" / "data" / "raw" / "settles.acl16.learning_traces.13m.csv.gz"
OUTPUT_DIR = ROOT / "ml" / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "next_review_candidates.csv"

CHUNK_SIZE = 500_000

COLUMNS = [
    "timestamp",
    "delta",
    "user_id",
    "lexeme_id",
    "history_seen",
    "history_correct",
    "session_seen",
    "session_correct",
]

def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally overwriting an earlier experiment.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. Rename it or choose a new "
            "output filename before running this script."
        )

    print("Reading the full dataset in chunks...")
    chunks = []

    for i, chunk in enumerate(
        pd.read_csv(
            DATASET,
            compression="gzip",
            usecols=COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        # Retain only rows with usable identifiers and counts.
        chunk = chunk.dropna(subset=["user_id", "lexeme_id"]).copy()

        chunk = chunk[
            (chunk["timestamp"] > 0)
            & (chunk["delta"] > 0)
            & (chunk["history_seen"] > 0)
            & (chunk["history_correct"] >= 0)
            & (chunk["history_correct"] <= chunk["history_seen"])
            & (chunk["session_seen"] > 0)
            & (chunk["session_correct"] >= 0)
            & (chunk["session_correct"] <= chunk["session_seen"])
        ].copy()

        if not chunk.empty:
            chunks.append(chunk)

        print(f"Chunks read: {i}; valid rows retained: "
              f"{sum(len(c) for c in chunks):,}")

    if not chunks:
        raise ValueError("No valid rows were found.")

    print("\nCombining valid rows...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    # Preserve original order for tied timestamps.
    df["source_order"] = np.arange(len(df), dtype=np.int64)

    # Stable ordering by learner, concept, timestamp, and source order.
    df = df.sort_values(
        ["user_id", "lexeme_id", "timestamp", "source_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = df.groupby(["user_id", "lexeme_id"], sort=False)

    df["next_timestamp"] = grouped["timestamp"].shift(-1)
    df["next_session_seen"] = grouped["session_seen"].shift(-1)
    df["next_session_correct"] = grouped["session_correct"].shift(-1)

    # Use only pairs with a strictly later observation.
    df["gap_seconds"] = df["next_timestamp"] - df["timestamp"]

    examples = df[
        df["gap_seconds"].notna()
        & (df["gap_seconds"] > 0)
    ].copy()

    examples["elapsed_days"] = examples["delta"] / 86400
    examples["next_gap_days"] = examples["gap_seconds"] / 86400
    examples["historical_accuracy"] = (
        examples["history_correct"] / examples["history_seen"]
    )
    examples["next_recall"] = (
        examples["next_session_correct"] / examples["next_session_seen"]
    )

    # Save only relevant columns for this experiment.
    output_columns = [
        "user_id",
        "lexeme_id",
        "timestamp",
        "elapsed_days",
        "history_seen",
        "history_correct",
        "historical_accuracy",
        "session_seen",
        "session_correct",
        "next_timestamp",
        "next_gap_days",
        "next_session_seen",
        "next_session_correct",
        "next_recall",
    ]

    examples[output_columns].to_csv(OUTPUT_FILE, index=False)

    print("\nFULL DATASET NEXT-REVIEW SUMMARY")
    print("=" * 55)
    print("Valid source rows:", len(df))
    print("Candidate next-review examples:", len(examples))
    print("Unique learners:", examples["user_id"].nunique())
    print(
        "Unique learner-concept pairs:",
        examples[["user_id", "lexeme_id"]].drop_duplicates().shape[0],
    )
    print("Mean next recall:", round(examples["next_recall"].mean(), 4))

    print("\nNext-review gap distribution (days):")
    print(
        examples["next_gap_days"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .round(4)
        .to_string()
    )

    print("\nGap intervals:")
    bins = [0, 1/24, 1, 3, 7, 14, float("inf")]
    labels = [
        "Under 1 hour",
        "1 hour to 1 day",
        "1 to 3 days",
        "3 to 7 days",
        "7 to 14 days",
        "14+ days",
    ]

    examples["gap_group"] = pd.cut(
        examples["next_gap_days"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    print(examples["gap_group"].value_counts().reindex(labels).to_string())

    print("\nSaved candidate data to:")
    print(OUTPUT_FILE)
    print("\nNo existing models or raw dataset files were modified.")

if __name__ == "__main__":
    main()
