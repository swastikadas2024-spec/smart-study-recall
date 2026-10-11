
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

print(f"Loading first {NROWS:,} rows...")
df = pd.read_csv(DATASET_PATH, compression="gzip", nrows=NROWS)

df = df.sort_values(
    ["user_id", "lexeme_id", "timestamp"]
).reset_index(drop=True)

print("\n1. DATASET SAMPLE")
print("=" * 55)
print("Rows:", len(df))
print("Unique learners:", df["user_id"].nunique())
print("Unique concepts:", df["lexeme_id"].nunique())
print(
    "Unique learner-concept pairs:",
    df[["user_id", "lexeme_id"]].drop_duplicates().shape[0],
)

pair_counts = df.groupby(["user_id", "lexeme_id"]).size()

print("\n2. OBSERVATIONS PER LEARNER-CONCEPT PAIR")
print("=" * 55)
print(pair_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2))
print("\nPair observation counts:")
print(
    pd.cut(
        pair_counts,
        bins=[0, 1, 2, 5, 10, float("inf")],
        labels=["1", "2", "3-5", "6-10", "11+"],
    ).value_counts().sort_index()
)

print("\n3. REPEATED EVENTS")
print("=" * 55)
print(
    "Pairs with at least 2 observations:",
    int((pair_counts >= 2).sum()),
)
print(
    "Pairs with at least 3 observations:",
    int((pair_counts >= 3).sum()),
)

print("\n4. EXAMPLE LEARNER-CONCEPT SEQUENCES")
print("=" * 55)

repeated_pairs = pair_counts[pair_counts >= 3]

if repeated_pairs.empty:
    print("No learner-concept pairs with 3 or more observations in this sample.")
else:
    user_id, lexeme_id = repeated_pairs.index[0]

    example = df[
        (df["user_id"] == user_id)
        & (df["lexeme_id"] == lexeme_id)
    ][
        [
            "user_id",
            "lexeme_id",
            "timestamp",
            "delta",
            "history_seen",
            "history_correct",
            "session_seen",
            "session_correct",
            "p_recall",
        ]
    ]

    print(example.head(10).to_string(index=False))

print("\nInspection complete. No files were modified.")
