
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT / "ml" / "data" / "processed" / "next_review_candidates.csv"
)

CHUNK_SIZE = 500_000

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Candidate file not found: {DATA_PATH}")

# Each interval tracks row count, recall sum, zero outcomes, and perfect outcomes.
bins = [0, 1 / 24, 1, 3, 7, 14]
labels = [
    "Under 1 hour",
    "1 hour to 1 day",
    "1 to 3 days",
    "3 to 7 days",
    "7 to 14 days",
]

stats = {
    label: {"count": 0, "recall_sum": 0.0, "zero": 0, "perfect": 0}
    for label in labels
}

print("Analyzing next-review targets in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=["next_gap_days", "next_recall"],
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    chunk = chunk.dropna(subset=["next_gap_days", "next_recall"])
    chunk = chunk[
        (chunk["next_gap_days"] > 0)
        & (chunk["next_gap_days"] <= 14)
        & (chunk["next_recall"] >= 0)
        & (chunk["next_recall"] <= 1)
    ].copy()

    chunk["gap_group"] = pd.cut(
        chunk["next_gap_days"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    for label, group in chunk.groupby("gap_group", observed=False):
        if pd.isna(label) or group.empty:
            continue

        name = str(label)
        stats[name]["count"] += len(group)
        stats[name]["recall_sum"] += group["next_recall"].sum()
        stats[name]["zero"] += int((group["next_recall"] == 0).sum())
        stats[name]["perfect"] += int((group["next_recall"] == 1).sum())

    print(f"Processed chunk {chunk_number}")

print("\nNEXT-REVIEW TARGETS BY GAP")
print("=" * 65)
print(
    f"{'Interval':<20} {'Rows':>10} {'Mean recall':>13} "
    f"{'Zero %':>10} {'Perfect %':>12}"
)

for label in labels:
    s = stats[label]
    n = s["count"]

    if n == 0:
        print(f"{label:<20} {0:>10}")
        continue

    mean_recall = s["recall_sum"] / n
    zero_pct = 100 * s["zero"] / n
    perfect_pct = 100 * s["perfect"] / n

    print(
        f"{label:<20} {n:>10,} {mean_recall:>12.4f} "
        f"{zero_pct:>9.2f}% {perfect_pct:>11.2f}%"
    )

print("\nAnalysis complete. No models or candidate data were modified.")
