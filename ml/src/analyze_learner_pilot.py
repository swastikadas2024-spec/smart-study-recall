"""Exploratory, participant-level analysis for the approved study pilot."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


REQUIRED = [
    "participant_code", "arm", "baseline_correct", "baseline_total",
    "delayed_correct", "delayed_total", "review_sessions_assigned",
    "review_sessions_completed",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Anonymized, one-row-per-participant CSV")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap resamples")

    data = pd.read_csv(args.csv_path)
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        parser.error(f"Missing required columns: {', '.join(missing)}")
    if data.participant_code.isna().any() or data.participant_code.duplicated().any():
        parser.error("Each non-empty participant_code must occur exactly once")
    data["arm"] = data.arm.astype(str).str.strip().str.lower()
    if not set(data.arm).issubset({"adaptive", "fixed"}):
        parser.error("arm values must be 'adaptive' or 'fixed'")
    for column in REQUIRED[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[REQUIRED[2:]].isna().any().any():
        parser.error("Outcome and completion fields must be numeric and complete")
    if (data["baseline_total"] <= 0).any() or (data["delayed_total"] <= 0).any():
        parser.error("baseline_total and delayed_total must be positive")
    if (data["baseline_correct"] < 0).any() or (data["baseline_correct"] > data["baseline_total"]).any():
        parser.error("baseline_correct must be between 0 and baseline_total")
    if (data["delayed_correct"] < 0).any() or (data["delayed_correct"] > data["delayed_total"]).any():
        parser.error("delayed_correct must be between 0 and delayed_total")
    if (data["review_sessions_completed"] < 0).any() or (data["review_sessions_completed"] > data["review_sessions_assigned"]).any():
        parser.error("Completed sessions must be between 0 and assigned sessions")

    data["baseline_score"] = data.baseline_correct / data.baseline_total
    data["delayed_score"] = data.delayed_correct / data.delayed_total
    data["completion_rate"] = np.where(
        data.review_sessions_assigned > 0,
        data.review_sessions_completed / data.review_sessions_assigned,
        np.nan,
    )
    adaptive = data.loc[data.arm == "adaptive", "delayed_score"].to_numpy(dtype=float)
    fixed = data.loc[data.arm == "fixed", "delayed_score"].to_numpy(dtype=float)
    if len(adaptive) < 2 or len(fixed) < 2:
        parser.error("Need at least two participants in each arm for the exploratory comparison")

    observed_difference = float(adaptive.mean() - fixed.mean())
    rng = np.random.default_rng(42)
    bootstrap = np.empty(args.bootstrap, dtype=float)
    for i in range(args.bootstrap):
        a = rng.choice(adaptive, size=len(adaptive), replace=True)
        f = rng.choice(fixed, size=len(fixed), replace=True)
        bootstrap[i] = a.mean() - f.mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    welch = ttest_ind(adaptive, fixed, equal_var=False)

    print("Exploratory pilot summary (learner is the analysis unit)")
    print(f"Adaptive n={len(adaptive)}; mean delayed score={adaptive.mean():.3f}; SD={adaptive.std(ddof=1):.3f}")
    print(f"Fixed n={len(fixed)}; mean delayed score={fixed.mean():.3f}; SD={fixed.std(ddof=1):.3f}")
    print(f"Adaptive minus fixed: {observed_difference:+.3f} ({observed_difference * 100:+.1f} percentage points)")
    print(f"Participant bootstrap 95% CI: [{low:+.3f}, {high:+.3f}]")
    print(f"Exploratory Welch t-test p-value: {welch.pvalue:.4g}")
    print(f"Mean review completion: {data.groupby('arm').completion_rate.mean().to_dict()}")
    print("Interpret as feasibility/pilot evidence only; review the approved protocol and analysis plan.")


if __name__ == "__main__":
    main()
