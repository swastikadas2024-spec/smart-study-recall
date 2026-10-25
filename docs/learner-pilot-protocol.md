# Learner pilot: adaptive versus fixed review intervals

## Purpose and claims

This pilot evaluates whether the app's **review policy** is usable and whether its outcomes justify a larger study. The model predicts the next-session correct-answer fraction conditional on a proposed interval. The policy then selects an interval. The Duolingo trace experiment does not establish that changing the interval causes better learning, and this pilot must measure that separately.

Do not describe a small classroom pilot as proof of efficacy or as validation of a patent claim. Treat it as feasibility evidence unless an a-priori power analysis supports stronger conclusions.

## Design

- Two-arm, learner-level randomized pilot: adaptive review policy versus a fixed spaced-review schedule.
- Use the same course material, quiz format, review opportunities, and study-time guidance in both arms.
- Collect a baseline quiz before assignment, session-level scores at each review, and a delayed assessment after the final scheduled review. Keep the delayed assessment content held out from model training and practice.
- Stratify random assignment by baseline score band where practical. Keep one participant in one arm to reduce schedule contamination.
- Pilot with approximately 30–60 consenting learners only to estimate recruitment, completion, usability, and outcome variance. Determine the sample size for a confirmatory study with a power analysis before making efficacy claims.

## Outcomes

- **Primary feasibility outcome:** proportion of assigned review sessions completed.
- **Exploratory learning outcome:** per-learner delayed-assessment correct fraction, compared between arms; also report baseline-adjusted change as a secondary analysis.
- **Operational outcomes:** enrollment, withdrawal, missing assessments, time spent, schedule adherence, and user-reported usability.
- Report arm sizes, missingness, effect sizes, and confidence intervals. Do not use only pooled question counts, since learners with more answered questions would dominate the result.

## Consent and privacy checklist

Obtain the institution's faculty/supervisor and ethics or human-subjects review before recruitment or analysis for research dissemination. Participation must be voluntary, separate from grades, and withdrawable without penalty. Explain the study purpose, activities, risks, data collected, retention period, who can access data, and how withdrawal/deletion works. Use random participant codes in analysis files; keep the identity-to-code key separately with restricted access. Do not export names, emails, auth IDs, uploaded course documents, or free-text chat content to the analysis CSV.

The app currently stores account-linked learning activity and does not implement randomized pilot enrollment or research consent capture. This document is a protocol starter, not an in-app consent workflow or ethics approval. Do not treat existing app users as research participants without the required approval and consent.

## Analysis file

Create one anonymized row per participant using the columns below, then run:

```powershell
python ml/src/analyze_learner_pilot.py path\to\anonymized_pilot.csv
```

Required columns:

```text
participant_code,arm,baseline_correct,baseline_total,delayed_correct,delayed_total,review_sessions_assigned,review_sessions_completed
```

Use only `adaptive` or `fixed` for `arm`. The script reports participant-level delayed-score means, the adaptive-minus-fixed difference with a participant bootstrap 95% confidence interval, and a Welch t-test as an exploratory summary. It does not replace a preregistered analysis plan or a statistician's review.
