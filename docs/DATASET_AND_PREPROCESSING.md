# Dataset and preprocessing

## Duolingo Learning Traces

The local ignored dataset is expected at `ml/data/raw/settles.acl16.learning_traces.13m.csv.gz`. The checked-in analysis uses these fields: `user_id`, `lexeme_id`, `timestamp`, `learning_language`, `ui_language`, `history_seen`, `history_correct`, `session_seen`, `session_correct`, `p_recall`, and related timing metadata. The repository does not vendor the source dataset; obtain it from its official source and comply with its license/access terms.

The current next-session training script streams the file, applies a deterministic 1-in-N learner hash sample (default N=100), filters invalid counts and timestamps, removes ambiguous tied learner-item timestamps, sorts by learner/item/time, and pairs an event with the next event for the same learner and item. It limits candidate gaps to 30 days. The target is next-session correct/seen fraction.

The separate KT experiment streams the same columns and applies the same deterministic learner hash key. It removes duplicate learner-item-timestamp observations, then orders learner events by timestamp and lexeme. Same-learner, same-timestamp records form a session boundary: the model may consume all of that group only for later timestamps, and no target sees another answer from its own simultaneous group. Its target is the current event's `session_correct / session_seen`; inputs include the target item ID, pre-event per-item history, gap since the prior timestamp group, and accuracy in the previous group. Item vocabulary and feature scaling are fit on train data; unknown items map to an OOV index. The weighted binomial loss treats `session_seen` as the number of observations summarized by the fraction.

The report records 617 sampled learners and 62,931 eligible pairs. These are the reported sample statistics, not the full dataset size. Splits are either chronological by target-event time (70/15/15) or learner-disjoint (70/15/15). Preprocessing parameters and identities must be fit on training data only where applicable.

## Limitations

- The target is an observed next-session score, not an independent delayed-memory assessment.
- A learner-held-out split tests unseen learners who have their own history at inference; it does not test a no-history cold start.
- The dataset's lexeme IDs and learning task do not map to app course-topic IDs or uploaded materials.
- KT event rows are session aggregates, not individual question-level labels; same-time rows are grouped as one information boundary.
- The deterministic sample may be cohort-biased; repeat using other hash buckets before broad claims.
- App focus duration is not included in this model target.
- No consented app-domain training export is currently documented.
