# Experiment protocol

## Existing next-session score comparison

The checked-in `ml/reports/next_session_evaluation.json` compares mean, HLR, linear regression, Random Forest and GRU on the same sampled Duolingo next-session target. It uses 70/15/15 chronological and learner-disjoint splits, seeds 42/52/62, validation-only model selection, test MAE/RMSE, 10-bin expected calibration error and learner-cluster bootstrap 95% intervals. No test metric is used to select the champion.

The existing champion objective is mean validation MAE across both splits. It is not a date-crossing objective. The GRU shows better ECE and test RMSE in the report, while HLR shows lower MAE. Future model selection for dashboard reminders should additionally evaluate threshold-crossing / recommended-review-date error on valid time-to-review labels and retain a paired learner-level comparison.

## Knowledge-tracing experiments

Use chronological target-time splits and learner-disjoint splits. The forward-time splitter snaps 70/85% event-count cutoffs to whole timestamp groups so simultaneous records cannot straddle partitions; reports record the cutoff timestamps. At each target, inputs must stop before the target response. Any user/item encoders, scalers, or vocabulary cutoffs must be fit on the training partition only; unknown validation/test IDs use a documented OOV path. Report ROC-AUC, log loss, Brier score, accuracy/F1 at a predeclared threshold, calibration, and per-user bootstrap intervals when labels are binary. For aggregated fractional outcomes, report MAE/RMSE and justify weighting.

The trainer implements MLP, LSTM and Transformer, with AdamW, Adam, RMSProp and momentum-SGD choices. The recorded MLP/optimizer comparison uses seed 42, a 1-in-100 forward-time split and a three-epoch maximum; the MLP/LSTM learner-held-out comparison uses the smaller 1-in-500 sample. These are engineering experiments, not the final repeated-run protocol. Before selecting a model, repeat multiple seeds, add learner-cluster intervals, compare the same models/configuration on both protocols and assess calibration differences with paired learner bootstrap samples.

## RAG experiments

Build a human-judged query-to-passage set before claims. Keep standard BM25, hybrid BM25+TF-IDF, and state-aware reranking as separate arms. Report Recall@k, MRR, nDCG@k, source coverage and paired query-level intervals. Keep student-state features available only from events before query time. No judged set or RAG metrics currently exist; report them as not evaluated.

## Generation and policy experiments

Evaluate answer correctness, evidence support, citation validity, unsupported claims and abstention on reviewed examples. Compare fixed schedule, current HLR score + same policy, and any future model under identical splits. Offline simulation is not evidence of causal learning benefit. The learner pilot protocol in `docs/learner-pilot-protocol.md` describes the separate feasibility study needed for human outcomes.

## Reproducibility

Record code revision, config, data source/version, sample rule, seeds, feature schema, split IDs or deterministic split rule, dependency versions, artifact hashes and all metrics. Do not check in personal data or large raw datasets. Mark experiments that were not run as **Not evaluated**.
