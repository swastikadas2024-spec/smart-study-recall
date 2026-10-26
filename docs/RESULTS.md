# Research results

## Knowledge-tracing model runs

These were real CPU runs on the local dataset using seed 42 and a three-epoch cap (early stopping selected the best validation loss). The main matched comparison uses the deterministic 1-in-100 sample, 119,140 event rows, 1,118 learners, and a chronological 70/15/15 split. Labels are session-level correct/seen fractions, weighted by question count. Results are preliminary and do not describe app course-topic recall.

| Model | Test weighted MAE | Test weighted RMSE | Test binomial log loss | Test weighted ECE | Test weighted AUC |
|---|---:|---:|---:|---:|---:|
| Training weighted mean | 0.1526 | 0.2266 | 0.3241 | 0.000005 | 0.5000 |
| Random Forest (same current-event numeric features) | 0.1473 | 0.2225 | 0.3172 | 0.0110 | 0.5954 |
| LSTM | 0.1485 | 0.2254 | 0.3211 | 0.0160 | 0.5861 |
| Transformer | 0.1479 | 0.2244 | 0.3189 | 0.0047 | 0.5913 |

On this one split/seed, neither deep model beats the Random Forest on MAE, RMSE, log loss or AUC. The Transformer has lower ECE. This is not evidence of reliable superiority; there are no repeated-seed or learner-bootstrap intervals for this new KT comparison, and the dataset target is an aggregated session outcome.

An additional LSTM learner-held-out run used the smaller 1-in-500 sample (22,670 events, 211 learners) because it was run as a split smoke experiment. Its test MAE was 0.1442 versus 0.1254 for the same-split Random Forest. Do not compare these numbers directly to the 1-in-100 chronological run; they are different samples and split protocols.

## Existing measured experiment: next-session score

Source: `ml/reports/next_session_evaluation.json`. Task: predict the next observed Duolingo session's correct/seen fraction for a candidate gap up to 30 days. Data: reported 617 deterministic 1-in-100-sampled learners and 62,931 examples. Seeds: 42, 52, 62.

| Model | Mean validation MAE across splits | Forward-time test MAE | Forward-time test RMSE | Forward-time test ECE | Learner-held-out test MAE | Learner-held-out test RMSE | Learner-held-out test ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean | 0.1791 | 0.1763 | 0.2798 | 0.0074 | see JSON report | see JSON report | see JSON report |
| HLR | 0.1304 | 0.1404 | 0.2926 | 0.0782 | 0.1001 | 0.2591 | 0.0695 |
| Linear regression | 0.1725 | see JSON report | see JSON report | see JSON report | see JSON report | see JSON report | see JSON report |
| Random Forest | 0.1722 | see JSON report | see JSON report | see JSON report | see JSON report | see JSON report | see JSON report |
| GRU | 0.1719 | 0.1711 | 0.2758 | 0.0091 | 0.1530 | 0.2474 | 0.0105 |

The artifact selection rule chooses HLR because it minimizes mean validation MAE. GRU has lower test RMSE and ECE in the two displayed protocols; HLR has lower MAE. The 95% learner-cluster intervals and full per-seed metrics are in the JSON report. These are next-session score results, not delayed-retention, forgetting-date, course-topic, RAG, or human-learning results.

## Not evaluated

- Retrieval relevance, RAG answer grounding, citation accuracy, or ablation metrics: **Not evaluated**.
- Adaptive-vs-fixed learning outcomes: **Not evaluated**; requires a consented learner study.
- App-domain delayed retention or topic-specific forgetting date accuracy: **Not evaluated**.
