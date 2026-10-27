# Research results

## Knowledge-tracing model runs

These were real CPU runs on the local dataset using seed 42. The MLP command pasted by the user used its default eight-epoch cap and early-stopped after four epochs; LSTM and Transformer used a three-epoch cap. The main comparison uses the deterministic 1-in-100 sample, 119,140 event rows, 1,118 learners, and a chronological 70/15/15 split with complete timestamp groups. Labels are session-level correct/seen fractions, weighted by question count. Results are preliminary and do not describe app course-topic recall.

| Model | Test weighted MAE | Test weighted RMSE | Test binomial log loss | Test weighted ECE | Test weighted AUC |
|---|---:|---:|---:|---:|---:|
| Training weighted mean | 0.1525 | 0.2266 | 0.3241 | 0.000032 | 0.5000 |
| Random Forest (same current-event numeric features) | 0.1472 | 0.2225 | 0.3170 | 0.0104 | 0.5959 |
| MLP (AdamW) | 0.1444 | 0.2247 | 0.3200 | 0.0082 | 0.5869 |
| LSTM | 0.1468 | 0.2246 | 0.3195 | 0.0108 | 0.5927 |
| Transformer | 0.1573 | 0.2249 | 0.3203 | 0.0154 | 0.5851 |

On this one split/seed, MLP and LSTM have lower MAE than Random Forest, while Random Forest has lower RMSE, log loss and higher AUC than all three neural candidates. Only the AdamW MLP has lower ECE than Random Forest in the architecture table. These are not evidence of reliable superiority: there are no repeated-seed or learner-bootstrap intervals for this KT comparison, and the dataset target is an aggregated session outcome.

The data split and target are shared, but the Random Forest baseline currently receives only the current event's four numeric features; the neural candidates also receive item embeddings and up to 29 preceding event representations. This is a baseline comparison, not a feature-controlled architecture ablation. Future ablations should give the non-sequential models equivalent, train-only sequence summary features.

For the MLP AdamW run, train weighted loss fell from 0.3425 to 0.3152 over four epochs. Validation loss improved from 0.3015 to 0.2999 by epoch 2, then remained slightly higher at 0.3003 and 0.2999 in epochs 3 and 4, so the saved checkpoint is epoch 2. That small divergence is consistent with early overfitting pressure, but four epochs and a single split are insufficient to characterize generalization. The per-epoch values are persisted in `knowledge_tracing_mlp_forward_time_evaluation.json`.

### MLP optimizer comparison

All runs below used the same 1-in-100 sample, forward-time split, seed 42, MLP architecture and weighted binomial loss. The AdamW run now reflects the user's later default-cap run (eight requested, stopped after four; best checkpoint epoch 2); Adam, RMSProp and SGD used a three-epoch maximum. The AdamW test metrics are unchanged from the earlier three-epoch run because both selected epoch 2, but the training budgets are not fully matched.

| Optimizer | Best validation binomial loss | Test weighted MAE | Test weighted RMSE | Test log loss | Test ECE | Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.2999 | 0.1444 | 0.2247 | 0.3200 | 0.0082 | 0.5869 |
| Adam | 0.2991 | 0.1394 | 0.2250 | 0.3210 | 0.0168 | 0.5914 |
| RMSProp | 0.2984 | 0.1473 | 0.2244 | 0.3189 | 0.0078 | 0.5923 |
| SGD + momentum | 0.3043 | 0.1496 | 0.2252 | 0.3208 | 0.0029 | 0.5738 |

Adam gives the lowest test MAE among the MLP optimizer runs; RMSProp has the best log loss/RMSE, and SGD has the lowest ECE. No optimizer wins on every metric. The MLP Adam and AdamW variants have lower MAE than the matched Random Forest but higher RMSE, log loss and lower AUC. Treat this as a short convergence experiment: one seed and at most four actual epochs do not support selecting a generally superior optimizer. Per-run JSON records include full metrics, split counts and artifact hashes; `ml/reports/mlp_optimizer_comparison.json` collects the comparison in one file.

The learner-held-out comparison uses the smaller 1-in-500 sample (22,670 events, 211 learners), seed 42 and the same 70/15/15 learner split for all rows below. These results are comparable within this table, but not to the 1-in-100 chronological results above.

| Model | Test weighted MAE | Test weighted RMSE | Test log loss | Test ECE | Test AUC |
|---|---:|---:|---:|---:|---:|
| Mean | 0.1330 | 0.2091 | 0.2666 | 0.0120 | 0.5000 |
| Random Forest | 0.1254 | 0.2061 | 0.2583 | 0.0085 | 0.6266 |
| MLP (AdamW) | 0.1338 | 0.2084 | 0.2648 | 0.0156 | 0.5795 |
| LSTM | 0.1442 | 0.2110 | 0.2707 | 0.0296 | 0.4883 |

On this small held-out sample, the Random Forest has the lowest MAE, RMSE and log loss. This is one split and one seed; uncertainty intervals are not available for these KT runs.

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
