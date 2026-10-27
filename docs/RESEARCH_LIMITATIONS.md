# Research limitations and unresolved questions

- The evaluated target is an observed next-session score in a language-learning trace. It is not a measured long-term memory probability or delayed recall outcome.
- The sampled public dataset and app's course-topic quiz events differ in item semantics, population and assessment format. Generalization to uploaded materials is unvalidated.
- The dashboard's 80% threshold and review scheduling rule are policy choices, not scientifically established universal thresholds.
- The existing GRU has lower reported ECE and RMSE than HLR but higher MAE; the current selection rule only optimizes validation MAE. Training-set metrics and paired difference intervals are absent.
- The new KT MLP, LSTM and Transformer runs are single-seed preliminary comparisons. MLP and LSTM have lower forward-time MAE than RF but lose on several other metrics; on the 1-in-500 learner-held-out sample, MLP and LSTM have worse MAE/AUC than RF. Only the AdamW MLP has lower ECE than RF in the forward-time architecture comparison. Optimizer comparisons are three-epoch single-seed runs. More seeds and same-sample evaluation across both split protocols are required.
- The KT sequence model is served by the ML API but is not proxied into app topic pages because app topic IDs and Duolingo lexeme IDs have no validated mapping.
- HLR/GRU scores are not calibrated against delayed course-topic assessments. Do not label them validated forgetting risks.
- Current RAG personalization uses repeated material-quiz concept errors and a deterministic overlap boost. There is no learned reranker, judged relevance set, citation audit set, or measured standard-vs-adaptive comparison.
- LLM output quality and latency depend on the configured external provider. Extractive fallback is available, but it is not equivalent to generated explanations.
- The app stores sensitive account-linked learning events and uploaded materials. Do not export or reuse them for research without an approved protocol, consent, minimization and access controls.
- Novelty and patentability have not been established. Conduct a relevant literature and patent search before making either claim.
