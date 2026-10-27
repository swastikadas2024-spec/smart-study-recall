# Recent commit review

Audit date: 2026-10-27. Repository branch: `main`. Working tree was clean before this follow-up work and `main` matched its local `origin/main` tracking ref. A network fetch could not update `.git/FETCH_HEAD` in the restricted workspace, so remote freshness was not independently rechecked in this audit.

## Commit inspected

- Commit: `05c5587f987c6e9cc4e40fe762c79e358f05828d`
- Message: `Add research prototype and adaptive study tools`
- Parent: `b4740a2` (`Add RAG study tools and next-session model`)
- Change size: 44 files, 4,300 insertions, 36 deletions.

## Files changed

- Configuration and overview: `.env.example`, `README.md`, `app/config.py`, `app/main.py`.
- Backend: `app/routers/rag_chatbot.py`, `app/services/next_session.py`, `app/services/rag_engine.py`, `app/services/study_materials.py`, `app/tests/test_rag_state.py`.
- Documentation: `docs/API.md`, `docs/DATASET_AND_PREPROCESSING.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/IMPLEMENTATION_AUDIT.md`, `docs/MODEL_ARCHITECTURE.md`, `docs/RESEARCH_LIMITATIONS.md`, `docs/RESEARCH_PROPOSAL.md`, `docs/RESULTS.md`, `docs/SETUP.md`.
- Frontend: `frontend/chatbot.html`, `frontend/css/app.css`, `frontend/css/rag-chatbot.css`, `frontend/dashboard.html`, `frontend/js/dashboard.js`, `frontend/js/rag-chatbot.js`, `frontend/js/study.js`, `frontend/materials.html`, `frontend/planner.html`, `frontend/profile.html`, `frontend/progress.html`, `frontend/study.html`, `frontend/topics.html`.
- Model artifacts and reports: `ml/models/knowledge_tracer_lstm_forward_time.pt`, `ml/models/knowledge_tracer_lstm_learner_held_out.pt`, `ml/models/knowledge_tracer_transformer_forward_time.pt`, `ml/reports/knowledge_tracing_evaluation.json`, `ml/reports/knowledge_tracing_lstm_forward_time_evaluation.json`, `ml/reports/knowledge_tracing_lstm_learner_held_out_evaluation.json`, and `ml/reports/knowledge_tracing_transformer_forward_time_evaluation.json`.
- ML code and tests: `ml/src/api.py`, `ml/src/knowledge_tracing.py`, `ml/src/knowledge_tracing_service.py`, `ml/src/train_knowledge_tracer.py`, `ml/tests/test_knowledge_tracing.py`, `ml/tests/test_knowledge_tracing_api.py`.

## What the commit introduced

1. A local ML API for next-interaction prediction using trained LSTM and causal Transformer checkpoints. The dataset target is an observed Duolingo session correct/seen fraction, not a delayed retention label.
2. A separate RAG chatbot page with standard BM25, BM25+TF-IDF hybrid retrieval, and heuristic recall-aware reranking using material quiz gaps and recency.
3. An explainable intervention response and authenticated material-scoped routes, with citations constrained to retrieved chunk IDs.
4. Dashboard edits retaining the existing next-session predictor and displaying an 80% score-crossing reminder estimate.
5. Research docs, CPU-trained model artifacts, evaluation JSON, and unit tests.

## Technical implications and review findings

- The RAG chatbot is connected to existing material chunks and answer generation; it is not a disconnected visual demo. Its recall-aware mode is a hand-weighted heuristic, not a learned ranker. There is no human-judged retrieval set, persistent research log, citation audit, or RAG quality score yet.
- The LSTM/Transformer code uses train-only vocabulary/scaling, causal interaction windows, same-timestamp group boundaries, early stopping, and CPU-compatible inference. The serving endpoint is in the separate local ML service. It is not called by course-topic dashboard forecasts because Duolingo item IDs have no validated mapping to app topics.
- The dashboard’s review-by date remains an estimate from the separate next-session score model and the selected 0.80 policy threshold. It is not a measured date of forgetting.
- The commit includes three small PyTorch checkpoints (about 3.9 MB total) and JSON reports; the source dataset remains local and gitignored.
- The unqualified `ml/reports/knowledge_tracing_evaluation.json` is a legacy 1-in-500 forward-time LSTM run, not part of the current 1-in-100 architecture comparison; keep its different sample/protocol explicit when interpreting it.
- Existing project features and authentication routes remain present. No schema migration was added in this commit.
- The prior report described only the new LSTM and Transformer. It omitted an MLP neural baseline and an optimizer comparison; this follow-up adds both.
- The previous `TestClient` health smoke request stalled in this Windows environment. Direct model inference and the model/API validation unit tests completed; a full live app/database smoke test remains unverified.
- No CI, linter, formatter, frontend build, type-check configuration, or pytest setup was found. Pytest is not installed; hermetic Python `unittest` suites are present.

## Detected defect and missing functionality

- The committed forward-time knowledge-tracing split cut at row indices rather than whole timestamp groups. On the sampled 1-in-100 data, 41 rows shared the train/validation cutoff timestamp and 7 shared the validation/test cutoff timestamp, so simultaneous events crossed split boundaries and could expose same-time item identities through the train-only item vocabulary. This follow-up changes the cutoff to complete timestamp groups, records the boundary timestamps in reports, and tests that groups do not cross partitions.
- The commit had no MLP neural baseline or optimizer comparison; this follow-up adds both and records matched-split results.
- The RAG UI, authenticated routes, retrieval, answer generation/fallback and citation filtering are implemented. A full end-to-end chat with the private database and configured provider was not exercised in this environment; retrieval quality, citations against human labels, and generation support remain unmeasured.
- The app dashboard's per-topic display uses the existing API-served candidate-gap predictor and review threshold; no research evaluation dashboard of model metrics was found. The checked-in reports contain measured offline experiments, not illustrative dashboard values.
- No app-topic delayed-retention model, consented learner evaluation, retrieval relevance benchmark, misconception annotation set or adaptive-policy outcome evaluation exists yet.

## Tests and results at commit review

Before follow-up edits, these commands passed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ml\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s app\tests -v
```

They ran 6 ML tests and 5 app/RAG tests. `git diff --check` had also passed before the commit. The tracked reports contain real single-seed preliminary LSTM and Transformer runs; neither beat the matched Random Forest on most forward-time test metrics. The 1-in-500 learner-held-out LSTM run is a different sample and is not directly comparable to the 1-in-100 forward-time table.

## Recommended next steps

- Follow-up work added an MLP baseline, Adam/AdamW/RMSProp/SGD controls, same-split optimizer results, and a held-out MLP result. Repeat all model comparisons across seeds and evaluate candidates on the same sample under both split protocols.
- Build reviewed course-specific retrieval and answer-evidence sets before claiming RAG quality.
- Collect consented app-domain delayed quiz outcomes before connecting public-trace models to course topics or evaluating learning gains.
- Keep the prototype claims narrow; integration alone does not establish research novelty or patentability.
