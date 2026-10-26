# Repository implementation audit

Audit snapshot: 2026-10-26, before the research-prototype changes in this task. Existing uncommitted work was present and is being preserved.

## Architecture found

- Frontend: static HTML, CSS and browser JavaScript under `frontend/`; no bundler or package manifest was found.
- Application API: FastAPI under `app/`, SQLAlchemy models and Supabase Auth token verification. App data uses PostgreSQL configured through the repository-root `.env`.
- ML API: a separate FastAPI service under `ml/src/api.py`.
- Dataset: a local ignored copy of Duolingo Learning Traces at `ml/data/raw/settles.acl16.learning_traces.13m.csv.gz`.
- Migrations: SQL files under `migrations/`. Existing schema contains profiles, topics, learning events, revision schedules, study materials/chunks, material quiz sessions and quiz responses.
- Deployment: local development instructions are in README; no Docker, CI workflow, frontend package/build configuration, or test configuration was found in the repository inventory.

## Existing functionality

- Authenticated student, topic, learning-event, schedule, dashboard, recommendation and topic-recall routes are mounted in `app/main.py`.
- A legacy predictions router is intentionally not mounted because its student-history endpoints lack ownership checks.
- Study quiz scores create `LearningEvent` rows. Focus timer and manual focus-time entries use browser `localStorage`; duration alone does not create recall labels.
- Material upload extracts PDF, DOCX, TXT and Markdown text, stores overlapping chunks, and supports notes, quizzes and chat. The RAG chatbot provides BM25-only and BM25 + TF-IDF cosine retrieval with reciprocal-rank fusion, then uses a configured OpenAI-compatible LLM or extractive fallback.
- Material quiz responses store concept labels and correctness. `concept_gaps` derives a simple per-material weakness score from repeated responses; the existing hybrid RAG uses that signal to boost matching passages. The `standard` path also currently includes a gap boost, so it is not a clean non-personalized baseline.
- Existing HLR, linear regression, Random Forest, GRU and mean baselines share a next-observed-session score task in `ml/src/train_next_session_model.py`. The saved champion is selected by mean validation MAE across forward-time and learner-held-out splits. The report records the GRU's lower ECE but higher MAE than HLR.
- The dashboard's current “forgetting date” is computed by interpolating the first predicted next-session score crossing below 0.80. This is an estimated review threshold, not a measured forgetting event.

## Data and model scope

The dataset header is `p_recall,timestamp,delta,user_id,learning_language,ui_language,lexeme_id,lexeme_string,history_seen,history_correct,session_seen,session_correct`. It supports interaction-sequence and next-session-score experiments. It does not contain the app's course concepts, uploaded passages, misconception annotations, or a controlled delayed-retention outcome. The current model's target is next session `session_correct / session_seen`, not delayed forgetting.

The latest evaluation report documents 617 deterministically sampled learners and 62,931 eligible next-session pairs from a 1-in-100 learner hash sample. It uses chronological and disjoint-learner splits, three seeds, MAE/RMSE/ECE, and learner-cluster bootstrap intervals. It does not report training-set metrics, retrieval/generation metrics, or app-domain validation.

## Baseline checks before changes

- `ml/src/test_prediction_pipeline.py` ran successfully as a prediction smoke script and printed three RF-1 estimates. It is not an assertion-based unit-test suite.
- `python -m pytest --collect-only -q` could not run because `pytest` is not installed in the project virtual environment; no pytest configuration or test directory was found.
- `backend/test_db_connection.py` is a live database connectivity probe, not a hermetic test. It was not run because it depends on the user's private `.env` and external Supabase/PostgreSQL access.

## Known gaps and risks

- No app-authenticated knowledge-tracing endpoint or trained LSTM/Transformer KT model existed at audit time.
- Existing RAG personalization is a deterministic concept-gap boost, not a learned reranker or a predicted mastery model. Its standard mode was not an unpersonalized baseline.
- There is no judged retrieval set or human-reviewed answer/evidence dataset, so RAG quality cannot yet be quantified.
- App course topics do not map to Duolingo lexeme IDs. Transferring Duolingo sequence-model scores to app topics would be unsupported without consented, task-matched learner data.
- Focus duration is browser-local; it is not a learning outcome and is not used as a recall target.
- No backend tests, frontend automated tests, schema migration tests, or CI checks were found.

## Existing uncommitted user changes

At audit time, `git status` showed pre-existing edits to dashboard, forecast service, ML API, README and several frontend pages, plus a new RAG chatbot route/service/page. These are user work and were kept intact while adding the research prototype.
