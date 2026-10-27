# Repository implementation audit

Audit date: 2026-10-27. Latest inspected commit: `05c5587` (`Add research prototype and adaptive study tools`). See [RECENT_COMMIT_REVIEW.md](RECENT_COMMIT_REVIEW.md) for the file-by-file change summary. At audit time the worktree was clean and `main` matched its local `origin/main` tracking ref; network fetch was blocked by write permission on `.git/FETCH_HEAD`, so no fresh remote comparison was possible.

## Architecture

- Frontend: static HTML, CSS and browser JavaScript under `frontend/`.
- App API: FastAPI and SQLAlchemy under `app/`; Supabase bearer authentication and PostgreSQL configuration from the repository-root `.env`.
- ML API: separate FastAPI service under `ml/src/api.py`, usually bound to localhost on port 8000.
- Data: local ignored Duolingo Learning Traces gzip at `ml/data/raw/settles.acl16.learning_traces.13m.csv.gz`; it is not included in Git.
- Persistence: existing SQL migrations and tables for users/profiles, topics, learning events, schedules, study materials/chunks and material quiz sessions/responses.
- Tooling: no frontend build system, CI, Docker, pytest configuration, linter, formatter or type checker was found. Tests are Python `unittest` suites.

## Working features

- Authenticated topic, learning-event, dashboard, schedule, recommendation, material upload and study-tool routes.
- Topic sessions and answer correctness are persisted as learning events. Manual focus duration and flashcards are browser-local; duration is not a recall label.
- Material upload extracts PDF, DOCX, TXT and Markdown, stores passage chunks and supports notes, quiz generation and grounded chat with provider or extractive fallback.
- `frontend/chatbot.html` invokes the existing app RAG API with Standard BM25, Hybrid BM25+TF-IDF/RRF, or experimental Recall-aware retrieval. The latter reorders evidence using observed material-quiz concept gaps and recency. It is an interpretable heuristic, not a neural reranker.
- The dashboard invokes the existing candidate-gap service using a validation-selected HLR champion. HLR, linear, Random Forest, GRU and mean baselines have an evaluation report. The dashboard's 80% crossing is a policy-derived estimate, not a validated forgetting date.
- The separate ML API serves next-interaction KT checkpoints. MLP, LSTM and Transformer input share the same causal per-event feature windows with a same-timestamp boundary; preprocessing is fit on the training partition. The MLP uses a masked order-agnostic event summary, while the recurrent/attention candidates model order.

## Prediction target, data, and model limits

The Duolingo target is observed `session_correct / session_seen` for a next interaction/session. It is not delayed recall, course-topic mastery, or a confirmed time of forgetting. The trace has lexeme IDs and interaction counts but no app material concept mapping, misconception labels, passage relevance judgments, or app-specific delayed test outcomes. Transfer to this app is unvalidated.

The existing next-session comparison uses 617 sampled learners and 62,931 examples, deterministic learner sampling, temporal and learner-held-out splits, three seeds, and learner-cluster intervals. The new MLP/LSTM/Transformer KT forward-time comparison uses a 119,140-event / 1,118-learner 1-in-100 sample, one seed, and timestamp-group-safe 70/15/15 cutoffs. A separate MLP/LSTM learner-held-out comparison uses the same smaller 1-in-500 sample. The older unqualified `knowledge_tracing_evaluation.json` is a separate 1-in-500 forward-time LSTM run. All KT metrics are preliminary and must not be mixed across samples/protocols.

## RAG state and evidence

Standard and hybrid retrieval are non-personalized baselines. Recall-aware mode uses material quiz history to derive heuristic concept gaps and time since last quiz, then applies intent-sensitive reranking. Answer citations are filtered to retrieved chunk IDs. The system has no reviewed query-passage benchmark, answer-support annotations, persistent user-state evaluation log, retrieval metrics, or learning-outcome study; these are **Not evaluated**.

## Test status

Before follow-up edits, baseline unit suites passed:

- `.\.venv\Scripts\python.exe -m unittest discover -s ml\tests -v`: 6 passed.
- `.\.venv\Scripts\python.exe -m unittest discover -s app\tests -v`: 5 passed.

After follow-up edits, `.\.venv\Scripts\python.exe -m unittest discover -s ml\tests -v` ran 8 tests and `.\.venv\Scripts\python.exe -m unittest discover -s app\tests -v` ran 5 tests; both passed. Python compilation and Node syntax checks passed, as did `git diff --check`. The MLP model loaded from its forward-time checkpoint and returned a local next-interaction prediction. New MLP/optimizer results use seed 42 and a three-epoch maximum; they are preliminary single-run measurements.

The PyTorch Transformer test emits a benign nested-tensor warning. A prior FastAPI TestClient health smoke request stalled on this Windows runtime, so a full live endpoint/database health smoke was inconclusive. Direct LSTM checkpoint inference succeeded. The live Supabase database was not contacted; it uses private credentials. Pytest is not installed. The database probe is not a hermetic test.

## Technical debt and next work

- Repeat MLP/optimizer and LSTM/Transformer comparisons with multiple seeds and learner-held-out splits using the same sample.
- Build and review course-topic delayed assessment data before using these models for the dashboard's topic-level forecast.
- Create human-judged passage relevance and answer-grounding sets before reporting RAG metrics.
- Improve robust API integration/smoke testing and add frontend behavior tests.
- Perform a literature and patent search before asserting novelty. Integration of known methods is not itself evidence of novelty.
