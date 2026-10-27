# Smart Study Recall

Smart Study Recall is a study tracker that estimates when each topic will be forgotten, shows that date, and generates quizzes so you can review before it happens.

> **Prototype.** The recall and forgetting models have not been validated. Every forecast in the app is labelled as an estimate.

## Overview

The project has three services that run together during development.

| Service | Folder | Port | Purpose |
|---------|--------|------|---------|
| Frontend | `frontend/` | 5500 | Website: dashboard, topics, study timer, quiz, planner, progress, profile |
| Backend | `app/` | 8002 | FastAPI API: student records, faculty materials, RAG study tools, schedules |
| ML service | `ml/` | 8000 | Candidate-gap next-session score model and baselines |

Accounts are handled by **Supabase Auth**. Application data is stored in **PostgreSQL** on Supabase.

## Features

- Sign up, log in and reset your password.
- Dashboard with key figures, next-session score scenarios, today's focus time and weekly activity.
- Topics with subjects, flashcards, study sessions and recall checks.
- Faculty study library with source-cited notes, quizzes, and a material chatbot.
- Recall Compass, which uses repeated material-quiz misses to prioritize related source passages and quiz concepts.
- RAG chatbot with standard BM25, unpersonalized BM25 + TF-IDF hybrid, and an experimental recall-aware mode based on observed quiz gaps and exposure recency.
- Focus timer in stopwatch, Pomodoro and countdown modes, plus manual study-hour logging. Focus duration is browser-local and is not a recall label.
- Quiz generated from your flashcards, with results saved to your history.
- Planner with a revision calendar, suggested reviews, scheduling and `.ics` export.
- Progress with measured results, candidate review intervals and experimental model estimates.
- Profile with account details, password change and full data export.

## Project structure

```text
smart-study-recall/
├── README.md
├── .env.example              Backend and optional AI provider settings
├── app/                      Authenticated FastAPI backend
│   ├── main.py               App setup, CORS and route registration
│   ├── models.py             PostgreSQL models
│   ├── routers/              Student, topic, schedule and material APIs
│   ├── services/             RAG extraction, retrieval and generation
│   └── tests/                Hermetic app/RAG unit tests
├── migrations/               SQL migrations for Supabase
├── frontend/                 Static website (HTML, CSS, JavaScript, no build step)
│   ├── index.html            Sign up, log in, reset password
│   ├── dashboard.html        Overview and forecast
│   ├── topics.html           Topics, sessions, flashcards
│   ├── materials.html        Faculty library and AI study workspace
│   ├── study.html            Focus timer and quiz
│   ├── planner.html          Revision calendar
│   ├── progress.html         Results and forgetting curves
│   ├── profile.html          Account and data export
│   ├── css/app.css
│   └── js/                   config.js, api.js, model.js, local.js, effects.js, page scripts
├── backend/                  Legacy API files kept for existing deployments
└── ml/
    ├── models/               Saved models and parameters
    └── src/
        ├── api.py            ML web service
        ├── next_session_model.py  HLR/GRU next-session score contract
        ├── knowledge_tracing.py   MLP, LSTM and Transformer KT candidates
        └── train_knowledge_tracer.py
```

## Getting started

### Prerequisites

- Python 3.13
- Git
- Visual Studio Code with the **Live Server** extension (by Ritwick Dey)
- A Supabase project with email sign-in turned on

### 1. Set up the database

1. In Supabase, open **SQL Editor** and run `migrations/002_topic_subject_and_reschedule.sql`, then `migrations/003_ai_study_materials.sql`, then `migrations/004_rag_chat_history.sql` to enable persistent chatbot history and saved source-support feedback.
2. Any signed-in user can upload materials; no faculty role change is needed. Published materials appear in the shared library for all signed-in users. Unpublished materials remain visible only to their uploader.
3. Copy the connection string from **Project Settings → Database** (URI format). URL-encode any special characters in the password.

Supabase Auth and the configured database are still required; the backend stores extracted passages and generated study data in that database.

### 2. Start the backend (port 8002)

```powershell
cd smart-study-recall
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and fill in your values:

```env
DATABASE_URL=postgresql+psycopg2://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@YOUR_POOLER_HOST:5432/postgres
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_PUBLIC_KEY
ML_SERVICE_URL=http://127.0.0.1:8000
FRONTEND_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
AI_API_KEY=YOUR_GEMINI_API_KEY
AI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AI_MODEL=gemini-3.8-flash
```

Start the server:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Check it at `http://127.0.0.1:8002/health`. Interactive API docs are at `http://127.0.0.1:8002/docs`.

The AI key and model are optional. Without them, uploaded materials still support extractive notes, source passage retrieval, and extractive chat. With OpenAI configured, the app uses the Responses API. Gemini is also supported through Google's OpenAI-compatible Chat Completions endpoint; set `AI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai` and choose an available Gemini text model. Keep the key in the repository-root `.env`; never put it in frontend JavaScript. Retrieved excerpts are sent to the configured AI provider.

Faculty can upload PDF, DOCX, TXT, and Markdown files up to 12 MB. Scanned PDFs need OCR before upload. The RAG responses include source excerpts; verify important details against the original material.

### 3. Train and start the ML service (port 8000)

Place the Duolingo Learning Traces file at `ml/data/raw/settles.acl16.learning_traces.13m.csv.gz`. Train and evaluate the next-session model from the repository root:

```powershell
.\.venv\Scripts\python.exe ml/src/train_next_session_model.py
```

The training job streams the full dataset and retains a deterministic 1-in-100 sample of learners by default. It compares a mean baseline, fitted HLR, linear regression, Random Forest, and GRU on the same forward-time and learner-held-out splits, across three seeds. It writes the evaluation, learner-cluster confidence intervals, calibration summaries, split sizes, and validation-selected deployment model to `ml/reports/next_session_evaluation.json`. Increase coverage with `--user-sample-modulus 20` if the machine has enough memory and training time.

Start the service in a separate terminal:

```powershell
cd ml\src
..\..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Check it at `http://127.0.0.1:8000/health`. The health response indicates whether the trained champion artifact is available. If the ML service is down or no artifact exists, the app uses a clearly labelled conservative schedule fallback.

### 4. Start the frontend (port 5500)

Open `frontend/js/config.js` and set your values:

```js
window.APP_CONFIG = {
  SUPABASE_URL: "https://YOUR_PROJECT_REF.supabase.co",
  SUPABASE_ANON_KEY: "YOUR_ANON_PUBLIC_KEY",
  API_URL: "http://127.0.0.1:8002"
};
```

Then:

1. Open the `frontend` folder in VS Code.
2. Right-click `index.html` and choose **Open with Live Server**.
3. The site opens at `http://127.0.0.1:5500/index.html`.

Keep three things running while you work: the backend, the trained ML service and Live Server.

> Use only the **anon (public)** key in the frontend. Never put the `service_role` key in this project.

## API reference

Every application endpoint except `/`, `/health`, `/health/database` and the static `GET /rag/architecture` metadata route needs a Supabase sign-in token in the `Authorization: Bearer` header. The backend checks the token and returns only the signed-in student's data.

| Area | Endpoints |
|------|-----------|
| Students | `GET /students/me`, `POST /students` |
| Topics | `GET /topics`, `POST /topics`, `PATCH /topics/{id}`, `DELETE /topics/{id}` |
| Study sessions and quiz results | `GET /learning-events`, `POST /learning-events` |
| Revision schedule | `GET /revision-schedules`, `POST /revision-schedules`, `PATCH /revision-schedules/{id}/complete`, `PATCH /revision-schedules/{id}/reschedule` |
| Recommendations | `GET /recommendations/{student_id}`, `POST /recommendations/{student_id}/schedule` |
| Dashboard | `GET /dashboard/{student_id}` |
| Topic recall | `GET /topic-recall/{id}` (needs the ML service) |
| AI study materials | `GET/POST /materials`, `POST /materials/{id}/notes`, `/quiz`, `/chat` |
| RAG chatbot | `GET /rag/architecture`, authenticated `GET /rag/{id}/architecture`, `POST /rag/{id}/chat`, `/quiz`, `/notes` |
| Health | `GET /`, `GET /health`, `GET /health/database` |

The older predictions routes are not mounted. They did not check who was signed in, so anyone could read another student's history. Enable them only after authentication checks are added.

The separate local ML service also exposes `POST /predict/knowledge-tracing`. It is not an authenticated public app endpoint; run it bound to localhost. Its target is next-interaction session correctness on the Duolingo task, and its item vocabulary is not mapped to app topics.

## Prediction task and review policy

The prediction task is narrowly defined: **given a learner's ordered study events through the current session and a proposed gap of at most 30 days, predict the next session's correct-answer fraction** (`next_session_correct / next_session_seen`). The predictor does not choose a review date.

The app evaluates a fixed set of candidate delays. A separate policy selects the latest candidate whose predicted next-session score is at least 80%; if none meets the threshold, it chooses the earliest candidate. This policy is experimental and is not evidence that the chosen interval causes better retention.

The dashboard also plots each studied topic's predicted next-session score over the model's supported 30-day horizon. It estimates a review-by date by interpolating where that score curve first crosses below 80%. Treat this as a reminder threshold derived from the next-session predictor, not a measured or guaranteed date when a learner will forget.

`ml/reports/next_session_evaluation.json` records the sampled population, test metrics, 95% learner-cluster bootstrap intervals, calibration error, model comparisons, and validation-selected champion. The public-trace task is a proxy: it predicts a language-learning session score, not course-concept memory. Do not describe it as a validated forgetting probability. See `docs/learner-pilot-protocol.md` for the consented adaptive-versus-fixed learner pilot needed to assess transfer and learning outcomes.

## Knowledge-tracing research prototype

`ml/src/knowledge_tracing.py` implements a PyTorch MLP baseline, LSTM and Transformer for predicting the next observed Duolingo interaction's session-correct fraction. Example training commands:

```powershell
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture mlp --protocol forward_time
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture lstm --protocol forward_time
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture transformer --protocol forward_time
```

The trainer supports AdamW, Adam, RMSProp and SGD with momentum. Its recorded one-seed optimizer comparison is in `docs/RESULTS.md`.

The script streams the local gzip file and defaults to a deterministic 1-in-100 learner sample. It writes per-protocol artifacts and actual metrics under `ml/models/` and `ml/reports/`. The ML API exposes `POST /predict/knowledge-tracing` after the corresponding artifacts are trained. This next-interaction task is not delayed retention, and Duolingo lexeme IDs do not map to app course topics. The KT models therefore are not wired into the course-topic forgetting-date dashboard. The current runs use one seed and remain experimental; see `docs/RESULTS.md`.

## Research documentation

See `docs/RECENT_COMMIT_REVIEW.md`, `docs/IMPLEMENTATION_AUDIT.md`, `docs/RESEARCH_PROPOSAL.md`, `docs/MODEL_ARCHITECTURE.md`, `docs/DATASET_AND_PREPROCESSING.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/RESULTS.md`, `docs/API.md`, `docs/SETUP.md`, and `docs/RESEARCH_LIMITATIONS.md` for the commit audit, model/data contracts, protocols and limitations.

## Data storage

| Data | Where it lives |
|------|----------------|
| Accounts and sign-in | Supabase Auth |
| Students, topics, study sessions, quiz results, revision schedules | PostgreSQL, through the backend |
| Flashcard text and focus-timer minutes | Your browser only (localStorage) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not import module "main"` | Use `uvicorn app.main:app` from inside the `backend` folder. |
| "Cannot reach the backend" banner | Check that the backend runs on port 8002 and the site opens on port 5500. Close any other Live Server window. |
| Browser shows a CORS error | Add the page's address to `FRONTEND_ORIGINS` in the repository-root `.env` and restart the backend. |
| Progress page says the estimate is unavailable | Start the ML service on port 8000. |
| Sign-in shows a configuration message | `frontend/js/config.js` still has placeholder values. |
| Database connection error | Check `DATABASE_URL` and URL-encode the password. Test at `http://127.0.0.1:8002/health/database`. |

## Security

- Keep `.env` out of Git. Never commit an AI provider key.
- The frontend uses only the Supabase anon key. The database URL belongs only in the repository-root `.env`.
- Never commit a `service_role` key, a database password or an access token.

## Limitations

- The recall models are prototypes and are not calibrated against measured memory.
- Display names cannot be changed yet, because the backend has no endpoint for it.
- Flashcards and focus minutes are stored in the browser, not on the server.
- The mentor role and email notifications are not implemented.

## Contributing

1. Create a branch for your change.
2. Test it locally by following the steps above.
3. Open a pull request.
