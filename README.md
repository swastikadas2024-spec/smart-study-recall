# Smart Study Recall

Smart Study Recall is a study tracker that estimates when each topic will be forgotten, shows that date, and generates quizzes so you can review before it happens.

> **Prototype.** The recall and forgetting models have not been validated. Every forecast in the app is labelled as an estimate.

## Overview

The project has three services that run together during development.

| Service | Folder | Port | Purpose |
|---------|--------|------|---------|
| Frontend | `frontend/` | 5500 | Website: dashboard, topics, study timer, quiz, planner, progress, profile |
| Backend | `backend/` | 8001 | FastAPI API: topics, study sessions, quiz results, revision schedules |
| ML service | `ml/` | 8000 | Recall models: forgetting forecast, GRU and random-forest estimates |

Accounts are handled by **Supabase Auth**. Application data is stored in **PostgreSQL** on Supabase.

## Features

- Sign up, log in and reset your password.
- Dashboard with key figures, a forgetting forecast for each topic, today's focus time and weekly activity.
- Topics with subjects, flashcards, study sessions and recall checks.
- Focus timer in stopwatch, Pomodoro and countdown modes.
- Quiz generated from your flashcards, with results saved to your history.
- Planner with a revision calendar, suggested reviews, scheduling and `.ics` export.
- Progress with measured results, a forgetting curve per topic and model estimates.
- Profile with account details, password change and full data export.

## Project structure

```text
smart-study-recall/
├── README.md
├── frontend/                 Static website (HTML, CSS, JavaScript, no build step)
│   ├── index.html            Sign up, log in, reset password
│   ├── dashboard.html        Overview and forecast
│   ├── topics.html           Topics, sessions, flashcards
│   ├── study.html            Focus timer and quiz
│   ├── planner.html          Revision calendar
│   ├── progress.html         Results and forgetting curves
│   ├── profile.html          Account and data export
│   ├── css/app.css
│   └── js/                   config.js, api.js, model.js, local.js, effects.js, page scripts
├── backend/                  FastAPI application
│   ├── app/
│   │   ├── main.py           App setup, CORS, routes, health checks
│   │   ├── config.py         Settings read from backend/.env
│   │   ├── database.py       Database connection
│   │   ├── models.py         Database tables
│   │   ├── auth.py           Checks Supabase sign-in tokens
│   │   └── routers/          One file per feature area
│   ├── migrations/           SQL to run in Supabase
│   ├── requirements.txt
│   └── .env.example
└── ml/
    ├── models/               Saved models and parameters
    └── src/
        ├── api.py            ML web service
        └── hlr_recommendation_service.py
```

## Getting started

### Prerequisites

- Python 3.13
- Git
- Visual Studio Code with the **Live Server** extension (by Ritwick Dey)
- A Supabase project with email sign-in turned on

### 1. Set up the database

1. In Supabase, open **SQL Editor** and run the files in `backend/migrations/` in order, starting with `002_topic_subject_and_reschedule.sql`.
2. Copy the connection string from **Project Settings → Database** (URI format). URL-encode any special characters in the password.

### 2. Start the backend (port 8001)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `backend/.env` and fill in your values:

```env
DATABASE_URL=postgresql+psycopg2://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@YOUR_POOLER_HOST:5432/postgres
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_PUBLIC_KEY
ML_SERVICE_URL=http://127.0.0.1:8000
FRONTEND_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
```

Start the server:

```powershell
uvicorn app.main:app --reload --port 8001
```

Check it at `http://127.0.0.1:8001/health`. Interactive API docs are at `http://127.0.0.1:8001/docs`.

### 3. Start the ML service (port 8000, optional)

The recall estimates on the Progress page come from this service. The rest of the app works without it.

```powershell
cd ml\src
python -m uvicorn api:app --port 8000
```

Check it at `http://127.0.0.1:8000/health`.

### 4. Start the frontend (port 5500)

Open `frontend/js/config.js` and set your values:

```js
window.APP_CONFIG = {
  SUPABASE_URL: "https://YOUR_PROJECT_REF.supabase.co",
  SUPABASE_ANON_KEY: "YOUR_ANON_PUBLIC_KEY",
  API_URL: "http://127.0.0.1:8001"
};
```

Then:

1. Open the `frontend` folder in VS Code.
2. Right-click `index.html` and choose **Open with Live Server**.
3. The site opens at `http://127.0.0.1:5500/index.html`.

Keep three things running while you work: the backend, the ML service (optional) and Live Server.

> Use only the **anon (public)** key in the frontend. Never put the `service_role` key in this project.

## API reference

Every endpoint except `/`, `/health` and `/health/database` needs a Supabase sign-in token in the `Authorization: Bearer` header. The backend checks the token and returns only the signed-in student's data.

| Area | Endpoints |
|------|-----------|
| Students | `GET /students/me`, `POST /students` |
| Topics | `GET /topics`, `POST /topics`, `PATCH /topics/{id}`, `DELETE /topics/{id}` |
| Study sessions and quiz results | `GET /learning-events`, `POST /learning-events` |
| Revision schedule | `GET /revision-schedules`, `POST /revision-schedules`, `PATCH /revision-schedules/{id}/complete`, `PATCH /revision-schedules/{id}/reschedule` |
| Recommendations | `GET /recommendations/{student_id}`, `POST /recommendations/{student_id}/schedule` |
| Dashboard | `GET /dashboard/{student_id}` |
| Topic recall | `GET /topic-recall/{id}` (needs the ML service) |
| Health | `GET /`, `GET /health`, `GET /health/database` |

The older predictions routes are not mounted. They did not check who was signed in, so anyone could read another student's history. Enable them only after authentication checks are added.

## How the forecast works

The forgetting forecast is calculated in the browser using the saved half-life model in `ml/models/hlr_parameters.json`.

- Half-life: `H = exp(θ₁ · log(1 + n) + θ₂ · accuracy)`, where `n` is the number of questions seen and `accuracy` is the share answered correctly.
- Recall after `t` days: `p = 2^(−t / H)`.
- A topic is due when recall falls to 80%, which happens after `H · log₂(1/0.8)` days.

Forecasts beyond 60 days are flagged as rough guides. Study time does not change the model. Recall checks and quiz results do.

The GRU model (`recall_gru_v3.pt`) gives a recall estimate for the current session. Its output does not fall smoothly over time, so it is not used to set forgetting dates.

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
| "Cannot reach the backend" banner | Check that the backend runs on port 8001 and the site opens on port 5500. Close any other Live Server window. |
| Browser shows a CORS error | Add the page's address to `FRONTEND_ORIGINS` in `backend/.env` and restart the backend. |
| Progress page says the estimate is unavailable | Start the ML service on port 8000. |
| Sign-in shows a configuration message | `frontend/js/config.js` still has placeholder values. |
| Database connection error | Check `DATABASE_URL` and URL-encode the password. Test at `http://127.0.0.1:8001/health/database`. |

## Security

- Keep `backend/.env` out of Git. The `.gitignore` should list it.
- The frontend uses only the Supabase anon key. The database URL belongs only in `backend/.env`.
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
