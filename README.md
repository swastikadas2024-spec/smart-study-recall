Smart Study Recall
Smart Study Recall is a study tracker that estimates when each topic will be forgotten, shows that date, and generates quizzes to help you review before it happens. It records study time, quiz results and recall checks, and turns them into a revision plan.
The project has three parts that run together during development:
Part	Folder	Port	Purpose
Frontend	`frontend/`	5500	Browser website: dashboard, topics, study timer, quiz, planner, progress and profile
Backend	`backend/`	8001	FastAPI service: topics, study sessions, quiz results, revision schedules and data access
ML service	`ml/`	8000	Recall models: forgetting forecast, GRU and random-forest estimates
Accounts are handled by Supabase Auth. Application data is stored in a PostgreSQL database hosted on Supabase.
> **Status:** The recall and forgetting models are prototypes and have not been validated. Every forecast in the app is labelled as an estimate.
Features
Sign up, log in and reset password through Supabase.
Dashboard with key figures, a forgetting forecast per topic, today's focus time and weekly activity.
Topics with subjects, flashcards, study sessions and recall checks.
Study timer in stopwatch, Pomodoro and countdown modes.
Quiz generated from your flashcards, with results saved as learning events.
Planner with a revision calendar, suggested reviews, scheduling, rescheduling and `.ics` export.
Progress with measured results, a forgetting curve per topic and model estimates.
Profile with account details, password change and full data export.
Repository layout
```
smart-study-recall/
├── README.md
├── frontend/              Static website (HTML, CSS, JavaScript; no build step)
│   ├── index.html · dashboard.html · topics.html · study.html
│   ├── planner.html · progress.html · profile.html
│   ├── css/app.css
│   └── js/                config.js, api.js, model.js, local.js, effects.js, one script per page
├── backend/               FastAPI application
│   ├── app/
│   │   ├── main.py        App setup, CORS, router registration, health checks
│   │   ├── config.py      Settings read from backend/.env
│   │   ├── database.py    SQLAlchemy engine and sessions
│   │   ├── models.py      Database tables
│   │   ├── auth.py        Checks Supabase access tokens
│   │   └── routers/       students, topics, learning_events, revision_schedules,
│   │                      recommendations, dashboard, topic_recall
│   ├── migrations/        SQL migrations to run in Supabase
│   ├── requirements.txt
│   └── .env.example
└── ml/
    ├── models/            Saved model files and parameters (hlr_parameters.json, recall_gru_v3.pt, ...)
    └── src/
        ├── api.py                       ML web service (port 8000)
        ├── hlr_recommendation_service.py Half-life model used by the backend
        └── …                            training and evaluation scripts
```
Requirements
Python 3.13 (the version used to test the backend)
Git
Visual Studio Code, with the Live Server extension (by Ritwick Dey) for the frontend
A Supabase project, with email sign-in enabled and the anon key and database URL available
Setup
1. Database
In Supabase, open SQL Editor and run the files in `backend/migrations/` in order. Start with `002_topic_subject_and_reschedule.sql`.
Copy the connection string from Project Settings → Database. Use the URI form and URL-encode any special characters in the password.
2. Backend (port 8001)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
Edit `backend/.env`:
```
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
Check it at `http://127.0.0.1:8001/health`. Interactive API documentation is at `http://127.0.0.1:8001/docs`.
3. ML service (port 8000, optional)
The ML service provides the recall estimates shown on the Progress page. The rest of the app works without it.
```powershell
cd ml\src
python -m uvicorn api:app --port 8000
```
Check it at `http://127.0.0.1:8000/health`. Install the Python packages the ML code needs. These include FastAPI, uvicorn, numpy, pandas, scikit-learn, joblib and PyTorch.
4. Frontend (port 5500)
Open `frontend/js/config.js` and set your values:
```js
   window.APP_CONFIG = {
     SUPABASE_URL: "https://YOUR_PROJECT_REF.supabase.co",
     SUPABASE_ANON_KEY: "YOUR_ANON_PUBLIC_KEY",
     API_URL: "http://127.0.0.1:8001"
   };
   ```
Open the `frontend` folder in VS Code.
Right-click `index.html` and choose Open with Live Server. The site opens at `http://127.0.0.1:5500/index.html`.
Use the anon (public) key in the frontend only. Never put the `service_role` key in this project.
Running everything
Keep three terminals open: the backend on 8001, the ML service on 8000 (optional), and Live Server on 5500. Then open the site and sign up.
API overview
All endpoints except `/`, `/health` and `/health/database` need a Supabase access token in the `Authorization: Bearer` header. The backend checks each token with Supabase and returns data only for the signed-in student.
Area	Endpoints
Students	`GET /students/me`, `POST /students`
Topics	`GET /topics`, `POST /topics`, `PATCH /topics/{id}`, `DELETE /topics/{id}`
Study sessions and quiz results	`GET /learning-events`, `POST /learning-events`
Revision schedule	`GET /revision-schedules`, `POST /revision-schedules`, `PATCH /revision-schedules/{id}/complete`, `PATCH /revision-schedules/{id}/reschedule`
Recommendations	`GET /recommendations/{student_id}`, `POST /recommendations/{student_id}/schedule`
Dashboard	`GET /dashboard/{student_id}`
Topic recall	`GET /topic-recall/{id}` (requires the ML service)
Health	`GET /`, `GET /health`, `GET /health/database`
The older predictions routes are deliberately not mounted. They did not check who was signed in, so any caller could read any student's history. They should be enabled only after authentication checks are added.
How the forecast works
The forgetting forecast is calculated in the browser from the saved half-life model in `ml/models/hlr_parameters.json`:
Half-life: `H = exp(θ₁ · log(1 + n) + θ₂ · accuracy)`, where `n` is the number of questions seen and `accuracy` is the share answered correctly.
Recall after `t` days: `p = 2^(−t / H)`.
A topic is due when recall falls to 80%, which happens after `H · log₂(1/0.8)` days.
The GRU model (`recall_gru_v3.pt`) gives a recall estimate for the current session. Its output does not fall smoothly over time, so it is not used to set forgetting dates.
Forecasts beyond 60 days are flagged as rough guides. Study time does not change the model. Recall checks and quiz results do.
Data storage
Data	Where it is stored
Accounts and sign-in	Supabase Auth
Students, topics, study sessions, quiz results, revision schedules	PostgreSQL database, through the backend
Flashcard text and focus-timer minutes	The user's browser only (localStorage). They are not sent to the server.
Troubleshooting
Symptom	Fix
`Could not import module "main"`	Run `uvicorn` from the folder that contains the module. For this layout use `uvicorn app.main:app --reload --port 8001` from `backend/`.
Website shows "Cannot reach the backend"	Check that the backend is running on port 8001 and the site is on port 5500. Close any other Live Server window that is using 5500.
Browser shows a CORS error	Add the page's address to `FRONTEND_ORIGINS` in `backend/.env` and restart the backend.
Progress page shows "estimate unavailable"	Start the ML service on port 8000.
Login fails with a configuration message	`frontend/js/config.js` still has placeholder values.
Database connection error	Check `DATABASE_URL` and that the password is URL-encoded. Test with `http://127.0.0.1:8001/health/database`.
Security notes
Keep `backend/.env` out of Git. The included `.gitignore` covers it.
Use the Supabase anon key in the frontend and the database URL only in `backend/.env`.
Never commit the `service_role` key, database passwords or access tokens.
Limitations and roadmap
The recall and forgetting models are prototypes and are not calibrated against measured memory.
Display names cannot be changed, because the backend has no endpoint for it.
Flashcards and focus minutes are kept in the browser, not on the server.
The mentor role and email notifications are not implemented.
Contributing
Create a branch for your change.
Test it locally, as described in Setup.
Open a pull request.
