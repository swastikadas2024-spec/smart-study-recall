# Smart Study Recall · Frontend

A static website (HTML, CSS and JavaScript, no build step) that connects to your
Supabase project and to the FastAPI backend. The backend is not changed by this frontend.

## Pages

| Page | What it does |
|---|---|
| `index.html` | Sign up, log in, reset password (Supabase Auth). |
| `dashboard.html` | Stats, forgetting forecast per topic, today's focus time, weekly questions. |
| `topics.html` | Add and edit topics, log study sessions, flashcards, recall checks. |
| `study.html` | Focus timer (stopwatch, Pomodoro, countdown) and a generated quiz. |
| `planner.html` | Revision calendar, suggested reviews, schedule and reschedule, export to `.ics`. |
| `progress.html` | Measured results, forgetting curve per topic, activity, backend model estimates. |
| `profile.html` | Account details, change password, export all your data, clear browser-only data. |

## Run in VS Code

1. Open VS Code, then **File → Open Folder…** and choose this `frontend` folder.
2. Install the **Live Server** extension (by Ritwick Dey) if prompted.
3. Open `js/config.js` and set your values:
   ```js
   window.APP_CONFIG = {
     SUPABASE_URL: "https://YOUR-PROJECT-REF.supabase.co",
     SUPABASE_ANON_KEY: "YOUR-ANON-PUBLIC-KEY",   // the anon/public key only, never service_role
     API_URL: "http://127.0.0.1:8002"
   };
   ```
4. Start your FastAPI backend on port 8002 the way you already do (for example
   `uvicorn <your-main-module>:app --host 127.0.0.1 --port 8002`). Its `FRONTEND_ORIGINS` setting must
   include `http://127.0.0.1:5500`, which is the default.
5. Right-click `index.html` and choose **Open with Live Server**. The site opens at
   `http://127.0.0.1:5500/index.html`.

Serve the site on port 5500. The backend only accepts that origin by default.

## Browser-only data

The backend has no fields for flashcard text or focus-timer minutes, so these are stored in
this browser (localStorage), separately for each student:

- Flashcard question and answer for each topic.
- Focus minutes logged from the study timer.

They do not sync to other devices or browsers. Clearing site data removes them.
Use **Profile → Export all my data** to keep a copy.

Quiz results and logged study sessions are sent to the backend as learning events.

## Forecasts are estimates

Forgetting dates are calculated in the browser from the saved half-life model in
`ml/models/hlr_parameters.json`:

- Half-life `H = exp(θ₁·log(1 + n) + θ₂·accuracy)`.
- Recall after `t` days is `2^(−t/H)`.
- A topic is due when recall falls to 80%.

This is a prototype and has not been validated. Dates beyond about 60 days are flagged as
rough guides. Every forecast is labelled as an estimate.

The "Model estimate · experimental" figures on the Progress page come from the backend's
`/topic-recall` endpoint, which needs the ML service on port 8000 to be running.

## Folder layout

```
frontend/
├── index.html · dashboard.html · topics.html · study.html · planner.html · progress.html · profile.html
├── css/app.css          Theme ("soot and lime") and motion layer
├── js/
│   ├── config.js        Your Supabase URL, anon key and API URL
│   ├── api.js           Auth (Supabase REST), backend calls, shared helpers
│   ├── model.js         Forgetting-curve maths
│   ├── local.js         Browser-only flashcards and focus minutes
│   ├── effects.js       Reveal-on-scroll, count-up numbers
│   └── *.js             One script per page
├── fonts/               Anton and Work Sans (SIL Open Font License)
└── README.md
```

## Motion and accessibility

Animations respect the operating system's "reduce motion" setting. With it on, everything
appears immediately with no movement.
