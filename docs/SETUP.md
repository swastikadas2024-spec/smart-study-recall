# Setup and reproducibility

Follow the local Supabase, database, frontend and service setup in the root README. Keep `.env` and browser configuration secrets out of Git. Never expose a database password, Supabase service-role key or LLM key in frontend code.

## Existing ML experiment

Install the repository requirements, place the licensed Duolingo Learning Traces gzip CSV at `ml/data/raw/settles.acl16.learning_traces.13m.csv.gz`, then run:

```powershell
.\.venv\Scripts\python.exe ml\src\train_next_session_model.py
```

The job streams the dataset and uses a deterministic 1-in-100 learner sample by default. The JSON report and validation-selected artifact are written under `ml/reports/` and `ml/models/`. To start the ML API from the repository root in PowerShell, change into `ml/src` first so the relative virtual-environment path resolves correctly:

```powershell
Set-Location .\ml\src
..\..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the app API from the repo root on port 8002; serve `frontend/` on port 5500. If port 8000 is already occupied, stop the older ML API in its terminal with Ctrl+C, then run the command above.

For the separate KT candidates:

```powershell
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture mlp --protocol forward_time
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture lstm --protocol forward_time
.\.venv\Scripts\python.exe ml\src\train_knowledge_tracer.py --architecture transformer --protocol forward_time
```

The trainer supports `mlp`, `lstm` and `transformer` architectures. Run the MLP optimizer comparison with `--optimizer adamw`, `adam`, `rmsprop` and `sgd` (momentum 0.9), holding `--sample-modulus 100 --seed 42 --max-epochs 3` fixed. Recorded results are in `docs/RESULTS.md`.

Use `--protocol learner_held_out` for a disjoint learner split. The default hash sample is 1-in-100; `--sample-modulus N` changes it explicitly. Each run writes a model checkpoint and a JSON report named for its architecture and split. Reports identify the single seed/configuration used.

## Local checks

Run the hermetic app and ML unit suites from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ml\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s app\tests -v
```

`ml/src/test_prediction_pipeline.py` is an additional smoke script for the existing next-session pipeline. The database probe in `backend/test_db_connection.py` requires the private `.env` and a live database, so run it only in a configured development environment. Pytest is not installed or configured in this checkout.
