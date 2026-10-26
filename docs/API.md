# API guide

The app API is FastAPI at port 8002. Except `/`, `/health`, `/health/database`, and public RAG architecture metadata, application routes require a Supabase bearer token. Student-scoped routes resolve the authenticated profile and enforce ownership.

## Existing routes

See `/docs` on a running backend for the complete OpenAPI schema. Main route groups include:

- `/students/me`, `/students`
- `/topics`
- `/learning-events`
- `/revision-schedules`
- `/recommendations/{student_id}` and `/dashboard/{student_id}`
- `/topic-recall/{topic_id}`
- `/materials` and material notes/quiz/chat
- `/rag/architecture`, `/rag/{material_id}/architecture`, and RAG chat/quiz/notes routes

The ML API runs separately on port 8000. `/predict/next-session/scenarios` receives ordered historical question counts and candidate gaps and returns a next-session correct fraction. The API does not return a validated forgetting probability.

## RAG request modes

RAG material chat accepts a question, mode (`standard`, `hybrid`, or `adaptive`), bounded conversation history and an optional diagnostic flag. Standard is BM25-only; hybrid is BM25 + TF-IDF/RRF. Both are non-personalized retrieval baselines. Adaptive adds a deterministic concept-gap and elapsed-quiz-time reranker, and returns the state signals and intervention heuristic used. Citations are restricted to retrieved chunk IDs.

## ML knowledge-tracing endpoint

`POST http://127.0.0.1:8000/predict/knowledge-tracing` accepts up to 500 timestamp-ordered interactions, a target item ID, pre-event history counts, elapsed days, and an architecture (`lstm` or `transformer`). The endpoint requires the corresponding trained artifact and otherwise returns 503. It returns an experimental next-interaction correct fraction, OOV status and explicit limitations. The local ML service is not mounted through the authenticated app API because the Duolingo item vocabulary does not map to app course topics. Do not use this score as a course-topic forgetting probability.

All request examples and validation constraints are defined by the Pydantic models in `app/routers` and `ml/src/api.py`.
