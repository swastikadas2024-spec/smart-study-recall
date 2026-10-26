
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction_pipeline import predict_session_recall
from gru_prediction_pipeline import predict_gru_session_recall
from prediction import get_memory_prediction
from next_session_model import MAX_CANDIDATE_GAP_DAYS, MODEL_DIR
from next_session_service import predict_next_session
from knowledge_tracing_service import predict_next_interaction
app = FastAPI(
    title="Smart Study Recall ML API",
    description=(
        "Experimental next-session score prediction conditional on a proposed "
        "review gap. The model predicts a score; a separate policy selects a date."
    ),
    version="2.0.0",
)


class PredictionRequest(BaseModel):
    history_seen: int = Field(ge=1)
    history_correct: int = Field(ge=0)
    elapsed_days: float = Field(gt=0, allow_inf_nan=False)

class GRUPredictionRequest(BaseModel):
    history_seen: int = Field(ge=1)
    history_correct: int = Field(ge=0)
    elapsed_days: float = Field(gt=0, allow_inf_nan=False)
    recent_history: list[dict] | None = None
class HLRPredictionRequest(BaseModel):
    history_seen: int = Field(ge=1)
    history_correct: int = Field(ge=0)
    elapsed_days: float = Field(ge=0, allow_inf_nan=False)


class Interaction(BaseModel):
    questions_seen: int = Field(ge=1, le=10000)
    questions_correct: int = Field(ge=0, le=10000)
    # Gap from this historical event to the next recorded event. Ignored for
    # the final event, whose gap is supplied separately as a candidate.
    gap_days: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class NextSessionScenarioRequest(BaseModel):
    events: list[Interaction] = Field(min_length=1, max_length=500)
    candidate_gap_days: list[float] = Field(min_length=1, max_length=64)


class KnowledgeTracingInteraction(BaseModel):
    item_id: str = Field(min_length=1, max_length=120)
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    history_seen: int = Field(ge=0, le=100000)
    history_correct: int = Field(ge=0, le=100000)
    session_seen: int = Field(ge=1, le=10000)
    session_correct: int = Field(ge=0, le=10000)


class KnowledgeTracingRequest(BaseModel):
    interactions: list[KnowledgeTracingInteraction] = Field(default_factory=list, max_length=500)
    target_item_id: str = Field(min_length=1, max_length=120)
    target_history_seen: int = Field(ge=0, le=100000)
    target_history_correct: int = Field(ge=0, le=100000)
    elapsed_days: float = Field(ge=0, le=3650, allow_inf_nan=False)
    architecture: Literal["lstm", "transformer"] = "lstm"


@app.post("/predict/knowledge-tracing")
def predict_knowledge_tracing(request: KnowledgeTracingRequest):
    if request.target_history_correct > request.target_history_seen:
        raise HTTPException(status_code=422, detail="target_history_correct cannot exceed target_history_seen")
    for event in request.interactions:
        if event.session_correct > event.session_seen:
            raise HTTPException(status_code=422, detail="session_correct cannot exceed session_seen")
        if event.history_correct > event.history_seen:
            raise HTTPException(status_code=422, detail="history_correct cannot exceed history_seen")
    if any(
        request.interactions[index].timestamp > request.interactions[index + 1].timestamp
        for index in range(len(request.interactions) - 1)
    ):
        raise HTTPException(status_code=422, detail="interactions must be ordered by timestamp")
    try:
        return predict_next_interaction(
            interactions=[item.model_dump() for item in request.interactions],
            target_item_id=request.target_item_id,
            target_history_seen=request.target_history_seen,
            target_history_correct=request.target_history_correct,
            elapsed_days=request.elapsed_days,
            architecture=request.architecture,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge tracer unavailable: {exc}") from exc

@app.post("/predict/gru")
def predict_gru(request: GRUPredictionRequest):
    try:
        return predict_gru_session_recall(
            history_seen=request.history_seen,
            history_correct=request.history_correct,
            elapsed_days=request.elapsed_days,
            recent_history=request.recent_history,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "next-session champion",
        "trained_artifact_available": (MODEL_DIR / "next_session_champion.joblib").exists(),
    }


@app.post("/predict/next-session/scenarios")
def predict_next_session_scenarios(request: NextSessionScenarioRequest):
    if any(event.questions_correct > event.questions_seen for event in request.events):
        raise HTTPException(status_code=422, detail="questions_correct cannot exceed questions_seen")
    if any(
        gap <= 0 or gap > MAX_CANDIDATE_GAP_DAYS
        for gap in request.candidate_gap_days
    ):
        raise HTTPException(
            status_code=422,
            detail=f"Each candidate gap must be greater than 0 and at most {MAX_CANDIDATE_GAP_DAYS:g} days.",
        )
    events = [event.model_dump() for event in request.events]
    try:
        return {
            "target_type": "next-session correct/seen fraction",
            "predictions": [
                predict_next_session(events, gap)
                for gap in request.candidate_gap_days
            ],
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The next-session model has not been trained yet. Run ml/src/train_next_session_model.py.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"The next-session model is unavailable: {exc}") from exc


@app.post("/predict")
def predict(request: PredictionRequest):
    if request.history_correct > request.history_seen:
        raise HTTPException(
            status_code=422,
            detail="history_correct cannot exceed history_seen",
        )

    try:
        return predict_session_recall(
            history_seen=request.history_seen,
            history_correct=request.history_correct,
            elapsed_days=request.elapsed_days,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
@app.post("/predict/hlr")
def predict_hlr_revision(request: HLRPredictionRequest):
    if request.history_correct > request.history_seen:
        raise HTTPException(
            status_code=422,
            detail="history_correct cannot exceed history_seen",
        )

    try:
        result = get_memory_prediction(
            (request.history_seen, request.history_correct), request.elapsed_days
        )
        result["suggested_interval_days"] = result["recommended_revision_days"]
        result["model"] = "legacy HLR heuristic"
        result["experimental"] = True
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
