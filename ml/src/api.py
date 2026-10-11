
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction_pipeline import predict_session_recall
from gru_prediction_pipeline import predict_gru_session_recall

app = FastAPI(
    title="Smart Study Recall ML API",
    description="Recall-rate prediction using the RF-1 model",
    version="1.0.0",
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
        "model": "RF-1",
    }


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
