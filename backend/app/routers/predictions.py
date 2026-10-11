from datetime import datetime, timezone

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LearningEvent, Profile
import httpx
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.orm import Session
from fastapi import Depends

from app.database import get_db
from app.models import LearningEvent, Profile
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class PredictionRequest(BaseModel):
    history_seen: int = Field(ge=1)
    history_correct: int = Field(ge=0)
    elapsed_days: float = Field(gt=0, allow_inf_nan=False)


@router.post("/recall")
def predict_recall(request: PredictionRequest):
    if request.history_correct > request.history_seen:
        raise HTTPException(
            status_code=422,
            detail="Correct history cannot exceed total history seen.",
        )

    url = f"{settings.ml_service_url.rstrip('/')}/predict"

    try:
        response = httpx.post(
            url,
            json=request.model_dump(),
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The ML service timed out.",
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="The ML service returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="The ML service is unavailable.",
        ) from exc

class GRUPredictionRequest(PredictionRequest):
    recent_history: list[dict] | None = None


@router.post("/recall/gru")
def predict_recall_gru(request: GRUPredictionRequest):
    if request.history_correct > request.history_seen:
        raise HTTPException(
            status_code=422,
            detail="Correct history cannot exceed total history seen.",
        )

    if request.recent_history is not None:
        if len(request.recent_history) > 9:
            raise HTTPException(
                status_code=422,
                detail="Provide at most 9 recent history rows.",
            )

        for row in request.recent_history:
            if not isinstance(row, dict):
                raise HTTPException(
                    status_code=422,
                    detail="Each history row must be an object.",
                )

            if not {"history_seen", "history_correct", "elapsed_days"} <= row.keys():
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Each history row must contain history_seen, "
                        "history_correct, and elapsed_days."
                    ),
                )

            if (
                not isinstance(row["history_seen"], int)
                or isinstance(row["history_seen"], bool)
                or row["history_seen"] < 1
                or not isinstance(row["history_correct"], int)
                or isinstance(row["history_correct"], bool)
                or row["history_correct"] < 0
                or row["history_correct"] > row["history_seen"]
                or not isinstance(row["elapsed_days"], (int, float))
                or isinstance(row["elapsed_days"], bool)
                or not 0 < row["elapsed_days"] < float("inf")
            ):
                raise HTTPException(
                    status_code=422,
                    detail="A history row contains invalid values.",
                )

    url = f"{settings.ml_service_url.rstrip('/')}/predict/gru"

    try:
        response = httpx.post(
            url,
            json=request.model_dump(),
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The GRU ML service timed out.",
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="The GRU ML service returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="The GRU ML service is unavailable.",
        ) from exc



@router.get("/students/{student_id}/recall")
def predict_student_recall_from_history(
    student_id: int,
    db: Session = Depends(get_db),
):
    student = db.get(Profile, student_id)

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    events = db.scalars(
        select(LearningEvent)
        .where(LearningEvent.student_id == student_id)
        .order_by(LearningEvent.event_time.asc(), LearningEvent.id.asc())
    ).all()

    if not events:
        raise HTTPException(
            status_code=404,
            detail="No learning events found for this student.",
        )

    total_seen = sum(event.questions_seen for event in events)
    total_correct = sum(event.questions_correct for event in events)

    if total_seen < 1:
        raise HTTPException(
            status_code=422,
            detail="The student's learning history contains no questions.",
        )

    latest_event = events[-1]
    event_time = latest_event.event_time

    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)

    elapsed_days = max(
        (datetime.now(timezone.utc) - event_time).total_seconds()
        / 86400,
        1 / 86400,
    )

    url = f"{settings.ml_service_url.rstrip('/')}/predict"

    try:
        response = httpx.post(
            url,
            json={
                "history_seen": total_seen,
                "history_correct": total_correct,
                "elapsed_days": elapsed_days,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        prediction = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The ML service timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="The ML service returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="The ML service is unavailable.",
        ) from exc

    return {
        "student_id": student_id,
        "events_used": len(events),
        "total_questions_seen": total_seen,
        "total_questions_correct": total_correct,
        "latest_event_time": latest_event.event_time,
        "elapsed_days_since_latest_event": elapsed_days,
        "prediction": prediction,
        "history_mapping_note": (
            "Session question counts were aggregated as history totals. "
            "This is a prototype mapping, not the original model's "
            "validated historical-feature reconstruction."
        ),
    }
@router.get("/students/{student_id}/recall/gru")
def predict_student_gru_from_history(
    student_id: int,
    db: Session = Depends(get_db),
):
    student = db.get(Profile, student_id)

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    events = db.scalars(
        select(LearningEvent)
        .where(LearningEvent.student_id == student_id)
        .order_by(
            LearningEvent.event_time.asc(),
            LearningEvent.id.asc(),
        )
    ).all()

    if not events:
        raise HTTPException(
            status_code=404,
            detail="No learning events found for this student.",
        )

    total_seen = sum(event.questions_seen for event in events)
    total_correct = sum(event.questions_correct for event in events)

    if total_seen < 1:
        raise HTTPException(
            status_code=422,
            detail="The student's learning history contains no questions.",
        )

    latest_event = events[-1]
    latest_time = latest_event.event_time

    if latest_time.tzinfo is None:
        latest_time = latest_time.replace(tzinfo=timezone.utc)

    elapsed_days = max(
        (
            datetime.now(timezone.utc) - latest_time
        ).total_seconds() / 86400,
        1 / 86400,
    )

    # Use the last nine earlier events as the GRU sequence.
    # Session-level counts are an approximation of the original
    # Duolingo-derived history features, not an exact reconstruction.
    recent_events = events[:-1][-9:]

    recent_history = []

    for event in recent_events:
        event_seen = event.questions_seen
        event_correct = event.questions_correct

        # The GRU pipeline requires history_seen >= 1.
        # Skip events that contain no questions.
        if event_seen < 1:
            continue

        event_time = event.event_time

        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        event_elapsed_days = max(
            (
                latest_time - event_time
            ).total_seconds() / 86400,
            1 / 86400,
        )

        recent_history.append(
            {
                "history_seen": event_seen,
                "history_correct": event_correct,
                "elapsed_days": event_elapsed_days,
            }
        )

    # The GRU endpoint receives the latest event as the current row.
    current_seen = latest_event.questions_seen
    current_correct = latest_event.questions_correct

    if current_seen < 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "The latest event has no questions. "
                "A GRU prediction cannot be formed from it."
            ),
        )

    url = f"{settings.ml_service_url.rstrip('/')}/predict/gru"

    payload = {
        "history_seen": current_seen,
        "history_correct": current_correct,
        "elapsed_days": elapsed_days,
        "recent_history": recent_history,
    }

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
        prediction = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The GRU ML service timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="The GRU ML service returned an error.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="The GRU ML service is unavailable.",
        ) from exc

    return {
        "student_id": student_id,
        "events_used": len(events),
        "previous_event_rows_sent": len(recent_history),
        "latest_event_time": latest_event.event_time,
        "elapsed_days_since_latest_event": elapsed_days,
        "prediction": prediction,
        "history_mapping_note": (
            "The sequence uses session-level question counts as "
            "approximations of the GRU's Duolingo-derived features. "
            "This mapping is experimental and has not been validated."
        ),
    }