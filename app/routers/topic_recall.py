from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student
from app.config import settings
from app.database import get_db
from app.models import LearningEvent, Profile, Topic

router = APIRouter(prefix="/topic-recall", tags=["Topic Recall"])


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.get("/{topic_id}")
def get_topic_recall(
    topic_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Experimental recall estimate for one of the student's own topics.

    Failures (no history, ML service down) return available=false with a
    reason, so the frontend can show it without an error page.
    """
    topic = db.get(Topic, topic_id)
    if topic is None or topic.student_id != current_student.id:
        raise HTTPException(status_code=404, detail="Topic not found.")

    base = {
        "topic_id": topic.id,
        "topic_name": topic.topic_name,
        "model": "RF-1",
        "experimental": True,
        "calibrated": False,
    }

    events = db.scalars(
        select(LearningEvent)
        .where(
            LearningEvent.student_id == current_student.id,
            LearningEvent.topic_id == topic.id,
            LearningEvent.questions_seen > 0,
        )
        .order_by(LearningEvent.event_time, LearningEvent.id)
    ).all()

    if not events:
        return {
            **base,
            "available": False,
            "reason": "Log at least one study session for this topic first.",
        }

    seen = sum(event.questions_seen for event in events)
    correct = sum(event.questions_correct for event in events)
    latest = _aware(events[-1].event_time)
    elapsed = max(
        (datetime.now(timezone.utc) - latest).total_seconds() / 86400,
        1 / 86400,
    )

    try:
        response = httpx.post(
            f"{settings.ml_service_url.rstrip('/')}/predict",
            json={
                "history_seen": seen,
                "history_correct": correct,
                "elapsed_days": elapsed,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        prediction = response.json()

    except httpx.TimeoutException:
        return {**base, "available": False, "reason": "The ML service timed out."}

    except (httpx.HTTPError, ValueError):
        return {
            **base,
            "available": False,
            "reason": (
                "The ML service is not running or returned an error. "
                "Start it on port 8000."
            ),
        }

    return {
        **base,
        "available": True,
        "questions_seen": seen,
        "questions_correct": correct,
        "accuracy_percent": round(correct / seen * 100, 2),
        "elapsed_days": round(elapsed, 2),
        "prediction": prediction,
    }
