from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student
from app.database import get_db
from app.models import LearningEvent, Profile, Topic
from app.services.next_session import predict_review_scenarios

router = APIRouter(prefix="/topic-recall", tags=["Topic Recall"])

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
        "model": "next-session champion",
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
    scenarios = predict_review_scenarios(events)
    if scenarios is None:
        return {**base, "available": False, "reason": "Log at least one study session for this topic first."}
    if not scenarios.get("available"):
        return {**base, **scenarios, "questions_seen": seen, "questions_correct": correct}

    return {
        **base,
        **scenarios,
        "questions_seen": seen,
        "questions_correct": correct,
        "accuracy_percent": round(correct / seen * 100, 2),
        "prediction": {
            "predicted_next_session_recall": scenarios["recommended_predicted_score"],
            "predicted_session_recall_rate": scenarios["recommended_predicted_score"],
            "target_type": "next-session correct/seen fraction",
            "model": scenarios["model"],
        },
    }
