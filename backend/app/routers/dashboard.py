
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student, require_student_owner
from app.database import get_db
from app.models import LearningEvent, Profile, RevisionSchedule, Topic

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def normalize_datetime(value: datetime) -> datetime:
    """Ensure a datetime is timezone-aware, treating naive values as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.get("/{student_id}")
def get_dashboard(
    student_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    if student_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="Student ID must be positive.",
        )

    # A student can access only their own dashboard.
    require_student_owner(student_id, current_student)

    student = db.get(Profile, student_id)
    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    topics = db.scalars(
        select(Topic)
        .where(Topic.student_id == student_id)
        .order_by(Topic.id)
    ).all()

    events = db.scalars(
        select(LearningEvent)
        .where(LearningEvent.student_id == student_id)
        .order_by(LearningEvent.event_time)
    ).all()

    schedules = db.scalars(
        select(RevisionSchedule)
        .where(RevisionSchedule.student_id == student_id)
        .order_by(RevisionSchedule.scheduled_at)
    ).all()

    now = datetime.now(timezone.utc)

    total_seen = sum(event.questions_seen for event in events)
    total_correct = sum(event.questions_correct for event in events)

    topic_performance = []

    for topic in topics:
        topic_events = [
            event for event in events
            if event.topic_id == topic.id
        ]

        seen = sum(event.questions_seen for event in topic_events)
        correct = sum(event.questions_correct for event in topic_events)

        topic_performance.append({
            "topic_id": topic.id,
            "topic_name": topic.topic_name,
            "events_count": len(topic_events),
            "questions_seen": seen,
            "questions_correct": correct,
            "accuracy_percent": round(correct / seen * 100, 2)
            if seen > 0 else None,
        })

    upcoming_count = 0
    overdue_count = 0

    for schedule in schedules:
        if schedule.completed:
            continue

        scheduled_at = normalize_datetime(schedule.scheduled_at)

        if scheduled_at > now:
            upcoming_count += 1
        else:
            overdue_count += 1

    revision_status = {
        "total": len(schedules),
        "completed": sum(schedule.completed for schedule in schedules),
        "upcoming": upcoming_count,
        "overdue": overdue_count,
    }

    return {
        "student_id": student.id,
        "student_name": student.student_name,
        "summary": {
            "topics_count": len(topics),
            "learning_events_count": len(events),
            "questions_seen": total_seen,
            "questions_correct": total_correct,
            "overall_accuracy_percent": round(
                total_correct / total_seen * 100, 2
            ) if total_seen > 0 else None,
        },
        "topic_performance": topic_performance,
        "revision_status": revision_status,
        "learning_history": [
            {
                "event_id": event.id,
                "topic_id": event.topic_id,
                "questions_seen": event.questions_seen,
                "questions_correct": event.questions_correct,
                "event_time": event.event_time,
            }
            for event in reversed(events[-20:])
        ],
    }
