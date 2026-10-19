
from datetime import datetime, timezone, timedelta
import math

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student, require_student_owner
from app.config import settings
from app.database import get_db
from app.models import LearningEvent, Profile, RevisionSchedule, Topic

router = APIRouter(
    prefix="/recommendations",
    tags=["Revision Recommendations"],
)


class ScheduleRecommendedRevisionRequest(BaseModel):
    topic_id: int = Field(gt=0)


@router.get("/{student_id}")
def get_revision_recommendations(
    student_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    if student_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="Student ID must be positive.",
        )

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
    ).all()

    now = datetime.now(timezone.utc)
    recommendations = []

    for topic in topics:
        topic_events = [
            event for event in events
            if event.topic_id == topic.id
        ]

        questions_seen = sum(
            event.questions_seen for event in topic_events
        )
        questions_correct = sum(
            event.questions_correct for event in topic_events
        )

        if questions_seen == 0:
            accuracy = None
        else:
            accuracy = questions_correct / questions_seen

        valid_events = [
            event for event in topic_events
            if event.questions_seen > 0
        ]

        if valid_events:
            latest_event_time = valid_events[-1].event_time

            if latest_event_time.tzinfo is None:
                latest_event_time = latest_event_time.replace(
                    tzinfo=timezone.utc
                )

            days_since_study = max(
                0,
                (now - latest_event_time).total_seconds() / 86400,
            )
        else:
            latest_event_time = None
            days_since_study = None

        # Experimental HLR prediction based on this topic's history.
        hlr_prediction = {
            "available": False,
            "experimental": True,
            "reason": "No usable learning history for this topic.",
        }

        if questions_seen > 0 and days_since_study is not None:
            try:
                response = httpx.post(
                    f"{settings.ml_service_url.rstrip('/')}/predict/hlr",
                    json={
                        "history_seen": questions_seen,
                        "history_correct": questions_correct,
                        "elapsed_days": days_since_study,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()

                hlr_prediction = {
                    "available": True,
                    **response.json(),
                }

            except httpx.TimeoutException:
                hlr_prediction = {
                    "available": False,
                    "experimental": True,
                    "reason": "The HLR service timed out.",
                }

            except httpx.HTTPError:
                hlr_prediction = {
                    "available": False,
                    "experimental": True,
                    "reason": (
                        "The HLR service is unavailable "
                        "or returned an error."
                    ),
                }

        topic_schedules = [
            schedule for schedule in schedules
            if schedule.topic_id == topic.id
        ]

        pending_schedules = [
            schedule for schedule in topic_schedules
            if not schedule.completed
        ]

        overdue_schedules = [
            schedule for schedule in pending_schedules
            if (
                schedule.scheduled_at.replace(tzinfo=timezone.utc)
                if schedule.scheduled_at.tzinfo is None
                else schedule.scheduled_at
            ) <= now
        ]

        # Transparent rule-based priority; not an ML prediction.
        if accuracy is None:
            priority = "high"
            reason = "No questions have been recorded for this topic."
        elif accuracy < 0.60:
            priority = "high"
            reason = "Recorded accuracy is below 60%."
        elif accuracy < 0.80:
            priority = "medium"
            reason = "Recorded accuracy is below 80%."
        else:
            priority = "low"
            reason = "Recorded accuracy is at least 80%."

        if overdue_schedules:
            priority = "high"
            reason += " An existing revision is overdue."

        if days_since_study is not None and days_since_study >= 7:
            if priority == "low":
                priority = "medium"
            reason += " The topic has not been studied for at least 7 days."

        if pending_schedules:
            next_revision_at = min(
                schedule.scheduled_at
                if schedule.scheduled_at.tzinfo is not None
                else schedule.scheduled_at.replace(tzinfo=timezone.utc)
                for schedule in pending_schedules
            )
        else:
            next_revision_at = None

        recommendations.append({
            "topic_id": topic.id,
            "topic_name": topic.topic_name,
            "priority": priority,
            "reason": reason,
            "questions_seen": questions_seen,
            "questions_correct": questions_correct,
            "accuracy_percent": round(accuracy * 100, 2)
            if accuracy is not None else None,
            "days_since_last_study": round(days_since_study, 2)
            if days_since_study is not None else None,
            "last_studied_at": latest_event_time,
            "next_existing_revision_at": next_revision_at,
            "overdue_revisions": len(overdue_schedules),
            "suggested_action": (
                "Revise this topic and review your mistakes."
                if priority == "high"
                else "Schedule a short review."
                if priority == "medium"
                else "Continue periodic revision."
            ),
            "hlr_prediction": hlr_prediction,
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(
        key=lambda item: (
            priority_order[item["priority"]],
            -(
                item["days_since_last_study"]
                if item["days_since_last_study"] is not None
                else -1
            ),
        )
    )

    return {
        "student_id": student.id,
        "student_name": student.student_name,
        "recommendation_method": "rule_based",
        "note": (
            "Priorities use recorded accuracy, study recency, and "
            "overdue revisions. They are heuristic recommendations, "
            "not validated predictions of forgetting."
        ),
        "recommendations": recommendations,
    }


@router.post("/{student_id}/schedule", status_code=201)
def schedule_recommended_revision(
    student_id: int,
    request: ScheduleRecommendedRevisionRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Create a revision schedule using history and available predictions."""

    if student_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="Student ID must be positive.",
        )

    require_student_owner(student_id, current_student)

    student = db.get(Profile, student_id)
    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    topic = db.get(Topic, request.topic_id)
    if topic is None or topic.student_id != student_id:
        raise HTTPException(
            status_code=404,
            detail="Topic not found for this student.",
        )

    schedules = db.scalars(
        select(RevisionSchedule).where(
            RevisionSchedule.student_id == student_id,
            RevisionSchedule.topic_id == request.topic_id,
        )
    ).all()

    pending = [
        schedule for schedule in schedules
        if not schedule.completed
    ]

    if pending:
        raise HTTPException(
            status_code=409,
            detail=(
                "An incomplete revision already exists for this topic. "
                "Complete it before scheduling another."
            ),
        )

    events = db.scalars(
        select(LearningEvent)
        .where(
            LearningEvent.student_id == student_id,
            LearningEvent.topic_id == request.topic_id,
        )
        .order_by(LearningEvent.event_time)
    ).all()

    valid_events = [
        event for event in events
        if event.questions_seen > 0
    ]
    questions_seen = sum(
        event.questions_seen for event in valid_events
    )
    questions_correct = sum(
        event.questions_correct for event in valid_events
    )

    now = datetime.now(timezone.utc)
    hlr_used = False
    experimental = False
    interval_source = "rule_based_fallback"

    # Conservative fallback: no history or weak accuracy means an early review.
    accuracy = (
        questions_correct / questions_seen
        if questions_seen > 0
        else None
    )

    if accuracy is None or accuracy < 0.60:
        interval_days = 1
    elif accuracy < 0.80:
        interval_days = 3
    else:
        interval_days = 7

    # Use HLR only when the service returns a usable, finite interval.
    if valid_events:
        latest_time = valid_events[-1].event_time
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)

        elapsed_days = max(
            0.0,
            (now - latest_time).total_seconds() / 86400,
        )

        try:
            response = httpx.post(
                f"{settings.ml_service_url.rstrip('/')}/predict/hlr",
                json={
                    "history_seen": questions_seen,
                    "history_correct": questions_correct,
                    "elapsed_days": elapsed_days,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            prediction = response.json()

            proposed_interval = float(
                prediction["suggested_interval_days"]
            )

            if (
                math.isfinite(proposed_interval)
                and proposed_interval > 0
            ):
                # Operational safety cap; this does not validate the model.
                interval_days = max(
                    1,
                    min(30, round(proposed_interval)),
                )
                hlr_used = True
                experimental = True
                interval_source = "experimental_hlr_capped"

        except (
            httpx.HTTPError,
            ValueError,
            TypeError,
            KeyError,
        ):
            # Keep the rule-based fallback if HLR is unavailable or invalid.
            pass

    scheduled_at = now + timedelta(days=interval_days)

    revision = RevisionSchedule(
        student_id=student_id,
        topic_id=request.topic_id,
        scheduled_at=scheduled_at,
    )

    try:
        db.add(revision)
        db.commit()
        db.refresh(revision)
    except Exception:
        db.rollback()
        raise

    return {
        "message": "Revision scheduled successfully.",
        "revision_id": revision.id,
        "student_id": student_id,
        "topic_id": topic.id,
        "topic_name": topic.topic_name,
        "scheduled_at": revision.scheduled_at,
        "interval_days": interval_days,
        "interval_source": interval_source,
        "hlr_used": hlr_used,
        "experimental": experimental,
        "validated_for_student_topics": False if hlr_used else None,
        "historical_accuracy_percent": (
            round(accuracy * 100, 2)
            if accuracy is not None
            else None
        ),
        "note": (
            "HLR is experimental and capped for scheduling."
            if hlr_used
            else "A rule-based interval was used because no usable "
                 "HLR interval was available."
        ),
    }
