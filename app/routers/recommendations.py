
from datetime import datetime, timezone, timedelta
import math

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student, require_student_owner
from app.database import get_db
from app.models import LearningEvent, Profile, RevisionSchedule, Topic
from app.services.next_session import predict_review_scenarios

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

        # Conditional model predicts next-session score at proposed intervals;
        # the review-date choice is a separate threshold policy.
        next_session_prediction = predict_review_scenarios(valid_events) if valid_events else {
            "available": False,
            "experimental": True,
            "reason": "Log at least one study session for this topic first.",
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

        # Use the model's estimated 80% crossing for priority when available;
        # retain an explicit accuracy/recency fallback for unsupported cases.
        if next_session_prediction.get("available"):
            days_to_threshold = next_session_prediction.get("days_until_predicted_forgetting")
            if next_session_prediction.get("forgetting_forecast_status") in {
                "threshold_already_crossed", "below_threshold_at_first_sample"
            } or (isinstance(days_to_threshold, (int, float)) and days_to_threshold <= 2):
                priority = "high"
                reason = "The model estimates this topic is at or below the 80% review threshold."
            elif isinstance(days_to_threshold, (int, float)) and days_to_threshold <= 7:
                priority = "medium"
                reason = f"The model estimates the 80% review threshold in about {days_to_threshold:.1f} days."
            else:
                priority = "low"
                reason = "The model estimates recall remains above the 80% review threshold for now."
        elif accuracy is None:
            priority = "high"
            reason = "No questions have been recorded for this topic; log a session to estimate its recall curve."
        elif accuracy < 0.60:
            priority = "high"
            reason = "The recall model is unavailable; recorded accuracy is below 60%."
        elif accuracy < 0.80:
            priority = "medium"
            reason = "The recall model is unavailable; recorded accuracy is below 80%."
        else:
            priority = "low"
            reason = "The recall model is unavailable; recorded accuracy is at least 80%."

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
            "recommended_review_in_days": next_session_prediction.get("recommended_review_in_days") if next_session_prediction.get("available") else None,
            "recommended_review_at": (
                now + timedelta(days=float(next_session_prediction["recommended_review_in_days"]))
                if next_session_prediction.get("available")
                and isinstance(next_session_prediction.get("recommended_review_in_days"), (int, float))
                else None
            ),
            "overdue_revisions": len(overdue_schedules),
            "suggested_action": (
                "Revise this topic and review your mistakes."
                if priority == "high"
                else "Schedule a short review."
                if priority == "medium"
                else "Continue periodic revision."
            ),
            "next_session_prediction": next_session_prediction,
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
        "recommendation_method": "model_threshold_with_rule_fallback",
        "note": "When available, priorities and review intervals use the trained next-session score curve and an 80% threshold. Forecasts are experimental proxies, not validated probabilities of long-term memory; accuracy-based rules are used when model output is unavailable.",
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
    model_policy_used = False
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

    # Predict the next-session score for candidate gaps, then apply the
    # independent review-date policy (latest gap meeting the 80% target).
    if valid_events:
        scenarios = predict_review_scenarios(valid_events, now=now)
        proposed_interval = (scenarios or {}).get("recommended_review_in_days")
        if (
            (scenarios or {}).get("available")
            and isinstance(proposed_interval, (int, float))
            and math.isfinite(proposed_interval)
            and proposed_interval > 0
        ):
            interval_days = float(proposed_interval)
            model_policy_used = True
            experimental = True
            interval_source = "next_session_score_threshold_policy"

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
        "hlr_used": False,
        "next_session_model_policy_used": model_policy_used,
        "experimental": experimental,
        "validated_for_student_topics": False if model_policy_used else None,
        "historical_accuracy_percent": (
            round(accuracy * 100, 2)
            if accuracy is not None
            else None
        ),
        "note": (
            "The model predicts next-session score conditional on a candidate gap; "
            "a separate 80% threshold policy selected this interval."
            if model_policy_used
            else "The model was unavailable, so a conservative rule-based interval was used."
        ),
    }


@router.post("/{student_id}/auto-schedule")
def auto_schedule_model_revisions(
    student_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Create one pending review for each topic with a supported model forecast."""
    require_student_owner(student_id, current_student)
    recommendation_result = get_revision_recommendations(student_id, db, current_student)
    schedules = db.scalars(
        select(RevisionSchedule).where(RevisionSchedule.student_id == student_id)
    ).all()
    now = datetime.now(timezone.utc)
    created = []
    skipped = []

    for recommendation in recommendation_result["recommendations"]:
        topic_id = recommendation["topic_id"]
        topic_schedules = [schedule for schedule in schedules if schedule.topic_id == topic_id]
        pending = [schedule for schedule in topic_schedules if not schedule.completed]
        if pending:
            skipped.append({"topic_id": topic_id, "reason": "A pending review is already scheduled."})
            continue

        completed_times = [
            schedule.scheduled_at.replace(tzinfo=timezone.utc)
            if schedule.scheduled_at.tzinfo is None else schedule.scheduled_at
            for schedule in topic_schedules if schedule.completed
        ]
        last_studied = recommendation.get("last_studied_at")
        if completed_times and (
            last_studied is None
            or (last_studied.replace(tzinfo=timezone.utc) if last_studied.tzinfo is None else last_studied)
            <= max(completed_times)
        ):
            skipped.append({"topic_id": topic_id, "reason": "Log a new study session after the completed review before scheduling the next cycle."})
            continue

        prediction = recommendation.get("next_session_prediction") or {}
        interval = prediction.get("recommended_review_in_days")
        if (
            not prediction.get("available")
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            skipped.append({
                "topic_id": topic_id,
                "reason": prediction.get("reason", "A trained model forecast is not available for this topic."),
            })
            continue

        revision = RevisionSchedule(
            student_id=student_id,
            topic_id=topic_id,
            scheduled_at=now + timedelta(days=float(interval)),
        )
        db.add(revision)
        db.flush()
        created.append({
            "id": revision.id,
            "student_id": student_id,
            "topic_id": topic_id,
            "topic_name": recommendation["topic_name"],
            "scheduled_at": revision.scheduled_at,
            "completed": False,
            "rescheduled_count": 0,
            "interval_days": interval,
            "predicted_next_session_score": prediction.get("recommended_predicted_score"),
            "target_score": prediction.get("target_score", 0.80),
            "predicted_forgetting_at": prediction.get("predicted_forgetting_at"),
            "source": "next_session_score_threshold_policy",
            "experimental": True,
        })
        schedules.append(revision)

    if created:
        db.commit()
    else:
        db.rollback()

    return {
        "student_id": student_id,
        "created_count": len(created),
        "created": created,
        "skipped": skipped,
        "recommendations": recommendation_result["recommendations"],
        "policy": "Schedule the latest supported interval whose predicted next-session score meets the 80% target.",
        "experimental": True,
    }
