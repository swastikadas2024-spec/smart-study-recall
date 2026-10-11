
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LearningEvent, Profile, Topic

router = APIRouter(
    prefix="/learning-events",
    tags=["Learning Events"],
)


class LearningEventCreate(BaseModel):
    student_id: int = Field(gt=0)
    topic_id: int | None = Field(default=None, gt=0)
    questions_seen: int = Field(ge=0)
    questions_correct: int = Field(ge=0)


class LearningEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    topic_id: int | None
    questions_seen: int
    questions_correct: int
    event_time: datetime


@router.post(
    "",
    response_model=LearningEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_learning_event(
    event: LearningEventCreate,
    db: Session = Depends(get_db),
):
    if event.questions_correct > event.questions_seen:
        raise HTTPException(
            status_code=422,
            detail="Correct answers cannot exceed questions seen.",
        )

    student = db.get(Profile, event.student_id)
    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    if event.topic_id is not None:
        topic = db.get(Topic, event.topic_id)
        if topic is None or topic.student_id != event.student_id:
            raise HTTPException(
                status_code=404,
                detail="Topic not found for this student.",
            )

    new_event = LearningEvent(
        student_id=event.student_id,
        topic_id=event.topic_id,
        questions_seen=event.questions_seen,
        questions_correct=event.questions_correct,
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return new_event


@router.get("", response_model=list[LearningEventResponse])
def list_learning_events(
    student_id: int,
    db: Session = Depends(get_db),
):
    if student_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="student_id must be positive.",
        )

    statement = (
        select(LearningEvent)
        .where(LearningEvent.student_id == student_id)
        .order_by(LearningEvent.event_time.desc())
    )

    return db.scalars(statement).all()
