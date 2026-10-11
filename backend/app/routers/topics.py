
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Profile, Topic

router = APIRouter(prefix="/topics", tags=["Topics"])


class TopicCreate(BaseModel):
    student_id: int = Field(gt=0)
    topic_name: str = Field(min_length=1, max_length=200)


class TopicResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    topic_name: str
    created_at: datetime


@router.post(
    "",
    response_model=TopicResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_topic(
    topic: TopicCreate,
    db: Session = Depends(get_db),
):
    topic_name = topic.topic_name.strip()

    if not topic_name:
        raise HTTPException(
            status_code=422,
            detail="Topic name cannot be empty.",
        )

    student = db.get(Profile, topic.student_id)

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    new_topic = Topic(
        student_id=topic.student_id,
        topic_name=topic_name,
    )

    db.add(new_topic)
    db.commit()
    db.refresh(new_topic)

    return new_topic


@router.get("", response_model=list[TopicResponse])
def list_topics(
    student_id: int = None,
    db: Session = Depends(get_db),
):
    statement = select(Topic).order_by(Topic.id)

    if student_id is not None:
        if student_id <= 0:
            raise HTTPException(
                status_code=422,
                detail="student_id must be positive.",
            )
        statement = statement.where(Topic.student_id == student_id)

    return db.scalars(statement).all()
