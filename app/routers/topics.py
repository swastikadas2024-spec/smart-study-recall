from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student, require_student_owner
from app.database import get_db
from app.models import Profile, Topic

router = APIRouter(prefix="/topics", tags=["Topics"])


class TopicCreate(BaseModel):
    student_id: int = Field(gt=0)
    topic_name: str = Field(min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=60)


class TopicUpdate(BaseModel):
    topic_name: str | None = Field(default=None, max_length=200)
    subject: str | None = Field(default=None, max_length=60)


class TopicResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    topic_name: str
    subject: str | None
    created_at: datetime


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _owned_topic(db: Session, topic_id: int, student: Profile) -> Topic:
    topic = db.get(Topic, topic_id)
    if topic is None or topic.student_id != student.id:
        raise HTTPException(status_code=404, detail="Topic not found.")
    return topic


@router.post(
    "",
    response_model=TopicResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_topic(
    topic: TopicCreate,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    require_student_owner(topic.student_id, current_student)

    name = topic.topic_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Topic name cannot be empty.")

    new_topic = Topic(
        student_id=current_student.id,
        topic_name=name,
        subject=_clean(topic.subject),
    )
    db.add(new_topic)
    db.commit()
    db.refresh(new_topic)
    return new_topic


@router.get("", response_model=list[TopicResponse])
def list_topics(
    student_id: int | None = None,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    if student_id is not None:
        if student_id <= 0:
            raise HTTPException(status_code=422, detail="student_id must be positive.")
        require_student_owner(student_id, current_student)

    statement = (
        select(Topic)
        .where(Topic.student_id == current_student.id)
        .order_by(Topic.id)
    )
    return db.scalars(statement).all()


@router.patch("/{topic_id}", response_model=TopicResponse)
def update_topic(
    topic_id: int,
    body: TopicUpdate,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    topic = _owned_topic(db, topic_id, current_student)
    changes = body.model_dump(exclude_unset=True)

    if "topic_name" in changes:
        name = (changes["topic_name"] or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="Topic name cannot be empty.")
        topic.topic_name = name

    if "subject" in changes:
        topic.subject = _clean(changes["subject"])

    db.commit()
    db.refresh(topic)
    return topic


@router.delete("/{topic_id}", status_code=204)
def delete_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    topic = _owned_topic(db, topic_id, current_student)
    db.delete(topic)
    db.commit()
    return Response(status_code=204)
