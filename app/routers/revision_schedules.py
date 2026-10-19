from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_student, require_student_owner
from app.database import get_db
from app.models import Profile, RevisionSchedule, Topic

router = APIRouter(
    prefix="/revision-schedules",
    tags=["Revision Schedules"],
)


class RevisionCreate(BaseModel):
    student_id: int = Field(gt=0)
    topic_id: int = Field(gt=0)
    scheduled_at: datetime


class RevisionReschedule(BaseModel):
    scheduled_at: datetime


class RevisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    topic_id: int
    scheduled_at: datetime
    completed: bool
    rescheduled_count: int
    created_at: datetime


def _require_timezone(value: datetime) -> None:
    if value.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="scheduled_at must include a timezone.",
        )


def _owned_revision(
    db: Session, revision_id: int, student: Profile
) -> RevisionSchedule:
    revision = db.scalar(
        select(RevisionSchedule).where(
            RevisionSchedule.id == revision_id,
            RevisionSchedule.student_id == student.id,
        )
    )
    if revision is None:
        raise HTTPException(
            status_code=404,
            detail="Revision schedule not found.",
        )
    return revision


@router.post(
    "",
    response_model=RevisionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_revision(
    revision: RevisionCreate,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    require_student_owner(revision.student_id, current_student)
    _require_timezone(revision.scheduled_at)

    topic = db.get(Topic, revision.topic_id)
    if topic is None or topic.student_id != current_student.id:
        raise HTTPException(
            status_code=404,
            detail="Topic not found for this student.",
        )

    new_revision = RevisionSchedule(
        student_id=current_student.id,
        topic_id=topic.id,
        scheduled_at=revision.scheduled_at,
    )
    db.add(new_revision)
    db.commit()
    db.refresh(new_revision)
    return new_revision


@router.get("", response_model=list[RevisionResponse])
def list_revisions(
    student_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    if student_id <= 0:
        raise HTTPException(status_code=422, detail="student_id must be positive.")
    require_student_owner(student_id, current_student)

    statement = (
        select(RevisionSchedule)
        .where(RevisionSchedule.student_id == current_student.id)
        .order_by(RevisionSchedule.scheduled_at)
    )
    return db.scalars(statement).all()


@router.patch("/{revision_id}/complete", response_model=RevisionResponse)
def complete_revision(
    revision_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    revision = _owned_revision(db, revision_id, current_student)
    revision.completed = True
    db.commit()
    db.refresh(revision)
    return revision


@router.patch("/{revision_id}/reschedule", response_model=RevisionResponse)
def reschedule_revision(
    revision_id: int,
    body: RevisionReschedule,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    _require_timezone(body.scheduled_at)
    revision = _owned_revision(db, revision_id, current_student)

    if revision.completed:
        raise HTTPException(
            status_code=409,
            detail="A completed revision cannot be rescheduled.",
        )

    revision.scheduled_at = body.scheduled_at
    revision.rescheduled_count = (revision.rescheduled_count or 0) + 1
    db.commit()
    db.refresh(revision)
    return revision
