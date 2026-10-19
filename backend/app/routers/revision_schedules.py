
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


class RevisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    topic_id: int
    scheduled_at: datetime
    completed: bool
    created_at: datetime


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
    # Students can schedule revisions only for themselves.
    require_student_owner(revision.student_id, current_student)

    student = db.get(Profile, revision.student_id)

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    topic = db.get(Topic, revision.topic_id)

    if topic is None or topic.student_id != revision.student_id:
        raise HTTPException(
            status_code=404,
            detail="Topic not found for this student.",
        )

    if revision.scheduled_at.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="scheduled_at must include a timezone.",
        )

    new_revision = RevisionSchedule(
        student_id=revision.student_id,
        topic_id=revision.topic_id,
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
        raise HTTPException(
            status_code=422,
            detail="student_id must be positive.",
        )

    require_student_owner(student_id, current_student)

    statement = (
        select(RevisionSchedule)
        .where(RevisionSchedule.student_id == current_student.id)
        .order_by(RevisionSchedule.scheduled_at)
    )

    return db.scalars(statement).all()


@router.patch(
    "/{revision_id}/complete",
    response_model=RevisionResponse,
)
def complete_revision(
    revision_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    if revision_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="revision_id must be positive.",
        )

    # Fetch only a revision belonging to the authenticated student.
    statement = select(RevisionSchedule).where(
        RevisionSchedule.id == revision_id,
        RevisionSchedule.student_id == current_student.id,
    )

    revision = db.scalar(statement)

    if revision is None:
        raise HTTPException(
            status_code=404,
            detail="Revision schedule not found.",
        )

    revision.completed = True
    db.commit()
    db.refresh(revision)

    return revision
