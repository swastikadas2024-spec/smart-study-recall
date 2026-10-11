
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Profile

router = APIRouter(prefix="/students", tags=["Students"])


class StudentCreate(BaseModel):
    student_name: str = Field(min_length=1, max_length=100)
    email: EmailStr | None = None


class StudentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_name: str
    email: str | None
    created_at: object


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db),
):
    new_student = Profile(
        student_name=student.student_name.strip(),
        email=str(student.email) if student.email else None,
    )

    if not new_student.student_name:
        raise HTTPException(
            status_code=422,
            detail="Student name cannot be empty.",
        )

    db.add(new_student)

    try:
        db.commit()
        db.refresh(new_student)
        return new_student
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A student with this email may already exist.",
        )


@router.get("", response_model=list[StudentResponse])
def list_students(db: Session = Depends(get_db)):
    statement = select(Profile).order_by(Profile.id)
    return db.scalars(statement).all()
