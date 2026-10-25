
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import get_current_student
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
    role: str
    created_at: object


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    # An authenticated user cannot create a profile for another email.
    if student.email and str(student.email).lower() != (
        current_student.email or ""
    ).lower():
        raise HTTPException(
            status_code=403,
            detail="You can only use your own account email.",
        )

    # Prevent creating another profile for the same authenticated user.
    existing_profile = db.scalar(
        select(Profile).where(
            Profile.supabase_user_id == current_student.supabase_user_id
        )
    )

    if existing_profile is not None:
        raise HTTPException(
            status_code=409,
            detail="Your student profile already exists.",
        )

    new_student = Profile(
        student_name=student.student_name.strip(),
        email=(
            str(student.email)
            if student.email
            else current_student.email
        ),
        supabase_user_id=current_student.supabase_user_id,
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
def list_students(
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    statement = select(Profile).where(
        Profile.id == current_student.id
    )
    return db.scalars(statement).all()


@router.get("/me", response_model=StudentResponse)
def get_my_profile(
    current_student: Profile = Depends(get_current_student),
):
    return current_student
