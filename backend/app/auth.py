
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import httpx

from app.config import settings
from app.database import get_db
from app.models import Profile

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_student(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> Profile:
    """Verify the Supabase token and resolve the user's student profile."""

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase authentication is not configured.",
        )

    try:
        response = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {credentials.credentials}",
            },
            timeout=10.0,
        )

        if response.status_code in (400, 401, 403):
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired access token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        response.raise_for_status()
        user = response.json()

    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not verify Supabase authentication.",
        ) from exc

    supabase_user_id = user.get("id")
    if not supabase_user_id:
        raise HTTPException(status_code=401, detail="Invalid user identity.")

    # A previously linked user always resolves to the same profile.
    profile = db.scalar(
        select(Profile).where(
            Profile.supabase_user_id == supabase_user_id
        )
    )
    if profile is not None:
        return profile

    # Do not silently link an existing legacy profile by email.
    email = user.get("email")
    if email:
        existing_email_profile = db.scalar(
            select(Profile).where(Profile.email == email)
        )
        if existing_email_profile is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An existing student profile uses this email but is "
                    "not linked to this Supabase account. A trusted "
                    "account-linking step is required."
                ),
            )

    metadata = user.get("user_metadata") or {}
    student_name = (
        metadata.get("full_name")
        or metadata.get("name")
        or (email.split("@")[0] if email else "Student")
    )
    student_name = str(student_name).strip()[:100] or "Student"

    profile = Profile(
        supabase_user_id=supabase_user_id,
        student_name=student_name,
        email=email,
    )

    try:
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile
    except IntegrityError:
        db.rollback()

        # Another simultaneous request may have created the same profile.
        profile = db.scalar(
            select(Profile).where(
                Profile.supabase_user_id == supabase_user_id
            )
        )
        if profile is not None:
            return profile

        raise HTTPException(
            status_code=409,
            detail=(
                "Could not create a student profile. "
                "Check for an existing email or account link."
            ),
        )
    
def require_student_owner(
    requested_student_id: int,
    current_student: Profile,
) -> None:
    """Prevent one authenticated student from accessing another's data."""

    if requested_student_id != current_student.id:
        raise HTTPException(
            status_code=404,
            detail="Student resource not found.",
        )


