from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    JSON,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Profile(Base):
    __tablename__ = "profiles"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True
    )
    supabase_user_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        nullable=True,
    )
    student_name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    email: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="student", server_default="student"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("public.profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic_name: Mapped[str] = mapped_column(
        String(200), nullable=False
    )
    subject: Mapped[str | None] = mapped_column(
        String(60), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LearningEvent(Base):
    __tablename__ = "learning_events"
    __table_args__ = (
        CheckConstraint("questions_seen >= 0"),
        CheckConstraint(
            "questions_correct >= 0 AND "
            "questions_correct <= questions_seen"
        ),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("public.profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("public.topics.id", ondelete="SET NULL"),
        nullable=True,
    )
    questions_seen: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    questions_correct: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RevisionSchedule(Base):
    __tablename__ = "revision_schedules"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("public.profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("public.topics.id", ondelete="CASCADE"),
        nullable=False,
    )
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    rescheduled_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StudyMaterial(Base):
    __tablename__ = "study_materials"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    faculty_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.profiles.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StudyMaterialChunk(Base):
    __tablename__ = "study_material_chunks"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    material_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.study_materials.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)


class StudyMaterialNote(Base):
    __tablename__ = "study_material_notes"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    material_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.study_materials.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.profiles.id", ondelete="CASCADE"), nullable=False
    )
    notes: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StudyQuizSession(Base):
    __tablename__ = "study_quiz_sessions"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    material_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.study_materials.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.profiles.id", ondelete="CASCADE"), nullable=False
    )
    questions: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StudyQuizResponse(Base):
    __tablename__ = "study_quiz_responses"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.study_quiz_sessions.id", ondelete="CASCADE"), nullable=False
    )
    material_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.study_materials.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.profiles.id", ondelete="CASCADE"), nullable=False
    )
    concept: Mapped[str] = mapped_column(String(120), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
