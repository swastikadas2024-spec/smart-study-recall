from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Uuid
    UniqueConstraint,
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
    created_at: Mapped[DateTime] = mapped_column(
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
    created_at: Mapped[DateTime] = mapped_column(
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
    event_time: Mapped[DateTime] = mapped_column(
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
    scheduled_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )