"""Faculty material library and source-grounded RAG study tools."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth import get_current_student
from app.database import get_db
from app.models import (
    Profile,
    StudyMaterial,
    StudyMaterialChunk,
    StudyMaterialNote,
    StudyQuizResponse,
    StudyQuizSession,
)
from app.services.study_materials import (
    answer_from_evidence,
    concept_gaps,
    diversify_evidence,
    extract_text,
    generate_notes,
    generate_questions,
    make_chunks,
    retrieve_chunks,
    source_ref,
)

router = APIRouter(prefix="/materials", tags=["AI Study Materials"])
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
logger = logging.getLogger(__name__)


class QuizGenerateRequest(BaseModel):
    question_count: int = Field(default=5, ge=1, le=10)


class QuizSubmitRequest(BaseModel):
    answers: list[str] = Field(min_length=1, max_length=10)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


def _material_for_user(db: Session, material_id: int, profile: Profile) -> StudyMaterial:
    material = db.get(StudyMaterial, material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="Study material not found.")
    if material.faculty_id != profile.id and not material.published and profile.role != "admin":
        raise HTTPException(status_code=404, detail="Study material not found.")
    return material


def _material_json(db: Session, material: StudyMaterial, profile: Profile) -> dict:
    chunks = db.scalars(
        select(StudyMaterialChunk.id).where(StudyMaterialChunk.material_id == material.id)
    ).all()
    return {
        "id": material.id,
        "title": material.title,
        "subject": material.subject,
        "filename": material.original_filename,
        "content_type": material.content_type,
        "published": material.published,
        "chunk_count": len(chunks),
        "created_at": material.created_at,
        "can_manage": material.faculty_id == profile.id or profile.role == "admin",
    }


def _citations(db: Session, citation_ids: list[int], evidence: list[dict]) -> list[dict]:
    chunks = {item["chunk"].id: item["chunk"] for item in evidence}
    return [source_ref(chunks[chunk_id]) for chunk_id in dict.fromkeys(citation_ids) if chunk_id in chunks]


@router.get("")
def list_materials(
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    # Any signed-in user can contribute. Everyone sees published materials;
    # each uploader also sees their own private materials.
    statement = select(StudyMaterial).where(
        or_(StudyMaterial.published.is_(True), StudyMaterial.faculty_id == current_student.id)
    )
    materials = db.scalars(statement.order_by(StudyMaterial.created_at.desc())).all()
    return [_material_json(db, material, current_student) for material in materials]


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_material(
    title: str = Form(min_length=1, max_length=200),
    subject: str | None = Form(default=None, max_length=100),
    published: bool = Form(default=True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    clean_title = title.strip()
    if not clean_title:
        raise HTTPException(status_code=422, detail="Give the material a title.")
    filename = (file.filename or "study-material").replace("\\", "/").split("/")[-1][:255]
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 12 MB upload limit.")
    digest = hashlib.sha256(payload).hexdigest()
    duplicate = db.scalar(
        select(StudyMaterial).where(
            StudyMaterial.faculty_id == current_student.id,
            StudyMaterial.content_sha256 == digest,
        )
    )
    if duplicate is not None:
        raise HTTPException(status_code=409, detail="You have already uploaded this file.")
    try:
        extracted = extract_text(filename, payload)
    except Exception as exc:
        # Parser errors vary by format; return a safe, actionable message.
        message = str(exc) if isinstance(exc, ValueError) else "The file could not be read. Check that it is a valid PDF, DOCX, TXT, or Markdown file."
        raise HTTPException(status_code=422, detail=message) from exc
    chunks = make_chunks(extracted)
    if not chunks:
        raise HTTPException(status_code=422, detail="The file did not contain usable study material.")

    material = StudyMaterial(
        faculty_id=current_student.id,
        title=clean_title,
        subject=(subject or "").strip() or None,
        original_filename=filename,
        content_type=file.content_type or "application/octet-stream",
        content_sha256=digest,
        published=published,
    )
    db.add(material)
    db.flush()
    for index, chunk in enumerate(chunks):
        db.add(StudyMaterialChunk(material_id=material.id, chunk_index=index, **chunk))
    db.commit()
    db.refresh(material)
    return _material_json(db, material, current_student)


@router.delete("/{material_id}", status_code=204)
def delete_material(
    material_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    material = _material_for_user(db, material_id, current_student)
    if material.faculty_id != current_student.id and current_student.role != "admin":
        raise HTTPException(status_code=403, detail="Only the uploading faculty member can remove this material.")
    db.delete(material)
    db.commit()


@router.post("/{material_id}/notes")
def create_notes(
    material_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    material = _material_for_user(db, material_id, current_student)
    evidence = diversify_evidence(
        retrieve_chunks(db, material, current_student.id, limit=8), 8
    )
    try:
        notes = generate_notes(evidence)
    except RuntimeError as exc:
        logger.exception("AI note generation failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not notes:
        raise HTTPException(status_code=422, detail="No notes could be extracted from this material.")
    note = StudyMaterialNote(material_id=material.id, student_id=current_student.id, notes=notes)
    db.add(note)
    db.commit()
    return {
        "id": note.id,
        "material_id": material.id,
        "mode": "grounded_ai" if any(item.get("mode") != "extractive" for item in notes) else "extractive",
        "notes": [
            {**item, "citations": _citations(db, item["citations"], evidence)}
            for item in notes
        ],
    }


@router.post("/{material_id}/quiz")
def create_quiz(
    material_id: int,
    request: QuizGenerateRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    material = _material_for_user(db, material_id, current_student)
    gaps = concept_gaps(db, current_student.id, material.id)
    evidence = diversify_evidence(
        retrieve_chunks(
            db,
            material,
            current_student.id,
            query=" ".join(gaps),
            limit=max(12, request.question_count * 2),
        ),
        max(12, request.question_count * 2),
    )
    if not evidence:
        raise HTTPException(status_code=422, detail="No searchable text is available in this material.")
    try:
        questions = generate_questions(evidence, request.question_count)
    except RuntimeError as exc:
        logger.exception("AI quiz generation failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not questions:
        raise HTTPException(status_code=422, detail="Could not form questions from this material. Try a different document.")
    session = StudyQuizSession(
        material_id=material.id,
        student_id=current_student.id,
        questions=questions,
    )
    db.add(session)
    db.commit()
    return {
        "session_id": session.id,
        "mode": "recall_gap_guided" if gaps else "grounded_ai_or_extract",
        "recall_gaps": sorted(gaps, key=gaps.get, reverse=True)[:5],
        "questions": [
            {
                "index": index,
                "question": question["question"],
                "concept": question["concept"],
                "citation": _citations(db, [question["source_chunk_id"]], evidence)[0],
            }
            for index, question in enumerate(questions)
        ],
    }


def _normalize_answer(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


@router.post("/{material_id}/quiz/{session_id}/submit")
def submit_quiz(
    material_id: int,
    session_id: int,
    request: QuizSubmitRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    _material_for_user(db, material_id, current_student)
    session = db.scalar(
        select(StudyQuizSession).where(
            StudyQuizSession.id == session_id,
            StudyQuizSession.material_id == material_id,
            StudyQuizSession.student_id == current_student.id,
        )
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Quiz session not found.")
    if session.submitted_at is not None:
        raise HTTPException(status_code=409, detail="This quiz has already been submitted.")
    if len(request.answers) != len(session.questions):
        raise HTTPException(status_code=422, detail="Submit one answer for every question.")

    results = []
    correct_count = 0
    for index, (question, response) in enumerate(zip(session.questions, request.answers)):
        expected = _normalize_answer(question["answer"])
        submitted = _normalize_answer(response)
        correct = bool(submitted) and (submitted == expected or (len(expected) > 3 and expected in submitted))
        correct_count += int(correct)
        db.add(
            StudyQuizResponse(
                session_id=session.id,
                material_id=material_id,
                student_id=current_student.id,
                concept=question["concept"],
                is_correct=correct,
            )
        )
        results.append(
            {
                "index": index,
                "correct": correct,
                "expected_answer": question["answer"],
                "concept": question["concept"],
                "source_chunk_id": question["source_chunk_id"],
            }
        )
    session.score = round(100 * correct_count / len(session.questions))
    session.submitted_at = datetime.now(timezone.utc)
    db.commit()
    gaps = concept_gaps(db, current_student.id, material_id)
    return {
        "score": session.score,
        "correct": correct_count,
        "total": len(session.questions),
        "results": results,
        "recall_gaps": sorted(gaps, key=gaps.get, reverse=True)[:8],
    }


@router.post("/{material_id}/chat")
def chat_with_material(
    material_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    material = _material_for_user(db, material_id, current_student)
    evidence = retrieve_chunks(db, material, current_student.id, query=request.question, limit=5)
    try:
        answer = answer_from_evidence(
            request.question,
            evidence,
            history=[turn.model_dump() for turn in request.history],
        )
    except RuntimeError as exc:
        logger.exception("AI chat response failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    citations = _citations(db, answer["citation_ids"], evidence)
    gaps = concept_gaps(db, current_student.id, material_id)
    return {
        "question": request.question,
        "answer": answer["answer"],
        "mode": answer["mode"],
        "citations": citations,
        "recall_gap_context": sorted(gaps, key=gaps.get, reverse=True)[:5],
        "recall_gap_boost_applied": any(item["gap_boost"] > 0 for item in evidence),
    }
