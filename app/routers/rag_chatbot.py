"""RAG chatbot router: full-featured chatbot with standard/hybrid RAG toggle.

Provides endpoints for:
  - Chatting with uploaded materials using standard or hybrid RAG
  - Generating quizzes from materials with RAG mode selection
  - Inspecting the RAG pipeline architecture and retrieval metadata
  - Viewing retrieval diagnostics (per-chunk BM25 and semantic scores)
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.auth import get_current_student
from app.config import settings
from app.database import get_db
from app.models import (
    Profile,
    StudyRagInteraction,
    StudyMaterial,
    StudyMaterialChunk,
)
from app.services.rag_engine import (
    RagMode,
    get_pipeline_metadata,
    hybrid_retrieve,
    rerank_with_observed_state,
    select_study_intervention,
    tokenize,
)
from app.services.study_materials import (
    answer_from_evidence,
    concept_gaps,
    concept_recency_priority,
    diversify_evidence,
    evidence_text,
    generate_notes,
    generate_questions,
    source_ref,
)

router = APIRouter(prefix="/rag", tags=["RAG Chatbot"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RagChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: RagMode = Field(default="hybrid")
    history: list[dict] = Field(default_factory=list, max_length=8)
    show_diagnostics: bool = Field(default=False)


class RagQuizRequest(BaseModel):
    question_count: int = Field(default=5, ge=1, le=10)
    mode: RagMode = Field(default="hybrid")


class RagNotesRequest(BaseModel):
    mode: RagMode = Field(default="hybrid")


class RagCompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class RagFeedbackRequest(BaseModel):
    supported_by_sources: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _material_for_user(db: Session, material_id: int, profile: Profile) -> StudyMaterial:
    material = db.get(StudyMaterial, material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="Study material not found.")
    if material.faculty_id != profile.id and not material.published and profile.role != "admin":
        raise HTTPException(status_code=404, detail="Study material not found.")
    return material


def _all_chunks(db: Session, material_id: int) -> list[StudyMaterialChunk]:
    return list(
        db.scalars(
            select(StudyMaterialChunk)
            .where(StudyMaterialChunk.material_id == material_id)
            .order_by(StudyMaterialChunk.chunk_index)
        ).all()
    )


def _citations(evidence: list[dict]) -> list[dict]:
    return [source_ref(item["chunk"]) for item in evidence if item.get("chunk")]


def _answer_citations(evidence: list[dict], citation_ids: list[int]) -> list[dict]:
    """Return only cited source chunks present in the retrieved evidence."""
    by_id = {item["chunk"].id: item["chunk"] for item in evidence if item.get("chunk")}
    result = []
    seen = set()
    for chunk_id in citation_ids:
        if chunk_id in by_id and chunk_id not in seen:
            result.append(source_ref(by_id[chunk_id]))
            seen.add(chunk_id)
    return result


def _diagnostics(evidence: list[dict]) -> list[dict]:
    """Build per-chunk retrieval diagnostic information."""
    return [
        {
            "chunk_id": item["chunk"].id,
            "chunk_index": item["chunk"].chunk_index,
            "bm25_score": item.get("bm25_score", 0),
            "semantic_score": item.get("semantic_score", 0),
            "combined_score": round(item.get("base_score", 0), 4),
            "gap_boost": round(item.get("gap_boost", 0), 4),
            "recency_boost": round(item.get("recency_boost", 0), 4),
            "state_gap_relevance": round(item.get("state_gap_relevance", 0), 4),
            "state_recency_relevance": round(item.get("state_recency_relevance", 0), 4),
            "matched_state_concepts": item.get("matched_state_concepts", []),
            "final_score": round(item.get("score", 0), 4),
            "retrieval_method": item.get("retrieval_method", "unknown"),
            "excerpt": item["chunk"].content[:250],
        }
        for item in evidence
    ]


def _history_turns(interaction: StudyRagInteraction) -> list[dict]:
    created_at = interaction.created_at.isoformat() if interaction.created_at else None
    return [
        {
            "role": "user",
            "content": interaction.question,
            "interaction_id": interaction.id,
            "created_at": created_at,
            "mode": interaction.rag_mode,
        },
        {
            "role": "assistant",
            "content": interaction.answer,
            "interaction_id": interaction.id,
            "created_at": created_at,
            "mode": interaction.rag_mode,
            "retrieval_method": interaction.retrieval_method,
            "citations": interaction.citations or [],
            "supported_by_sources": interaction.supported_by_sources,
        },
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/architecture")
def get_architecture():
    """Return both standard and hybrid RAG pipeline architecture metadata."""
    return {
        "standard": get_pipeline_metadata("standard"),
        "hybrid": get_pipeline_metadata("hybrid"),
        "adaptive": get_pipeline_metadata("adaptive"),
        "comparison": {
            "standard": {
                "pros": [
                    "Fast — pure keyword matching with no vectorisation overhead",
                    "Transparent — easy to explain which terms matched",
                    "Low memory footprint",
                ],
                "cons": [
                    "Misses synonyms and paraphrased content",
                    "No semantic understanding of query intent",
                ],
            },
            "adaptive": {
                "pros": [
                    "Uses the learner's observed material quiz gaps and exposure recency",
                    "Reduces personalization boost for direct factual questions",
                    "Displays the state signals and heuristic action used",
                ],
                "cons": [
                    "Depends on repeated quiz history and concept-label quality",
                    "State weights and recency thresholds are not empirically tuned",
                ],
            },
            "hybrid": {
                "pros": [
                    "Catches both exact keyword matches AND semantic similarity",
                    "RRF fusion balances precision and recall",
                    "Better at understanding paraphrased questions",
                    "More robust retrieval for diverse query styles",
                ],
                "cons": [
                    "Slightly higher latency from TF-IDF vectorisation",
                    "More complex pipeline to debug",
                ],
            },
        },
    }


@router.get("/{material_id}/architecture")
def get_material_architecture(
    material_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Return RAG architecture metadata scoped to a specific material."""
    material = _material_for_user(db, material_id, current_student)
    chunk_count = db.scalar(
        select(func.count(StudyMaterialChunk.id)).where(
            StudyMaterialChunk.material_id == material.id
        )
    ) or 0
    gaps = concept_gaps(db, current_student.id, material.id)
    recency = concept_recency_priority(db, current_student.id, material.id)

    return {
        "material": {
            "id": material.id,
            "title": material.title,
            "chunk_count": chunk_count,
        },
        "recall_gaps": sorted(gaps, key=gaps.get, reverse=True)[:8],
        "recall_gap_count": len(gaps),
        "recently_quizzed_concept_count": len(recency),
        "state_source": "observed material-quiz responses; recency is elapsed time, not predicted forgetting risk",
        "architectures": {
            "standard": get_pipeline_metadata("standard"),
            "hybrid": get_pipeline_metadata("hybrid"),
            "adaptive": get_pipeline_metadata("adaptive"),
        },
    }


@router.get("/{material_id}/history")
def get_chat_history(
    material_id: int,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Return persisted chat turns for this learner and material."""
    material = _material_for_user(db, material_id, current_student)
    interactions = db.scalars(
        select(StudyRagInteraction)
        .where(
            StudyRagInteraction.material_id == material.id,
            StudyRagInteraction.student_id == current_student.id,
        )
        .order_by(StudyRagInteraction.created_at.desc(), StudyRagInteraction.id.desc())
        .limit(200)
    ).all()
    interactions = list(reversed(interactions))
    return {"material_id": material.id, "messages": [turn for row in interactions for turn in _history_turns(row)]}


@router.post("/{material_id}/interactions/{interaction_id}/feedback")
def rate_chat_answer(
    material_id: int,
    interaction_id: int,
    request: RagFeedbackRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Save the learner's explicit judgment of whether citations support an answer."""
    _material_for_user(db, material_id, current_student)
    interaction = db.scalar(
        select(StudyRagInteraction).where(
            StudyRagInteraction.id == interaction_id,
            StudyRagInteraction.material_id == material_id,
            StudyRagInteraction.student_id == current_student.id,
        )
    )
    if interaction is None:
        raise HTTPException(status_code=404, detail="Chat answer not found.")
    interaction.supported_by_sources = request.supported_by_sources
    db.commit()
    return {"interaction_id": interaction.id, "supported_by_sources": interaction.supported_by_sources}


@router.post("/{material_id}/compare")
def compare_retrieval_modes(
    material_id: int,
    request: RagCompareRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Run the same question through each retrieval strategy and report measured retrieval diagnostics."""
    material = _material_for_user(db, material_id, current_student)
    chunks = _all_chunks(db, material.id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexed passages found for this material.")

    gaps = concept_gaps(db, current_student.id, material.id)
    recency = concept_recency_priority(db, current_student.id, material.id)
    modes: list[RagMode] = ["standard", "hybrid", "adaptive"]
    comparisons = []
    for mode in modes:
        started = time.perf_counter()
        base_mode = "standard" if mode == "standard" else "hybrid"
        evidence = hybrid_retrieve(chunks, request.question, mode=base_mode, limit=len(chunks))
        intent = "direct_question"
        if mode == "adaptive":
            evidence, intent = rerank_with_observed_state(
                evidence,
                gaps,
                recency,
                request.question,
                gap_weight=settings.rag_state_gap_weight,
                recency_weight=settings.rag_state_recency_weight,
                direct_question_scale=settings.rag_direct_question_state_scale,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        top = evidence[:6]
        rated = db.execute(
            select(
                func.count(StudyRagInteraction.id),
                func.sum(case((StudyRagInteraction.supported_by_sources.is_(True), 1), else_=0)),
            ).where(
                StudyRagInteraction.student_id == current_student.id,
                StudyRagInteraction.material_id == material.id,
                StudyRagInteraction.rag_mode == mode,
                StudyRagInteraction.supported_by_sources.is_not(None),
            )
        ).one()
        rated_count = int(rated[0] or 0)
        supported_count = int(rated[1] or 0)
        comparisons.append({
            "mode": mode,
            "retrieval_method": top[0].get("retrieval_method", "none") if top else "none",
            "retrieval_latency_ms": round(elapsed_ms, 3),
            "retrieved_count": len(top),
            "top_rank_score": round(float(top[0].get("score", 0)), 4) if top else 0.0,
            "mean_top_k_rank_score": round(sum(float(item.get("score", 0)) for item in top) / len(top), 4) if top else 0.0,
            "rank_score_is_calibrated_confidence": False,
            "answer_accuracy": None,
            "accuracy_status": "Requires human-labeled answer ground truth; not measured by this comparison.",
            "source_support_votes": rated_count,
            "source_support_rate": round(supported_count / rated_count, 4) if rated_count else None,
            "source_support_status": "Learner feedback proxy; not objective answer accuracy." if rated_count else "No learner feedback collected for this mode yet.",
            "query_intent": intent,
            "top_passages": [
                {
                    "chunk_id": item["chunk"].id,
                    "chunk_index": item["chunk"].chunk_index,
                    "rank_score": round(float(item.get("score", 0)), 4),
                    "excerpt": item["chunk"].content[:240],
                }
                for item in top
            ],
        })

    return {
        "question": request.question,
        "material_id": material.id,
        "comparison_scope": "retrieval only; generation is not repeated for each strategy",
        "metrics_note": "Rank scores are query-relative diagnostics, not calibrated confidence. Accuracy requires a judged test set.",
        "modes": comparisons,
    }


@router.post("/{material_id}/chat")
def rag_chat(
    material_id: int,
    request: RagChatRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Chat with a material using standard or hybrid RAG retrieval."""
    material = _material_for_user(db, material_id, current_student)
    chunks = _all_chunks(db, material.id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexed passages found for this material.")

    gaps = concept_gaps(db, current_student.id, material.id)
    recency = concept_recency_priority(db, current_student.id, material.id)
    baseline_mode = "standard" if request.mode == "standard" else "hybrid"
    evidence = hybrid_retrieve(
        chunks,
        query=request.question,
        mode=baseline_mode,
        limit=len(chunks),
    )
    intent = "direct_question"
    if request.mode == "adaptive":
        evidence, intent = rerank_with_observed_state(
            evidence,
            gaps,
            recency,
            request.question,
            gap_weight=settings.rag_state_gap_weight,
            recency_weight=settings.rag_state_recency_weight,
            direct_question_scale=settings.rag_direct_question_state_scale,
        )
    evidence = evidence[:6]

    try:
        answer = answer_from_evidence(
            request.question,
            evidence,
            history=request.history[-8:] if request.history else None,
        )
    except RuntimeError as exc:
        logger.exception("RAG chat failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    citations = _answer_citations(evidence, answer.get("citation_ids", []))
    intervention = select_study_intervention(gaps, recency, intent) if request.mode == "adaptive" else None
    saved_diagnostics = _diagnostics(evidence)
    interaction = StudyRagInteraction(
        material_id=material.id,
        student_id=current_student.id,
        question=request.question,
        answer=answer["answer"],
        rag_mode=request.mode,
        retrieval_method=evidence[0].get("retrieval_method", "none") if evidence else "none",
        citations=citations,
        diagnostics=saved_diagnostics,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    response = {
        "interaction_id": interaction.id,
        "question": request.question,
        "answer": answer["answer"],
        "mode": answer["mode"],
        "rag_mode": request.mode,
        "retrieval_method": evidence[0]["retrieval_method"] if evidence else "none",
        "citations": citations,
        "recall_gaps": sorted(gaps, key=gaps.get, reverse=True)[:5] if request.mode == "adaptive" else [],
        "state_conditioning": {
            "enabled": request.mode == "adaptive",
            "signal": "observed material-quiz gaps and elapsed quiz time" if request.mode == "adaptive" else None,
            "query_intent": intent if request.mode == "adaptive" else None,
            "concepts_used": sorted({concept for item in evidence for concept in item.get("matched_state_concepts", [])}),
            "weights": {
                "gap": settings.rag_state_gap_weight,
                "recency": settings.rag_state_recency_weight,
                "direct_question_scale": settings.rag_direct_question_state_scale,
            } if request.mode == "adaptive" else None,
        },
        "intervention": intervention,
        "pipeline": get_pipeline_metadata(request.mode),
    }

    if request.show_diagnostics:
        response["diagnostics"] = saved_diagnostics

    return response


@router.post("/{material_id}/quiz")
def rag_quiz(
    material_id: int,
    request: RagQuizRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Generate a quiz from material using the chosen RAG mode."""
    material = _material_for_user(db, material_id, current_student)
    chunks = _all_chunks(db, material.id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexed passages found for this material.")

    gaps = concept_gaps(db, current_student.id, material.id)
    recency = concept_recency_priority(db, current_student.id, material.id)
    baseline_mode = "standard" if request.mode == "standard" else "hybrid"
    evidence = hybrid_retrieve(
        chunks,
        query="",
        mode=baseline_mode,
        limit=len(chunks),
    )
    intent = "revision" if request.mode == "adaptive" else "direct_question"
    if request.mode == "adaptive":
        evidence, intent = rerank_with_observed_state(
            evidence, gaps, recency, "create a revision quiz",
            gap_weight=settings.rag_state_gap_weight,
            recency_weight=settings.rag_state_recency_weight,
            direct_question_scale=settings.rag_direct_question_state_scale,
        )

    evidence = diversify_evidence(evidence, max(12, request.question_count * 2))
    intervention = select_study_intervention(gaps, recency, intent) if request.mode == "adaptive" else None

    if not evidence:
        raise HTTPException(status_code=422, detail="No searchable text is available in this material.")

    try:
        questions = generate_questions(evidence, request.question_count)
    except RuntimeError as exc:
        logger.exception("RAG quiz generation failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not questions:
        raise HTTPException(status_code=422, detail="Could not form questions from this material.")

    return {
        "rag_mode": request.mode,
        "retrieval_method": evidence[0]["retrieval_method"] if evidence else "none",
        "recall_gaps": sorted(gaps, key=gaps.get, reverse=True)[:5] if request.mode == "adaptive" else [],
        "intervention": intervention,
        "questions": [
            {
                "index": idx,
                "question": q["question"],
                "answer": q["answer"],
                "concept": q["concept"],
                "citation": source_ref(
                    next(
                        (item["chunk"] for item in evidence if item["chunk"].id == q["source_chunk_id"]),
                        evidence[0]["chunk"],
                    )
                ),
            }
            for idx, q in enumerate(questions)
        ],
        "pipeline": get_pipeline_metadata(request.mode),
    }


@router.post("/{material_id}/notes")
def rag_notes(
    material_id: int,
    request: RagNotesRequest,
    db: Session = Depends(get_db),
    current_student: Profile = Depends(get_current_student),
):
    """Generate study notes from material using the chosen RAG mode."""
    material = _material_for_user(db, material_id, current_student)
    chunks = _all_chunks(db, material.id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexed passages found for this material.")

    gaps = concept_gaps(db, current_student.id, material.id)
    recency = concept_recency_priority(db, current_student.id, material.id)
    baseline_mode = "standard" if request.mode == "standard" else "hybrid"
    evidence = hybrid_retrieve(
        chunks,
        query="",
        mode=baseline_mode,
        limit=len(chunks),
    )
    intent = "revision" if request.mode == "adaptive" else "direct_question"
    if request.mode == "adaptive":
        evidence, intent = rerank_with_observed_state(
            evidence, gaps, recency, "create revision notes",
            gap_weight=settings.rag_state_gap_weight,
            recency_weight=settings.rag_state_recency_weight,
            direct_question_scale=settings.rag_direct_question_state_scale,
        )

    # For notes with no query, fall back to evenly sampled chunks if hybrid returns nothing useful
    if not evidence or all(item["base_score"] == 0.0 for item in evidence):
        total = len(chunks)
        sample_count = min(total, 8)
        indexes = sorted({
            round(i * (total - 1) / max(1, sample_count - 1))
            for i in range(sample_count)
        })
        evidence = [
            {
                "chunk": chunks[idx],
                "score": 0.25,
                "gap_boost": 0.0,
                "base_score": 0.25,
                "retrieval_method": request.mode,
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
            for idx in indexes
        ]

    evidence = diversify_evidence(evidence, 8)

    try:
        notes = generate_notes(evidence)
    except RuntimeError as exc:
        logger.exception("RAG note generation failed for material %s", material_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not notes:
        raise HTTPException(status_code=422, detail="No notes could be extracted from this material.")

    return {
        "rag_mode": request.mode,
        "mode": "grounded_ai" if any(item.get("mode") != "extractive" for item in notes) else "extractive",
        "intervention": select_study_intervention(gaps, recency, intent) if request.mode == "adaptive" else None,
        "notes": [
            {
                **item,
                "citations": [
                    source_ref(ev["chunk"])
                    for ev in evidence
                    if ev["chunk"].id in item.get("citations", [])
                ],
            }
            for item in notes
        ],
        "state_conditioning": {
            "enabled": request.mode == "adaptive",
            "signal": "observed material-quiz gaps and elapsed quiz time" if request.mode == "adaptive" else None,
        },
        "pipeline": get_pipeline_metadata(request.mode),
    }
