"""Hybrid RAG engine: combines BM25 keyword search + semantic similarity retrieval.

Supports two modes:
  - "standard": BM25-only retrieval (the existing approach)
  - "hybrid":   BM25 + TF-IDF cosine similarity, merged via Reciprocal Rank Fusion (RRF)

The semantic component uses scikit-learn TF-IDF vectorisation as a lightweight,
dependency-free proxy for dense embeddings.  This avoids heavyweight model downloads
while still capturing meaningful term-frequency based semantic similarity.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import StudyMaterialChunk

logger = logging.getLogger(__name__)

RagMode = Literal["standard", "hybrid", "adaptive"]
REVISION_INTENT_TERMS = {
    "revise", "review", "recap", "practice", "quiz", "test", "remember",
    "forgot", "forgetting", "weak", "revision", "revisit",
}

# ---------------------------------------------------------------------------
# BM25 scorer (mirrors the existing retrieve_chunks logic)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "been", "being",
    "between", "both", "could", "does", "each", "from", "have", "into",
    "more", "most", "other", "over", "same", "should", "some", "such",
    "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "under", "until", "very", "what", "when",
    "where", "which", "while", "with", "would", "your", "and", "for",
    "the", "are", "was", "were", "but", "not", "you", "can", "has", "had",
    "its", "our", "use", "using", "used", "all", "any", "how", "why",
}

import re
TOKEN_RE = re.compile(r"[^\W_][\w'-]*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [
        token.lower().strip("'-")
        for token in TOKEN_RE.findall(text)
        if len(token.strip("'-")) > 2 and token.lower() not in STOPWORDS
    ]


def _bm25_scores(
    docs: list[list[str]],
    query_terms: set[str],
    k1: float = 2.5,
    b: float = 0.75,
) -> list[float]:
    """Return raw BM25 scores for each document given the query terms."""
    if not docs or not query_terms:
        return [0.0] * len(docs)

    document_frequency: Counter[str] = Counter(
        term for terms in docs for term in set(terms)
    )
    average_length = sum(map(len, docs)) / max(1, len(docs))
    n = len(docs)

    scores: list[float] = []
    for terms in docs:
        freq = Counter(terms)
        score = 0.0
        for term in query_terms:
            if term not in freq:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = freq[term]
            denom = tf + k1 * (1 - b + b * len(terms) / max(1, average_length))
            score += idf * tf * k1 / denom
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Semantic (TF-IDF cosine) scorer
# ---------------------------------------------------------------------------

def _semantic_scores(
    chunk_texts: list[str],
    query: str,
) -> list[float]:
    """Return cosine similarity scores between query and each chunk via TF-IDF."""
    if not chunk_texts or not query.strip():
        return [0.0] * len(chunk_texts)

    corpus = chunk_texts + [query]
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(chunk_texts)

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vec, doc_vecs).flatten().tolist()
    return similarities


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists of document indices using RRF.

    Args:
        ranked_lists: Each inner list is document indices sorted by relevance (best first).
        k: RRF constant (default 60).

    Returns:
        List of (doc_index, rrf_score) sorted by descending RRF score.
    """
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    chunks: list[StudyMaterialChunk],
    query: str,
    mode: RagMode = "hybrid",
    limit: int = 6,
    gap_terms: set[str] | None = None,
    gaps: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a query.

    Args:
        chunks: All chunks for a material.
        query: User's question / search text.
        mode: "standard" for BM25-only; "hybrid" or "adaptive" use BM25 + TF-IDF + RRF.
        limit: Max chunks to return.
        gap_terms: Extra query terms from recall gap analysis.
        gaps: Concept gap map for boosting.

    Returns:
        Scored evidence list with keys: chunk, score, gap_boost, base_score,
        retrieval_method, bm25_score, semantic_score.
    """
    if not chunks:
        return []

    gaps = gaps or {}
    gap_terms = gap_terms or set()
    query_terms = set(tokenize(query)) | gap_terms

    chunk_texts = [chunk.content for chunk in chunks]
    docs = [tokenize(text) for text in chunk_texts]

    # --- BM25 scoring ---
    bm25_raw = _bm25_scores(docs, query_terms)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_raw]

    if mode == "standard":
        if query.strip() and max_bm25 <= 0:
            return []
        # Standard mode: BM25 only, same as existing retrieval
        ranked = []
        for i, chunk in enumerate(chunks):
            termset = set(docs[i])
            base_score = bm25_norm[i]
            gap_boost = 0.0
            for concept, gap in gaps.items():
                concept_terms = set(tokenize(concept))
                if concept_terms and termset.intersection(concept_terms):
                    gap_boost = max(gap_boost, 0.35 * gap)
            ranked.append({
                "chunk": chunk,
                "score": base_score + gap_boost,
                "gap_boost": gap_boost,
                "base_score": base_score,
                "retrieval_method": "bm25",
                "bm25_score": round(bm25_norm[i], 4),
                "semantic_score": 0.0,
            })
        ranked.sort(key=lambda item: (-item["score"], item["chunk"].chunk_index))
        return ranked[:limit]

    # --- Hybrid mode: BM25 + Semantic + RRF ---
    sem_raw = _semantic_scores(chunk_texts, query)
    max_sem = max(sem_raw) if sem_raw else 0.0
    if query.strip() and max_bm25 <= 0 and max_sem <= 0:
        return []
    sem_norm = [s / max_sem if max_sem > 0 else 0.0 for s in sem_raw]

    # Build ranked lists for RRF
    bm25_ranked = sorted(range(len(chunks)), key=lambda i: -bm25_raw[i])
    sem_ranked = sorted(range(len(chunks)), key=lambda i: -sem_raw[i])

    rrf_results = reciprocal_rank_fusion([bm25_ranked, sem_ranked])
    max_rrf = rrf_results[0][1] if rrf_results else 1.0

    # Build output with full metadata
    ranked = []
    for doc_idx, rrf_score in rrf_results:
        chunk = chunks[doc_idx]
        termset = set(docs[doc_idx])
        base_score = rrf_score / max_rrf  # normalise to [0, 1]
        gap_boost = 0.0
        for concept, gap in gaps.items():
            concept_terms = set(tokenize(concept))
            if concept_terms and termset.intersection(concept_terms):
                gap_boost = max(gap_boost, 0.35 * gap)

        ranked.append({
            "chunk": chunk,
            "score": base_score + gap_boost,
            "gap_boost": gap_boost,
            "base_score": base_score,
            "retrieval_method": "hybrid_rrf",
            "bm25_score": round(bm25_norm[doc_idx], 4),
            "semantic_score": round(sem_norm[doc_idx], 4),
        })

    ranked.sort(key=lambda item: (-item["score"], item["chunk"].chunk_index))
    return ranked[:limit]


def rerank_with_observed_state(
    evidence: list[dict[str, Any]],
    gaps: dict[str, float],
    recency_priority: dict[str, float],
    query: str,
    *,
    gap_weight: float = 0.30,
    recency_weight: float = 0.15,
    direct_question_scale: float = 0.25,
) -> tuple[list[dict[str, Any]], str]:
    """Apply a transparent concept-gap and exposure-recency reranker.

    This is a deterministic baseline over observed material-quiz history, not
    a learned model or an estimated forgetting probability.
    """
    query_terms = set(tokenize(query))
    revision_intent = bool(query_terms & REVISION_INTENT_TERMS)
    intent = "revision" if revision_intent else "direct_question"
    state_scale = 1.0 if revision_intent else direct_question_scale
    ranked = []
    for item in evidence:
        tokens = set(tokenize(item["chunk"].content))
        gap_relevance = 0.0
        recency_relevance = 0.0
        matched_concepts = []
        for concept in set(gaps) | set(recency_priority):
            concept_terms = set(tokenize(concept))
            if not concept_terms:
                continue
            overlap = len(tokens & concept_terms) / len(concept_terms)
            if overlap <= 0:
                continue
            gap_component = min(1.0, max(0.0, float(gaps.get(concept, 0.0)) / 0.8)) * overlap
            recency_component = min(1.0, max(0.0, float(recency_priority.get(concept, 0.0)))) * overlap
            gap_relevance = max(gap_relevance, gap_component)
            recency_relevance = max(recency_relevance, recency_component)
            matched_concepts.append(concept)
        gap_boost = gap_weight * state_scale * gap_relevance
        recency_boost = recency_weight * state_scale * recency_relevance
        reranked = {
            **item,
            "gap_boost": gap_boost,
            "recency_boost": recency_boost,
            "state_gap_relevance": gap_relevance,
            "state_recency_relevance": recency_relevance,
            "matched_state_concepts": sorted(matched_concepts),
            "score": float(item.get("base_score", item.get("score", 0.0))) + gap_boost + recency_boost,
        }
        ranked.append(reranked)
    ranked.sort(key=lambda item: (-item["score"], item["chunk"].chunk_index))
    return ranked, intent


def select_study_intervention(
    gaps: dict[str, float], recency_priority: dict[str, float], intent: str
) -> dict[str, str]:
    """Select an explainable heuristic action from observed quiz state."""
    if gaps:
        concept, value = max(gaps.items(), key=lambda pair: pair[1])
        if value >= 0.20:
            return {
                "action": "targeted_practice",
                "reason": f"Repeated quiz results show a gap for {concept}; review its cited passage and try a check question.",
                "basis": "observed quiz errors; heuristic threshold",
            }
    if recency_priority:
        concept, value = max(recency_priority.items(), key=lambda pair: pair[1])
        if value >= 0.80:
            return {
                "action": "retrieval_practice",
                "reason": f"It has been at least 24 days since a quiz response for {concept}; try recalling it before rereading.",
                "basis": "elapsed time since material quiz; heuristic threshold, not forgetting probability",
            }
    if intent == "revision":
        return {
            "action": "self_test",
            "reason": "This request is about review; try a short recall question before reading the explanation.",
            "basis": "request intent heuristic",
        }
    if not gaps and not recency_priority:
        return {
            "action": "build_baseline",
            "reason": "There is not enough quiz history for personalization yet; answer a few material questions to establish a baseline.",
            "basis": "no observed material quiz history",
        }
    return {
        "action": "cited_explanation",
        "reason": "No repeated low-score or long-gap signal crossed the current heuristic thresholds.",
        "basis": "observed quiz state; heuristic thresholds",
    }


def get_pipeline_metadata(mode: RagMode) -> dict[str, Any]:
    """Return metadata describing the RAG pipeline configuration for the given mode."""
    base = {
        "mode": mode,
        "chunking": {
            "strategy": "sliding_window",
            "window_size": 210,
            "overlap": 35,
            "unit": "words",
        },
        "generation": {
            "model": "configurable (AI_MODEL env var)",
            "grounding": "source_citations_enforced",
            "max_output_tokens": 2500,
        },
    }
    if mode == "standard":
        base["retrieval"] = {
            "method": "bm25",
            "description": "Keyword-based retrieval using BM25 (Okapi) scoring",
            "parameters": {"k1": 2.5, "b": 0.75},
            "ranking": "BM25 score only (unpersonalized retrieval baseline)",
        }
        base["pipeline_steps"] = [
            {"step": 1, "name": "Document Upload", "detail": "PDF/DOCX/TXT/MD → raw text extraction"},
            {"step": 2, "name": "Chunking", "detail": "Sliding window (210 words, 35 overlap) + keyword extraction"},
            {"step": 3, "name": "Keyword Index", "detail": "BM25 inverted index built at ingestion time"},
            {"step": 4, "name": "Query", "detail": "User question tokenised into search terms"},
            {"step": 5, "name": "BM25 Search", "detail": "Rank chunks by BM25 (Okapi) score against query terms"},
            {"step": 6, "name": "Select Evidence", "detail": "Return the highest BM25-ranked source chunks"},
            {"step": 7, "name": "Generate", "detail": "Top-K chunks sent to LLM with citation instructions"},
            {"step": 8, "name": "Answer", "detail": "Grounded response with source chunk citations"},
        ]
    elif mode == "hybrid":
        base["retrieval"] = {
            "methods": [
                {
                    "name": "bm25_keyword_search",
                    "description": "Keyword-based retrieval using BM25 (Okapi) scoring",
                    "parameters": {"k1": 2.5, "b": 0.75},
                },
                {
                    "name": "tfidf_semantic_search",
                    "description": "TF-IDF vectorisation + cosine similarity for meaning-based retrieval",
                    "parameters": {"max_features": 8000, "ngram_range": [1, 2], "sublinear_tf": True},
                },
            ],
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "description": "Merges BM25 and semantic ranked lists using RRF",
                "parameter_k": 60,
            },
            "ranking": "RRF score only (unpersonalized hybrid retrieval baseline)",
        }
        base["pipeline_steps"] = [
            {"step": 1, "name": "Document Upload", "detail": "PDF/DOCX/TXT/MD → raw text extraction"},
            {"step": 2, "name": "Chunking", "detail": "Sliding window (210 words, 35 overlap) + keyword extraction"},
            {"step": 3, "name": "Dual Indexing", "detail": "BM25 keyword index + TF-IDF vector index built at ingestion"},
            {"step": 4, "name": "Query", "detail": "User question processed for both keyword and semantic search"},
            {"step": 5, "name": "BM25 Keyword Search", "detail": "Rank chunks by BM25 (Okapi) term-frequency score"},
            {"step": 6, "name": "Semantic Search", "detail": "Rank chunks by TF-IDF cosine similarity (meaning-based)"},
            {"step": 7, "name": "Rank Fusion (RRF)", "detail": "Reciprocal Rank Fusion merges both ranked lists → top K"},
            {"step": 8, "name": "Select Evidence", "detail": "Return the highest RRF-ranked source chunks"},
            {"step": 9, "name": "Generate", "detail": "Top-K chunks sent to LLM with citation instructions"},
            {"step": 10, "name": "Answer", "detail": "Grounded response with source chunk citations + retrieval scores"},
        ]
    else:
        base["retrieval"] = {
            "methods": ["BM25 keyword ranking", "TF-IDF cosine ranking"],
            "fusion": {"method": "reciprocal_rank_fusion", "parameter_k": 60},
            "state_features": ["repeated material-quiz concept gaps", "days since the concept was last quizzed"],
            "state_scoring": "RRF relevance + configurable weighted concept-gap and exposure-recency boosts",
            "intent_rule": "full state boost for explicit review/practice intent; reduced boost for direct questions",
            "status": "deterministic heuristic; not a trained reranker or forgetting-risk model",
        }
        base["pipeline_steps"] = [
            {"step": 1, "name": "Document Upload", "detail": "PDF/DOCX/TXT/MD → raw text extraction"},
            {"step": 2, "name": "Chunking", "detail": "Sliding-window passages with source identifiers"},
            {"step": 3, "name": "Hybrid Retrieval", "detail": "BM25 + TF-IDF ranked with reciprocal-rank fusion"},
            {"step": 4, "name": "Learner State", "detail": "Read repeated concept quiz outcomes and last exposure time"},
            {"step": 5, "name": "Intent-Aware Reranking", "detail": "Apply configurable gap/recency boosts based on the request"},
            {"step": 6, "name": "Generate", "detail": "Answer from selected source passages with chunk citations"},
        ]
    return base
