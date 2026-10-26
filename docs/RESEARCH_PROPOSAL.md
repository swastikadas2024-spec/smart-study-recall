# Research proposal: RecallMind

## Research question

Does conditioning educational retrieval and tutoring interventions on a learner's observed knowledge state and estimated review risk improve next-question prediction, evidence relevance, and review efficiency compared with standard knowledge tracing, non-personalized RAG, and fixed-interval review?

This is a hypothesis. The repository currently has no evidence that the combined system improves learning outcomes.

## Proposed system

RecallMind combines (1) sequential prediction of a learner's next response, (2) a separate recall/retention baseline, (3) source-grounded retrieval over uploaded course materials, and (4) an explainable rule-based tutoring/revision policy. RAG can use the learner's repeated concept-level quiz errors as a weak-state signal to prioritize evidence. The source evidence remains visible and citations must point to retrieved passages.

## Hypotheses

1. A sequence model can predict held-out next interactions better than a non-sequential baseline under matched learner and temporal splits.
2. Gap-aware reranking can improve nDCG/Recall@k over standard retrieval on a human-judged query/passages set.
3. Gap-aware retrieval and tutoring may reduce review workload while maintaining or improving delayed quiz performance; this requires a consented, controlled learner study.

## Candidate contributions

- A reproducible comparison between sequence-based knowledge tracing and simpler baselines.
- A transparent interface between student-state estimates, evidence retrieval and explainable interventions.
- A task-matched evaluation protocol for course materials, which the public Duolingo trace alone cannot provide.

These are research directions, not established novelty or patent claims. Literature and patent review is required before asserting novelty.

## Scope and assumptions

The available Duolingo trace supports interaction-sequence prediction but is not the same as student performance on uploaded course concepts. App quiz correctness is the relevant future source for app-domain models, but there is not yet a sufficiently documented export/training corpus. No labels for misconception classes, passage relevance, or delayed retention are assumed.

## Evaluation plan

First evaluate next-response prediction on temporal and learner-held-out splits. Separately evaluate retrieval with human judgments and the adaptive policy with simulation only when labels support it. A later randomized pilot should compare adaptive and fixed review using identical material, assessments and time guidance. Require ethics review and voluntary informed consent before recruiting learners or disseminating participant data.
