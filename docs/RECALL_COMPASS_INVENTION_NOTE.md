# Recall Compass: invention working note

This is an engineering disclosure draft, not a patentability or freedom-to-operate opinion. Keep it private until the team has discussed filing timing with a patent professional.

## Working title

Learner-state-weighted evidence retrieval for source-grounded study generation.

## Technical problem

Conventional document chat retrieves passages mainly from the wording of the current query. That can repeatedly surface already-mastered material while missing passages tied to concepts a learner has repeatedly answered incorrectly.

## Implemented mechanism

1. Faculty material is normalized into overlapping text chunks. Each chunk stores searchable text and a small set of extracted keywords.
2. A short-answer quiz is generated from retrieved chunks. The server stores the expected answers and source chunk IDs, then grades submitted answers server-side.
3. Each graded response updates a learner/material/concept performance record. A concept is treated as a recall gap after at least two responses and below 80% accuracy. The gap weight is `max(0, 0.8 - accuracy) × min(attempts / 3, 1)`.
4. Retrieval computes a normalized BM25 score for the current question and adds up to `0.35 × gap weight` when a candidate passage's indexed keywords overlap a weak concept.
5. Notes, quizzes, and chat receive the ranked excerpts. Generated answers may cite only chunk IDs in that retrieved set; unsupported chat questions receive an explicit no-source answer.
6. Notes and quiz contexts mix top gap matches with samples spread across the document, so a long lecture is not reduced to its opening pages.

The intended technical effect is a retrieval order that adapts to repeated learner errors while retaining query relevance and document coverage. That effect remains a hypothesis until measured against suitable baselines.

## Evidence to collect before any filing

- Compare BM25-only retrieval against BM25 plus concept-gap weighting on the same questions and documents.
- Measure recall@k for previously missed concepts, answer correctness, citation support, repeated-miss rate, and model-token use.
- Include ablations for the confidence threshold, gap formula, overlap weight, and document-spread selection.
- Preserve dated design notes, source code history, prompt versions, test corpora with permission, and inventor contributions.

## Preliminary prior-art leads

- [US 11,348,476 B2](https://patents.google.com/patent/US11348476B2/en) describes extracting knowledge items from digital source material, using user data to filter learning items, and generating structured learning assets such as questions.
- [US 10,861,344 B2](https://patents.google.com/patent/US10861344B2/en) describes personalized learning assets and conversational interfaces.
- [US 2026/0120585 A1](https://patents.google.com/patent/US20260120585A1/en) describes personalized Q&A using instructor-material retrieval and student-performance/past-query retrieval, with quiz assessment and learner-state updates.
- [US 2025/0095507 A1](https://patents.google.com/patent/US20250095507A1/en) describes a vocabulary assessment system combining RAG over educational content with weakness detection and adaptive content.

These references are close enough that broad claims to “RAG for education,” automatic notes/quizzes, learner personalization, or using weak areas to adapt study content are unlikely to distinguish the project by themselves. A narrower technical claim around the exact retrieval/assessment feedback loop would still need a professional claim-by-claim search and analysis.

For an India filing, assess the claim against section 3(k) of the Patents Act and the current [IP India 2025 computer-related invention guidelines](https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf). IP India also summarizes novelty, inventive step, industrial applicability, and the section 3 exclusions in its [patent basics](https://ipindia.gov.in/basics-of-patents). A code-level improvement and measured technical effect would be stronger evidence than a product feature list, but do not establish patentability by themselves.
