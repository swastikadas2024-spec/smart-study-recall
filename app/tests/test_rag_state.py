import unittest
from types import SimpleNamespace

from app.services.rag_engine import (
    get_pipeline_metadata,
    hybrid_retrieve,
    rerank_with_observed_state,
    select_study_intervention,
)
from app.routers.rag_chatbot import _answer_citations


class RecallAwareRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            SimpleNamespace(id=1, chunk_index=0, content="Photosynthesis uses chlorophyll and light energy."),
            SimpleNamespace(id=2, chunk_index=1, content="The cell cycle includes growth and division."),
        ]

    def test_standard_and_hybrid_metadata_are_unpersonalized_baselines(self):
        self.assertIn("only", get_pipeline_metadata("standard")["retrieval"]["ranking"])
        self.assertIn("only", get_pipeline_metadata("hybrid")["retrieval"]["ranking"])

    def test_revision_intent_gets_stronger_gap_boost_than_direct_question(self):
        evidence = [
            {"chunk": self.chunks[0], "base_score": 0.8, "score": 0.8},
            {"chunk": self.chunks[1], "base_score": 0.7, "score": 0.7},
        ]
        gaps = {"cell cycle": 0.8}
        revision, revision_intent = rerank_with_observed_state(
            evidence, gaps, {}, "Help me review this"
        )
        direct, direct_intent = rerank_with_observed_state(
            evidence, gaps, {}, "What is the cell cycle?"
        )
        self.assertEqual(revision_intent, "revision")
        self.assertEqual(direct_intent, "direct_question")
        self.assertEqual(revision[0]["chunk"].id, 2)
        self.assertEqual(direct[0]["chunk"].id, 1)
        self.assertGreater(revision[0]["gap_boost"], direct[0]["gap_boost"])

    def test_intervention_is_explainable_and_labels_heuristics(self):
        recommendation = select_study_intervention({"cell cycle": 0.3}, {}, "revision")
        self.assertEqual(recommendation["action"], "targeted_practice")
        self.assertIn("observed quiz errors", recommendation["basis"])

        cold_start = select_study_intervention({}, {}, "direct_question")
        self.assertEqual(cold_start["action"], "build_baseline")

    def test_unmatched_chat_query_does_not_get_arbitrary_rrf_evidence(self):
        self.assertEqual(
            hybrid_retrieve(self.chunks, "quantum entanglement", mode="hybrid"),
            [],
        )

    def test_answer_citations_cannot_reference_unretrieved_chunks(self):
        evidence = [{"chunk": self.chunks[0]}]
        citations = _answer_citations(evidence, [1, 999, 1])
        self.assertEqual([item["chunk_id"] for item in citations], [1])


if __name__ == "__main__":
    unittest.main()
