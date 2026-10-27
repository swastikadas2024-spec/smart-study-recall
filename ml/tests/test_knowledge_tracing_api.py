import sys
import unittest
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from api import KnowledgeTracingRequest, predict_knowledge_tracing  # noqa: E402


class KnowledgeTracingApiTests(unittest.TestCase):
    def request(self, **overrides):
        payload = {
            "interactions": [],
            "target_item_id": "topic-A",
            "target_history_seen": 2,
            "target_history_correct": 1,
            "elapsed_days": 1.0,
            "architecture": "lstm",
        }
        payload.update(overrides)
        return KnowledgeTracingRequest(**payload)

    def test_target_history_count_relationship_is_validated(self):
        with self.assertRaises(HTTPException) as caught:
            predict_knowledge_tracing(self.request(target_history_correct=3))
        self.assertEqual(caught.exception.status_code, 422)

    def test_history_timestamp_order_is_validated(self):
        events = [
            {"item_id": "a", "timestamp": 20, "history_seen": 0, "history_correct": 0, "session_seen": 1, "session_correct": 1},
            {"item_id": "b", "timestamp": 10, "history_seen": 1, "history_correct": 1, "session_seen": 1, "session_correct": 0},
        ]
        with self.assertRaises(HTTPException) as caught:
            predict_knowledge_tracing(self.request(interactions=events))
        self.assertEqual(caught.exception.status_code, 422)

    def test_architecture_is_restricted_to_available_candidates(self):
        with self.assertRaises(ValueError):
            self.request(architecture="gru")
        self.assertEqual(self.request(architecture="mlp").architecture, "mlp")


if __name__ == "__main__":
    unittest.main()
