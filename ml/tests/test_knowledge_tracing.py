import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from knowledge_tracing import SequenceKnowledgeTracer, build_sequences  # noqa: E402


class KnowledgeTracingSequenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"user_id": "a", "lexeme_id": "x", "timestamp": 100, "history_seen": 0, "history_correct": 0, "session_seen": 2, "session_correct": 2},
            {"user_id": "a", "lexeme_id": "x", "timestamp": 200, "history_seen": 2, "history_correct": 2, "session_seen": 1, "session_correct": 1},
            {"user_id": "a", "lexeme_id": "y", "timestamp": 200, "history_seen": 1, "history_correct": 1, "session_seen": 1, "session_correct": 0},
            {"user_id": "a", "lexeme_id": "x", "timestamp": 300, "history_seen": 3, "history_correct": 3, "session_seen": 2, "session_correct": 1},
            {"user_id": "b", "lexeme_id": "z", "timestamp": 100, "history_seen": 0, "history_correct": 0, "session_seen": 1, "session_correct": 0},
        ])

    def test_windows_are_causal_and_do_not_cross_learners(self):
        windows, lengths, labels, seen, users = build_sequences(
            self.frame, {"x": 2, "y": 3}, sequence_length=3
        )
        self.assertEqual(lengths.tolist(), [1, 2, 2, 3, 1])
        self.assertEqual(windows[3, 2, 0], 2)  # current target item
        self.assertAlmostEqual(float(windows[3, 2, 4]), 0.5)  # prior timestamp group average
        self.assertEqual(lengths[2], 2)  # same-time peer answer was excluded from history
        self.assertAlmostEqual(float(windows[2, 1, 4]), 1.0)  # previous timestamp score
        self.assertEqual(int(windows[4, 0, 0]), 1)  # held-out item uses OOV token
        self.assertEqual(int(windows[4, 1, 0]), 0)  # learner b has no learner-a history
        self.assertEqual(labels.tolist(), [1.0, 1.0, 0.0, 0.5, 0.0])
        self.assertEqual(seen.tolist(), [2.0, 1.0, 1.0, 2.0, 1.0])
        self.assertEqual(users.tolist(), ["a", "a", "a", "a", "b"])

    def test_transformer_and_lstm_forward_shapes_and_ignore_padding(self):
        base = np.zeros((2, 4, 5), dtype=np.float32)
        base[0, :3, 0] = [2, 3, 2]
        base[1, 0, 0] = 2
        x = torch.from_numpy(base)
        lengths = torch.tensor([3, 1])
        for architecture in ("lstm", "transformer"):
            model = SequenceKnowledgeTracer(
                item_count=4, architecture=architecture, sequence_length=4
            ).build().eval()
            with torch.no_grad():
                original = model(x, lengths)
                changed = x.clone()
                changed[1, 3, 0] = 3
                changed[1, 3, 1:] = 99
                after_padding_change = model(changed, lengths)
            self.assertEqual(tuple(original.shape), (2,))
            self.assertTrue(torch.all((original >= 0) & (original <= 1)))
            self.assertTrue(torch.allclose(original, after_padding_change, atol=1e-5))

    def test_input_contract_rejects_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "Missing interaction fields"):
            build_sequences(pd.DataFrame([{"user_id": "a"}]), {}, sequence_length=2)


if __name__ == "__main__":
    unittest.main()
