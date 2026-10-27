import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from knowledge_tracing import SequenceKnowledgeTracer, build_sequences  # noqa: E402
from train_knowledge_tracer import make_splits, train_deep  # noqa: E402


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
        for architecture in ("mlp", "lstm", "transformer"):
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

    def test_mlp_training_returns_train_and_validation_history(self):
        rng = np.random.default_rng(7)
        windows = np.zeros((8, 3, 5), dtype=np.float32)
        lengths = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.int64)
        for row, length in enumerate(lengths):
            windows[row, :length, 0] = 2
            windows[row, :length, 1:] = rng.normal(size=(length, 4))
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        seen = np.ones(8, dtype=np.float32)
        model, scaler, best_epoch, best_loss, history = train_deep(
            "mlp", windows[:6], lengths[:6], labels[:6], seen[:6],
            windows[6:], lengths[6:], labels[6:], seen[6:],
            seed=11, max_epochs=2, batch_size=3, item_count=4,
            device=torch.device("cpu"), optimizer_name="adamw",
        )
        self.assertIn(best_epoch, [entry["epoch"] for entry in history])
        self.assertTrue(np.isfinite(best_loss))
        self.assertTrue(np.isfinite(history[0]["train_weighted_binomial_loss"]))
        self.assertTrue(np.isfinite(history[0]["validation_weighted_binomial_loss"]))
        self.assertIsNotNone(scaler)
        self.assertIsNotNone(model)

    def test_forward_time_split_keeps_equal_timestamps_together(self):
        frame = pd.DataFrame({"timestamp": np.repeat(np.arange(10), [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])})
        splits = make_splits(frame, "forward_time", seed=42)
        assigned = {}
        for split_name, indexes in zip(("train", "validation", "test"), splits):
            for timestamp in frame.timestamp.iloc[indexes].unique():
                self.assertNotIn(timestamp, assigned)
                assigned[timestamp] = split_name
        self.assertEqual(len(assigned), frame.timestamp.nunique())


if __name__ == "__main__":
    unittest.main()
