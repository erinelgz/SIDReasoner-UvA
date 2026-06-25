import unittest

import numpy as np

from adaptive_reasoning_utils import (
    confidence_from_scores,
    deranged_source_rows,
    extract_reasoning_text,
    reciprocal_rank_fusion,
    truncate_reasoning,
)
from analyze_adaptive_reasoning import adaptive_gate_summary, causal_summary


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(token_ids)


def result(row_index, target, predictions, margin, entropy, reasoning_tokens=0):
    return {
        "row_index": row_index,
        "target_sid": target,
        "predict": predictions,
        "history_sids": [],
        "confidence": {
            "normalized_margin": margin,
            "normalized_entropy": entropy,
        },
        "reasoning_token_count": reasoning_tokens,
        "total_latency_seconds": 1.0,
    }


class AdaptiveReasoningTests(unittest.TestCase):
    def test_extract_and_truncate_reasoning(self):
        text, has_open, has_close = extract_reasoning_text("<think>alpha beta gamma</think><a_1>")
        self.assertEqual(text, "alpha beta gamma")
        self.assertTrue(has_open)
        self.assertTrue(has_close)
        truncated, count = truncate_reasoning(text, FakeTokenizer(), 0.5)
        self.assertEqual(truncated, "alpha beta")
        self.assertEqual(count, 2)

    def test_reasoning_shuffle_is_deterministic_derangement(self):
        rows = list(range(10))
        first = deranged_source_rows(rows, seed=42)
        second = deranged_source_rows(rows, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), set(rows))
        self.assertTrue(all(target != source for target, source in first.items()))

    def test_beam_confidence_is_normalized(self):
        confidence = confidence_from_scores([2.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(confidence["probabilities"]), 1.0)
        self.assertGreater(confidence["normalized_margin"], 0.0)
        self.assertGreaterEqual(confidence["normalized_entropy"], 0.0)
        self.assertLessEqual(confidence["normalized_entropy"], 1.0)

    def test_reciprocal_rank_fusion_combines_trajectories(self):
        ranking, scores = reciprocal_rank_fusion(
            [["a", "b", "c"], ["b", "a", "d"]],
            limit=3,
            rank_constant=0,
        )
        self.assertEqual(ranking[:2], ["a", "b"])
        self.assertEqual(len(scores), 3)

    def test_causal_oracle_and_calibration_gate(self):
        direct = []
        generated = []
        for row in range(20):
            target = f"target-{row}"
            uncertain = row % 2 == 0
            direct_predictions = ["wrong", target] if uncertain else [target]
            generated_predictions = [target] if uncertain else ["wrong", target]
            direct.append(
                result(
                    row,
                    target,
                    direct_predictions,
                    margin=0.05 if uncertain else 0.9,
                    entropy=0.95 if uncertain else 0.05,
                )
            )
            generated.append(
                result(
                    row,
                    target,
                    generated_predictions,
                    margin=0.0,
                    entropy=0.0,
                    reasoning_tokens=100,
                )
            )

        causal = causal_summary(direct, generated, bootstrap_samples=200, seed=42)
        self.assertGreater(causal["oracle_improvement_over_best_fixed"], 0.003)
        self.assertTrue(causal["continue_to_adaptive_gate"])

        gate = adaptive_gate_summary(
            list(range(20)),
            direct,
            generated,
            calibration_fraction=0.5,
            budgets=[0.25, 0.5, 0.75],
            bootstrap_samples=200,
            seed=42,
        )
        self.assertEqual(gate["calibration_examples"], 10)
        self.assertEqual(gate["heldout_examples"], 10)
        self.assertIn(gate["selected_on_calibration"]["feature"], {"normalized_margin", "normalized_entropy"})
        self.assertTrue(np.isfinite(gate["selected_on_calibration"]["heldout_NDCG@10"]))
        self.assertIn("heldout_vs_direct", gate["selected_on_calibration"])
        self.assertIn("ci95", gate["selected_on_calibration"]["heldout_vs_direct"])


if __name__ == "__main__":
    unittest.main()
