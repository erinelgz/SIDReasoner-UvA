import unittest

from analyze_final_office_test import adaptive_rows, verdict


def row(index, target_rank, margin, entropy):
    target = f"target-{index}"
    predictions = [f"wrong-{index}", target] if target_rank == 2 else [target, f"wrong-{index}"]
    return {
        "row_index": index,
        "target_sid": target,
        "predict": predictions,
        "confidence": {
            "normalized_margin": margin,
            "normalized_entropy": entropy,
        },
    }


class FinalOfficeTestAnalysisTests(unittest.TestCase):
    def test_frozen_low_margin_gate_selects_generated_rows(self):
        direct = [row(0, 2, 0.1, 0.2), row(1, 1, 0.8, 0.2)]
        generated = [row(0, 1, 0.0, 0.0), row(1, 2, 0.0, 0.0)]
        adaptive, mask = adaptive_rows(
            direct,
            generated,
            {"feature": "normalized_margin", "direction": "low", "threshold": 0.25},
        )
        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(adaptive[0]["predict"], generated[0]["predict"])
        self.assertEqual(adaptive[1]["predict"], direct[1]["predict"])

    def test_frozen_high_entropy_gate_selects_generated_rows(self):
        direct = [row(0, 2, 0.2, 0.9), row(1, 1, 0.2, 0.1)]
        generated = [row(0, 1, 0.0, 0.0), row(1, 2, 0.0, 0.0)]
        _, mask = adaptive_rows(
            direct,
            generated,
            {"feature": "normalized_entropy", "direction": "high", "threshold": 0.8},
        )
        self.assertEqual(mask.tolist(), [True, False])

    def test_verdict_uses_confidence_interval(self):
        self.assertEqual(verdict({"ci95": [0.001, 0.01]}), "improved")
        self.assertEqual(verdict({"ci95": [-0.001, 0.01]}), "inconclusive")
        self.assertEqual(verdict({"ci95": [-0.01, -0.001]}), "regressed")


if __name__ == "__main__":
    unittest.main()
