import unittest

from analyze_ablation_results import align_results, paired_bootstrap
from sid_eval_utils import compute_metrics, extract_generated_sid


class EvaluationTests(unittest.TestCase):
    def test_extract_generated_sid(self):
        self.assertEqual(extract_generated_sid("reason <a_1><b_2><c_3> tail"), "<a_1><b_2><c_3>")
        self.assertEqual(extract_generated_sid("invalid"), "invalid")

    def test_metrics_include_reasoning_and_invalid_generation(self):
        valid = {"<a_1><b_2><c_3>"}
        results = [
            {
                "row_index": 1,
                "target_sid": "<a_1><b_2><c_3>",
                "predict": ["<a_1><b_2><c_3>"],
                "history_sids": [],
                "is_valid_catalog_sid": True,
                "has_reasoning_open": True,
                "has_reasoning_close": True,
                "reasoning_enclosed": True,
                "reward_parseable": True,
            },
            {
                "row_index": 2,
                "target_sid": "<a_1><b_2><c_3>",
                "predict": ["invalid"],
                "history_sids": [],
                "is_valid_catalog_sid": False,
                "has_reasoning_open": True,
                "has_reasoning_close": False,
                "reasoning_enclosed": False,
                "reward_parseable": False,
            },
        ]
        metrics = compute_metrics(results, valid_sids=valid)
        self.assertEqual(metrics["exact_match@1"], 0.5)
        self.assertEqual(metrics["catalog_valid@1"], 0.5)
        self.assertEqual(metrics["reasoning_open@1"], 1.0)
        self.assertEqual(metrics["reasoning_close@1"], 0.5)
        self.assertEqual(metrics["reasoning_enclosed@1"], 0.5)
        self.assertEqual(metrics["reward_parseable@1"], 0.5)

    def test_paired_bootstrap_and_alignment(self):
        baseline = [
            {"row_index": 2, "target_sid": "b", "predict": ["x"], "history_sids": []},
            {"row_index": 1, "target_sid": "a", "predict": ["x"], "history_sids": []},
        ]
        candidate = [
            {"row_index": 1, "target_sid": "a", "predict": ["a"], "history_sids": []},
            {"row_index": 2, "target_sid": "b", "predict": ["b"], "history_sids": []},
        ]
        aligned_baseline, aligned_candidate = align_results(baseline, candidate)
        comparison = paired_bootstrap(
            aligned_baseline,
            aligned_candidate,
            "NDCG@10",
            samples=500,
            seed=42,
        )
        self.assertEqual(comparison["delta"], 1.0)
        self.assertGreater(comparison["ci95"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
