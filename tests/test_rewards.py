import tempfile
import unittest
from pathlib import Path

from verl.utils.reward_score.direct_recommendation_StepRule_Office import (
    calculate_format_reward,
    calculate_reward,
    construct_prefix_allowed_hashmap,
    extract_solution,
)
from verl.utils.reward_score.direct_recommendation_StepRule_Office_exact_only import (
    MyRewardComputer as ExactRewardComputer,
)
from verl.utils.reward_score.direct_recommendation_StepRule_Office_no_validity import (
    MyRewardComputer as NoValidityRewardComputer,
)


class RewardTests(unittest.TestCase):
    def test_solution_extraction_requires_three_sid_parts_after_thinking(self):
        self.assertEqual(
            extract_solution("<think>reason</think>\n<a_1><b_2><c_3>"),
            ["<a_1>", "<b_2>", "<c_3>"],
        )
        self.assertIsNone(extract_solution("<a_1><b_2><c_3>"))
        self.assertIsNone(extract_solution("</think><a_1><b_2>"))

    def test_hierarchical_reward(self):
        target = ["<a_1>", "<b_2>", "<c_3>"]
        self.assertEqual(calculate_reward(target, target), 1.0)
        self.assertEqual(calculate_reward(["<a_1>", "<b_2>", "<c_9>"], target), 0.5)
        self.assertEqual(calculate_reward(["<a_1>", "<b_9>", "<c_3>"], target), 0.25)
        self.assertEqual(calculate_reward(["<a_9>", "<b_2>", "<c_3>"], target), 0.0)

    def test_catalog_validity_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            info_path = Path(directory) / "items.txt"
            info_path.write_text("<a_1><b_2><c_3>\tItem\t1\n")
            prefix_map = construct_prefix_allowed_hashmap(info_path)
        self.assertTrue(calculate_format_reward(["<a_1>", "<b_2>", "<c_3>"], prefix_map))
        self.assertFalse(calculate_format_reward(["<a_1>", "<b_2>", "<c_4>"], prefix_map))

    def test_ablation_rewards_and_invalid_output(self):
        exact = ExactRewardComputer()
        no_validity = NoValidityRewardComputer()
        target = "<a_1><b_2><c_3>"
        self.assertEqual(exact.compute("rec", "</think><a_1><b_2><c_3>", target), 1.0)
        self.assertEqual(exact.compute("rec", "</think><a_1><b_2><c_4>", target), 0.0)
        self.assertEqual(no_validity.compute("rec", "</think><a_1><b_2><c_4>", target), 0.5)
        self.assertEqual(no_validity.compute("rec", "invalid", target), 0.0)


if __name__ == "__main__":
    unittest.main()
