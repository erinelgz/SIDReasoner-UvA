# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

from verl.utils.reward_score.direct_recommendation_StepRule_Office import extract_sid_tokens, extract_solution


class MyRewardComputer:
    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict | None = None,
    ) -> float:
        answer = extract_solution(solution_str=solution_str)
        ground_truth_sids = extract_sid_tokens(ground_truth)[:3]

        if answer is None or len(ground_truth_sids) < 3:
            return 0.0
        return 1.0 if answer[:3] == ground_truth_sids else 0.0


_reward_computer: MyRewardComputer | None = None


def _get_reward_computer() -> MyRewardComputer:
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = MyRewardComputer()
    return _reward_computer


def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
    rc = _get_reward_computer()
    return rc.compute(data_source, solution_str, ground_truth, extra_info)
