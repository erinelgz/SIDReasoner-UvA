"""
Reward function for the Yelp Restaurants RL training stage.

Mirrors direct_recommendation_StepRule_Office.py; reads the Yelp info file
from SIDREASONER_YELP_INFO_FILE (or the default path below).

Reward structure:
  +0.25  if <a_X> matches
  +0.25  (×2 to 0.50) if <a_X><b_Y> both match
  +0.50  (×2 to 1.00) if all three tokens match
  +0.10  format bonus if the predicted SID is a valid trie entry
"""

import os
import re
from collections import defaultdict
from pathlib import Path

_SOLUTION_CLIP_CHARS = 50
_PROJECT_DIR = Path(os.environ.get("SIDREASONER_PROJECT_DIR", Path(__file__).resolve().parents[3]))


def extract_sid_tokens(s: str) -> list[str]:
    return re.findall(r"<[^>]+>", s)


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        answer_sids = extract_sid_tokens(final_answer)[:3]
        if len(answer_sids) == 3:
            return answer_sids
    return None


def calculate_reward(answer_sids, ground_truth_sids):
    score = 0.0
    if answer_sids[0] == ground_truth_sids[0]:
        score += 0.25
        if answer_sids[1] == ground_truth_sids[1]:
            score *= 2
            if answer_sids[2] == ground_truth_sids[2]:
                score *= 2
    return score


def calculate_format_reward(answer_sids, prefix_map):
    def is_valid(sid_list):
        if len(sid_list) < 3:
            return False
        a, b, c = sid_list[:3]
        if (a,) not in prefix_map:
            return False
        if b not in prefix_map[(a,)]:
            return False
        if (a, b) not in prefix_map:
            return False
        if c not in prefix_map[(a, b)]:
            return False
        return True

    return is_valid(answer_sids)


def construct_prefix_allowed_hashmap(item_info_path):
    sid_pattern = re.compile(r"<[^>]+>")
    prefix_map = defaultdict(set)
    with open(item_info_path, "r") as f:
        for line in f:
            semantic_id = line.split("\t")[0].strip()
            sid_list = sid_pattern.findall(semantic_id)
            if len(sid_list) != 3:
                continue
            a, b, c = sid_list
            prefix_map[(a,)].add(b)
            prefix_map[(a, b)].add(c)
    return {k: list(v) for k, v in prefix_map.items()}


class MyRewardComputer:
    def __init__(self):
        item_info_path = os.environ.get(
            "SIDREASONER_YELP_INFO_FILE",
            str(_PROJECT_DIR / "data/Yelp/info/Yelp_Restaurants_5core.txt"),
        )
        self.sid_hash = construct_prefix_allowed_hashmap(item_info_path)

    def compute(self, data_source, solution_str, ground_truth, extra_info=None):
        answer = extract_solution(solution_str=solution_str)
        ground_truth = extract_sid_tokens(ground_truth)[:3]
        if answer is None:
            return 0
        return calculate_reward(answer, ground_truth) + 0.1 * calculate_format_reward(answer, self.sid_hash)


_reward_computer: MyRewardComputer | None = None


def _get_reward_computer() -> MyRewardComputer:
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = MyRewardComputer()
    return _reward_computer


def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
    return _get_reward_computer().compute(data_source, solution_str, ground_truth, extra_info)
