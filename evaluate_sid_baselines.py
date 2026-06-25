#!/usr/bin/env python
"""
Cheap SID recommendation baselines on the VERL parquet train/test splits.
"""

import json
import os
import random
from collections import Counter, defaultdict

import fire

from sid_eval_utils import (
    compute_metrics,
    derive_metrics_path,
    load_valid_sids,
    load_verl_examples,
    parse_sid_parts,
    set_seed,
)


def unique_ranked(candidates, fill, topk: int) -> list[str]:
    ranked = []
    seen = set()
    for sid in list(candidates) + list(fill):
        if sid and sid not in seen:
            ranked.append(sid)
            seen.add(sid)
        if len(ranked) >= topk:
            break
    return ranked


def build_results(examples, predictions_by_row, valid_sids: set[str]) -> list[dict]:
    results = []
    for example in examples:
        predictions = predictions_by_row(example)
        generated_sid = predictions[0] if predictions else ""
        results.append(
            {
                "row_index": example["row_index"],
                "input_history": example["input_history"],
                "history_sids": example["history_sids"],
                "target_sid": example["target_sid"],
                "output": example["target_sid"],
                "generated_output": generated_sid,
                "predict": predictions,
                "is_valid_catalog_sid": generated_sid in valid_sids,
                "is_exact_match": generated_sid == example["target_sid"],
            }
        )
    return results


def main(
    train_parquet: str = "./data/Amazon/rec_reasoning_verl/Office_Products/train.parquet",
    test_parquet: str = "./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet",
    info_file: str = "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt",
    index_file: str = "./data/Amazon/index/Office_Products.index.json",
    result_json_data: str = "./temp/baselines_verl_sid.json",
    metrics_json_data: str | None = None,
    topk: int = 50,
    num_samples: int = -1,
    seed: int = 42,
):
    set_seed(seed)
    valid_sids = load_valid_sids(index_file=index_file, info_file=info_file)
    valid_sid_list = sorted(valid_sids)

    print("Loading VERL train/test examples...")
    train_examples = load_verl_examples(train_parquet, tokenizer=None, num_samples=-1, seed=seed)
    test_examples = load_verl_examples(test_parquet, tokenizer=None, num_samples=num_samples, seed=seed)

    target_counts = Counter(example["target_sid"] for example in train_examples)
    global_popular = [sid for sid, _ in target_counts.most_common()]

    popular_by_a = defaultdict(Counter)
    popular_by_ab = defaultdict(Counter)
    for sid, count in target_counts.items():
        parts = parse_sid_parts(sid)
        if parts is None:
            continue
        popular_by_a[parts[0]][sid] += count
        popular_by_ab[parts[:2]][sid] += count

    ranked_by_a = {key: [sid for sid, _ in counts.most_common()] for key, counts in popular_by_a.items()}
    ranked_by_ab = {key: [sid for sid, _ in counts.most_common()] for key, counts in popular_by_ab.items()}

    def random_valid(example):
        rng = random.Random(seed + int(example["row_index"]))
        if len(valid_sid_list) <= topk:
            return valid_sid_list
        return rng.sample(valid_sid_list, topk)

    def global_popular_baseline(example):
        return unique_ranked([], global_popular, topk)

    def last_history(example):
        history = example["history_sids"]
        candidates = [history[-1]] if history else []
        return unique_ranked(candidates, global_popular, topk)

    def popular_same_a(example):
        history = example["history_sids"]
        last_parts = parse_sid_parts(history[-1]) if history else None
        candidates = ranked_by_a.get(last_parts[0], []) if last_parts else []
        return unique_ranked(candidates, global_popular, topk)

    def popular_same_ab(example):
        history = example["history_sids"]
        last_parts = parse_sid_parts(history[-1]) if history else None
        candidates = ranked_by_ab.get(last_parts[:2], []) if last_parts else []
        return unique_ranked(candidates, global_popular, topk)

    baselines = {
        "random_valid": random_valid,
        "global_popular": global_popular_baseline,
        "last_history": last_history,
        "popular_same_last_a": popular_same_a,
        "popular_same_last_ab": popular_same_ab,
    }

    all_results = {}
    all_metrics = {}
    for name, predictor in baselines.items():
        print(f"Evaluating baseline: {name}")
        results = build_results(test_examples, predictor, valid_sids)
        all_results[name] = results
        all_metrics[name] = compute_metrics(results, valid_sids=valid_sids)

    os.makedirs(os.path.dirname(result_json_data) or ".", exist_ok=True)
    with open(result_json_data, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics_json_data = metrics_json_data or derive_metrics_path(result_json_data)
    os.makedirs(os.path.dirname(metrics_json_data) or ".", exist_ok=True)
    with open(metrics_json_data, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nBaseline summary:")
    for name, metrics in all_metrics.items():
        print(
            f"{name:22s} HR@10={metrics.get('HR@10', 0):.4f} "
            f"NDCG@10={metrics.get('NDCG@10', 0):.4f} "
            f"exact@1={metrics.get('exact_match@1', 0):.4f}"
        )
    print(f"\nResults saved to: {result_json_data}")
    print(f"Metrics saved to: {metrics_json_data}")


if __name__ == "__main__":
    fire.Fire(main)
