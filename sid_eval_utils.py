import json
import math
import os
import random
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


SID_RE = re.compile(r"<a_\d+><b_\d+><c_\d+>")
SID_PART_RE = re.compile(r"<([abc])_(\d+)>")
ANSWER_SEPARATOR = "</think>\n\n"
DEFAULT_TOPK = (1, 3, 5, 10, 20, 50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def derive_metrics_path(result_json_data: str) -> str:
    root, ext = os.path.splitext(result_json_data)
    if ext:
        return f"{root}.metrics{ext}"
    return f"{result_json_data}.metrics.json"


def parse_sid_parts(sid: str) -> tuple[str, str, str] | None:
    parts = SID_PART_RE.findall(sid or "")
    if len(parts) != 3:
        return None
    labels = [label for label, _ in parts]
    if labels != ["a", "b", "c"]:
        return None
    return tuple(f"<{label}_{value}>" for label, value in parts)


def extract_generated_sid(text: str) -> str:
    sid_match = SID_RE.search(text or "")
    return sid_match.group(0) if sid_match else (text or "").strip()


def extract_history_sids(text: str) -> list[str]:
    if not text:
        return []
    if "interacted with items " in text:
        text = text.split("interacted with items ", 1)[1]
    if ". Can you recommend" in text:
        text = text.split(". Can you recommend", 1)[0]
    return SID_RE.findall(text)


def normalize_prompt(prompt) -> list[dict]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt]


def load_verl_examples(
    parquet_path: str,
    tokenizer=None,
    num_samples: int = -1,
    seed: int = 42,
    direct_sid_decoding: bool = True,
) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    if num_samples is not None and num_samples > 0:
        df = df.sample(n=min(num_samples, len(df)), random_state=seed)

    examples = []
    for row_idx, row in df.iterrows():
        messages = normalize_prompt(row["prompt"])
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        target_sid = str(row["reward_model"]["ground_truth"])

        input_text = None
        if tokenizer is not None:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if direct_sid_decoding:
                input_text += f"<think>\n{ANSWER_SEPARATOR}"

        examples.append(
            {
                "row_index": int(row_idx),
                "prompt": messages,
                "input_text": input_text,
                "input_history": ", ".join(extract_history_sids(user_content)),
                "history_sids": extract_history_sids(user_content),
                "target_sid": target_sid,
            }
        )
    return examples


def load_valid_sids(index_file: str | None = None, info_file: str | None = None) -> set[str]:
    valid_sids = set()
    if index_file:
        with open(index_file) as f:
            index_data = json.load(f)
        for value in index_data.values():
            if isinstance(value, str):
                valid_sids.add(value)
            else:
                parts = list(value)
                if len(parts) >= 3:
                    valid_sids.add("".join(str(part) for part in parts[:3]))
    if info_file:
        with open(info_file) as f:
            for line in f:
                sid = line.split("\t", 1)[0].strip()
                if SID_RE.fullmatch(sid):
                    valid_sids.add(sid)
    return valid_sids


def compute_metrics(
    results: list[dict],
    valid_sids: set[str] | None = None,
    topk_values: Iterable[int] = DEFAULT_TOPK,
) -> dict:
    evaluated = [result for result in results if not result.get("error")]
    total = len(results)
    n = len(evaluated)
    metrics = {"total": total, "evaluated": n, "errors": total - n}
    if n == 0:
        return metrics

    valid_sids = valid_sids or set()
    first_predictions = [(result.get("predict") or [""])[0] for result in evaluated]
    targets = [result.get("target_sid", "") for result in evaluated]
    target_parts = [parse_sid_parts(target) for target in targets]
    pred_parts = [parse_sid_parts(prediction) for prediction in first_predictions]
    pred_counts = Counter(first_predictions)

    def is_catalog_valid(prediction: str, result: dict) -> bool:
        if "is_valid_catalog_sid" in result:
            return bool(result["is_valid_catalog_sid"])
        return bool(prediction in valid_sids) if valid_sids else False

    metrics.update(
        {
            "sid_pattern@1": sum(bool(SID_RE.fullmatch(prediction)) for prediction in first_predictions) / n,
            "catalog_valid@1": sum(
                is_catalog_valid(prediction, result)
                for prediction, result in zip(first_predictions, evaluated)
            )
            / n,
            "exact_match@1": sum(prediction == target for prediction, target in zip(first_predictions, targets)) / n,
            "a_match@1": sum(
                pred is not None and target is not None and pred[0] == target[0]
                for pred, target in zip(pred_parts, target_parts)
            )
            / n,
            "ab_match@1": sum(
                pred is not None and target is not None and pred[:2] == target[:2]
                for pred, target in zip(pred_parts, target_parts)
            )
            / n,
            "b_match@1": sum(
                pred is not None and target is not None and pred[1] == target[1]
                for pred, target in zip(pred_parts, target_parts)
            )
            / n,
            "c_match@1": sum(
                pred is not None and target is not None and pred[2] == target[2]
                for pred, target in zip(pred_parts, target_parts)
            )
            / n,
            "unique_pred@1": len(pred_counts),
            "unique_pred@1_rate": len(pred_counts) / n,
            "top10_pred_concentration@1": sum(count for _, count in pred_counts.most_common(10)) / n,
        }
    )
    for field, metric in (
        ("has_reasoning_open", "reasoning_open@1"),
        ("has_reasoning_close", "reasoning_close@1"),
        ("reasoning_enclosed", "reasoning_enclosed@1"),
    ):
        if any(field in result for result in evaluated):
            metrics[metric] = sum(bool(result.get(field)) for result in evaluated) / n
    if any("reward_parseable" in result for result in evaluated):
        metrics["reward_parseable@1"] = sum(bool(result.get("reward_parseable")) for result in evaluated) / n

    max_predictions = max(len(result.get("predict") or []) for result in evaluated)
    for topk in [topk for topk in topk_values if topk <= max_predictions]:
        hits = 0
        ndcg = 0.0
        copy_hits = 0
        for result in evaluated:
            predictions = result.get("predict") or []
            target = result.get("target_sid", "")
            history = set(result.get("history_sids") or extract_history_sids(result.get("input_history", "")))
            rank = next((idx for idx, prediction in enumerate(predictions[:topk]) if prediction == target), None)
            if rank is not None:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
            if any(prediction in history for prediction in predictions[:topk]):
                copy_hits += 1
        metrics[f"HR@{topk}"] = hits / n
        metrics[f"NDCG@{topk}"] = ndcg / n
        metrics[f"copy_from_history@{topk}"] = copy_hits / n

    return metrics
