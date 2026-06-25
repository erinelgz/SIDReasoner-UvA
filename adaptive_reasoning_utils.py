"""Utilities shared by adaptive reasoning evaluation and analysis."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import numpy as np


def extract_reasoning_text(completion: str) -> tuple[str, bool, bool]:
    """Return reasoning content plus whether opening/closing tags were generated."""
    text = (completion or "").strip()
    has_open = "<think>" in text
    has_close = "</think>" in text
    if has_open:
        text = text.split("<think>", 1)[1]
    if has_close:
        text = text.split("</think>", 1)[0]
    return text.strip(), has_open, has_close


def truncate_reasoning(reasoning: str, tokenizer, fraction: float) -> tuple[str, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"truncate fraction must be in (0, 1], got {fraction}")
    token_ids = tokenizer.encode(reasoning or "", add_special_tokens=False)
    if not token_ids:
        return "", 0
    keep = max(1, math.ceil(len(token_ids) * fraction))
    kept_ids = token_ids[:keep]
    return tokenizer.decode(kept_ids, skip_special_tokens=True).strip(), len(kept_ids)


def deranged_source_rows(row_ids: Sequence[int], seed: int) -> dict[int, int]:
    """Create a deterministic derangement for shuffled-reasoning controls."""
    row_ids = list(row_ids)
    if len(row_ids) < 2:
        raise ValueError("At least two examples are required to shuffle reasoning")

    shuffled = row_ids.copy()
    rng = random.Random(seed)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = rng.randrange(index)
        shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    if any(target == source for target, source in zip(row_ids, shuffled)):
        raise RuntimeError("Failed to construct a reasoning derangement")
    return dict(zip(row_ids, shuffled))


def confidence_from_scores(scores: Sequence[float]) -> dict[str, float | list[float]]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return {
            "normalized_margin": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "probabilities": [],
        }
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    ordered = np.sort(probabilities)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
    entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum())
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return {
        "normalized_margin": margin,
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "probabilities": probabilities.tolist(),
    }


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]],
    *,
    limit: int = 10,
    rank_constant: float = 60.0,
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (rank_constant + rank)
            if item not in first_seen:
                first_seen[item] = sequence
                sequence += 1
    ordered = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))[:limit]
    return ordered, [scores[item] for item in ordered]


def per_example_rank_metric(result: dict, metric: str = "NDCG@10") -> float:
    predictions = result.get("predict") or []
    target = result.get("target_sid", result.get("output", ""))
    if metric == "exact_match@1":
        return float(bool(predictions) and predictions[0] == target)
    rank = next((index for index, prediction in enumerate(predictions[:10]) if prediction == target), None)
    if metric == "HR@10":
        return float(rank is not None)
    if metric == "NDCG@10":
        return 0.0 if rank is None else 1.0 / math.log2(rank + 2)
    raise KeyError(metric)


def paired_bootstrap_delta(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("Paired bootstrap requires equally sized, non-empty arrays")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap_means[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "baseline": float(baseline.mean()),
        "candidate": float(candidate.mean()),
        "delta": float(differences.mean()),
        "ci95": [float(low), float(high)],
    }
