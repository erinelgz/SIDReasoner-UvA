#!/usr/bin/env python
"""Compare ablation predictions with the stage-2 baseline using paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("exact_match@1", "HR@10", "NDCG@10", "catalog_valid@1", "copy_from_history@10")


def load_results(path: Path) -> list[dict]:
    with path.open() as handle:
        results = json.load(handle)
    return [result for result in results if not result.get("error")]


def align_results(baseline: list[dict], candidate: list[dict]) -> tuple[list[dict], list[dict]]:
    baseline_by_row = {int(result["row_index"]): result for result in baseline}
    candidate_by_row = {int(result["row_index"]): result for result in candidate}
    row_ids = sorted(set(baseline_by_row) & set(candidate_by_row))
    if len(row_ids) != len(baseline) or len(row_ids) != len(candidate):
        raise ValueError(
            f"Results are not paired: baseline={len(baseline)}, candidate={len(candidate)}, overlap={len(row_ids)}"
        )
    return [baseline_by_row[row_id] for row_id in row_ids], [candidate_by_row[row_id] for row_id in row_ids]


def per_example_metric(result: dict, metric: str) -> float:
    predictions = result.get("predict") or []
    target = result.get("target_sid", "")
    history = set(result.get("history_sids") or [])
    if metric == "exact_match@1":
        return float(bool(predictions) and predictions[0] == target)
    if metric == "catalog_valid@1":
        return float(bool(result.get("is_valid_catalog_sid")))
    if metric == "copy_from_history@10":
        return float(any(prediction in history for prediction in predictions[:10]))
    rank = next((index for index, prediction in enumerate(predictions[:10]) if prediction == target), None)
    if metric == "HR@10":
        return float(rank is not None)
    if metric == "NDCG@10":
        return 0.0 if rank is None else 1.0 / math.log2(rank + 2)
    raise KeyError(metric)


def paired_bootstrap(
    baseline: list[dict],
    candidate: list[dict],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict:
    baseline_values = np.array([per_example_metric(result, metric) for result in baseline], dtype=np.float64)
    candidate_values = np.array([per_example_metric(result, metric) for result in candidate], dtype=np.float64)
    differences = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap_means[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "baseline": float(baseline_values.mean()),
        "candidate": float(candidate_values.mean()),
        "delta": float(differences.mean()),
        "ci95": [float(low), float(high)],
    }


def classify(comparison: dict, candidate_metrics: dict) -> str:
    if candidate_metrics.get("errors", 1) != 0:
        return "regressed"
    if candidate_metrics.get("catalog_valid@1", 0.0) < 1.0:
        return "regressed"
    low, high = comparison["NDCG@10"]["ci95"]
    if low > 0:
        return "improved"
    if high < 0:
        return "regressed"
    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated subset of manifest variants to analyze.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    baseline_result_path = args.artifact_root / "stage2_baseline" / "evaluation" / "common.json"
    baseline = load_results(baseline_result_path)
    summaries = {}
    selected_variants = {name.strip() for name in args.variants.split(",") if name.strip()}

    for variant in manifest["variants"]:
        if selected_variants and variant["name"] not in selected_variants:
            continue
        step = variant["target_step"]
        evaluation_dir = args.artifact_root / variant["name"] / f"step_{step}" / "evaluation"
        result_path = evaluation_dir / "common.json"
        metrics_path = evaluation_dir / "common.metrics.json"
        candidate = load_results(result_path)
        candidate_metrics = json.loads(metrics_path.read_text())
        aligned_baseline, aligned_candidate = align_results(baseline, candidate)
        comparison = {
            metric: paired_bootstrap(
                aligned_baseline,
                aligned_candidate,
                metric,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for metric in METRICS
        }
        verdict = classify(comparison, candidate_metrics)
        variant["independent_metrics"]["common"] = str(metrics_path)
        variant["comparison_to_stage2"] = comparison
        variant["verdict"] = verdict
        summaries[variant["name"]] = {"verdict": verdict, "metrics": candidate_metrics, "comparison": comparison}

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    report_json = args.manifest.with_name("evaluation_report.json")
    report_json.write_text(json.dumps(summaries, indent=2) + "\n")

    lines = [
        "# Ablation Evaluation",
        "",
        "Primary metric: paired change in NDCG@10 against the stage-2 model.",
        "",
        "| Variant | Verdict | NDCG@10 | Delta | 95% CI | Exact@1 | HR@10 | Valid@1 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        metrics = summary["metrics"]
        ndcg = summary["comparison"]["NDCG@10"]
        lines.append(
            f"| {name} | {summary['verdict']} | {metrics.get('NDCG@10', 0):.4f} | "
            f"{ndcg['delta']:+.4f} | [{ndcg['ci95'][0]:+.4f}, {ndcg['ci95'][1]:+.4f}] | "
            f"{metrics.get('exact_match@1', 0):.4f} | {metrics.get('HR@10', 0):.4f} | "
            f"{metrics.get('catalog_valid@1', 0):.4f} |"
        )
    args.manifest.with_name("evaluation_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
