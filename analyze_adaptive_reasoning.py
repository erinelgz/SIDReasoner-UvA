#!/usr/bin/env python
"""Analyze causal reasoning controls and calibrate adaptive reasoning gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_reasoning_utils import paired_bootstrap_delta, per_example_rank_metric


FEATURES = {
    "normalized_margin": {"direction": "low", "label": "beam probability margin"},
    "normalized_entropy": {"direction": "high", "label": "normalized beam entropy"},
}


def load_results(path: Path) -> list[dict]:
    results = json.loads(path.read_text())
    errors = [result for result in results if result.get("error")]
    if errors:
        raise ValueError(f"{path} contains {len(errors)} evaluation errors")
    return results


def align_results(*result_sets: list[dict]) -> tuple[list[int], list[list[dict]]]:
    maps = [{int(result["row_index"]): result for result in results} for results in result_sets]
    row_sets = [set(mapping) for mapping in maps]
    if not row_sets or any(rows != row_sets[0] for rows in row_sets[1:]):
        sizes = [len(rows) for rows in row_sets]
        overlap = len(set.intersection(*row_sets)) if row_sets else 0
        raise ValueError(f"Result files are not paired: sizes={sizes}, overlap={overlap}")
    row_ids = sorted(row_sets[0])
    return row_ids, [[mapping[row_id] for row_id in row_ids] for mapping in maps]


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def metric_values(results: list[dict], metric: str = "NDCG@10") -> np.ndarray:
    return np.asarray([per_example_rank_metric(result, metric) for result in results], dtype=np.float64)


def summarize_mode(results: list[dict]) -> dict:
    ndcg = metric_values(results)
    hr = metric_values(results, "HR@10")
    exact = metric_values(results, "exact_match@1")
    reasoning_tokens = np.asarray([result.get("reasoning_token_count", 0) for result in results], dtype=np.float64)
    latency = np.asarray([result.get("total_latency_seconds", 0.0) for result in results], dtype=np.float64)
    return {
        "examples": len(results),
        "NDCG@10": float(ndcg.mean()),
        "HR@10": float(hr.mean()),
        "exact_match@1": float(exact.mean()),
        "reasoning_tokens_total": int(reasoning_tokens.sum()),
        "reasoning_tokens_mean": float(reasoning_tokens.mean()),
        "latency_seconds_total": float(latency.sum()),
        "latency_seconds_mean": float(latency.mean()),
    }


def reasoning_length(result: dict) -> int:
    return int(result.get("reasoning_token_count", 0))


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def causal_summary(direct: list[dict], generated: list[dict], bootstrap_samples: int, seed: int) -> dict:
    direct_ndcg = metric_values(direct)
    generated_ndcg = metric_values(generated)
    delta = generated_ndcg - direct_ndcg
    oracle = np.maximum(direct_ndcg, generated_ndcg)
    better_fixed = max(float(direct_ndcg.mean()), float(generated_ndcg.mean()))
    lengths = np.asarray([reasoning_length(result) for result in generated], dtype=np.float64)
    repeat_mask = np.asarray(
        [
            bool(result.get("repeat_target", result.get("target_sid") in set(result.get("history_sids") or [])))
            for result in generated
        ],
        dtype=bool,
    )

    groups = {}
    for name, mask in (("repeat_target", repeat_mask), ("novel_target", ~repeat_mask)):
        groups[name] = {
            "examples": int(mask.sum()),
            "direct_NDCG@10": float(direct_ndcg[mask].mean()) if mask.any() else None,
            "generated_NDCG@10": float(generated_ndcg[mask].mean()) if mask.any() else None,
            "delta": float(delta[mask].mean()) if mask.any() else None,
        }

    return {
        "direct": summarize_mode(direct),
        "generated": summarize_mode(generated),
        "generated_vs_direct": paired_bootstrap_delta(
            direct_ndcg,
            generated_ndcg,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "helped_examples": int((delta > 0).sum()),
        "harmed_examples": int((delta < 0).sum()),
        "unchanged_examples": int((delta == 0).sum()),
        "oracle_NDCG@10": float(oracle.mean()),
        "oracle_improvement_over_best_fixed": float(oracle.mean() - better_fixed),
        "continue_to_adaptive_gate": bool(oracle.mean() - better_fixed >= 0.003),
        "reasoning_length_delta_correlation": safe_correlation(lengths, delta),
        "mean_reasoning_tokens_helped": float(lengths[delta > 0].mean()) if (delta > 0).any() else None,
        "mean_reasoning_tokens_harmed": float(lengths[delta < 0].mean()) if (delta < 0).any() else None,
        "groups": groups,
    }


def select_mask(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "low":
        return values <= threshold
    return values >= threshold


def feature_values(results: list[dict], feature: str) -> np.ndarray:
    values = []
    for result in results:
        confidence = result.get("confidence")
        if not confidence or feature not in confidence:
            raise ValueError(f"Direct result row {result.get('row_index')} lacks confidence feature {feature}")
        values.append(float(confidence[feature]))
    return np.asarray(values, dtype=np.float64)


def random_baseline(
    direct: np.ndarray,
    generated: np.ndarray,
    thinking_count: int,
    *,
    seed: int,
    samples: int = 200,
) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        mask = np.zeros(len(direct), dtype=bool)
        selected = rng.choice(len(direct), size=thinking_count, replace=False)
        mask[selected] = True
        means[index] = np.where(mask, generated, direct).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_NDCG@10": float(means.mean()), "ci95_across_assignments": [float(low), float(high)]}


def adaptive_gate_summary(
    row_ids: list[int],
    direct: list[dict],
    generated: list[dict],
    *,
    calibration_fraction: float,
    budgets: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    direct_ndcg = metric_values(direct)
    generated_ndcg = metric_values(generated)
    generated_tokens = np.asarray([reasoning_length(result) for result in generated], dtype=np.float64)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(row_ids))
    calibration_size = max(1, min(len(row_ids) - 1, round(len(row_ids) * calibration_fraction)))
    calibration_indices = np.sort(permutation[:calibration_size])
    heldout_indices = np.sort(permutation[calibration_size:])

    candidates = []
    for feature, metadata in FEATURES.items():
        values = feature_values(direct, feature)
        calibration_values = values[calibration_indices]
        for budget in budgets:
            quantile = budget if metadata["direction"] == "low" else 1.0 - budget
            threshold = float(np.quantile(calibration_values, quantile))
            calibration_mask = select_mask(calibration_values, threshold, metadata["direction"])
            heldout_mask = select_mask(values[heldout_indices], threshold, metadata["direction"])
            calibration_adaptive = np.where(
                calibration_mask,
                generated_ndcg[calibration_indices],
                direct_ndcg[calibration_indices],
            )
            heldout_adaptive = np.where(
                heldout_mask,
                generated_ndcg[heldout_indices],
                direct_ndcg[heldout_indices],
            )
            heldout_generated = generated_ndcg[heldout_indices]
            adaptive_tokens = float(generated_tokens[heldout_indices][heldout_mask].sum())
            always_tokens = float(generated_tokens[heldout_indices].sum())
            token_reduction = 1.0 - adaptive_tokens / always_tokens if always_tokens else 0.0
            comparison = paired_bootstrap_delta(
                heldout_generated,
                heldout_adaptive,
                samples=bootstrap_samples,
                seed=seed,
            )
            comparison_direct = paired_bootstrap_delta(
                direct_ndcg[heldout_indices],
                heldout_adaptive,
                samples=bootstrap_samples,
                seed=seed,
            )
            candidates.append(
                {
                    "feature": feature,
                    "feature_label": metadata["label"],
                    "target_thinking_rate": budget,
                    "threshold": threshold,
                    "calibration_thinking_rate": float(calibration_mask.mean()),
                    "heldout_thinking_rate": float(heldout_mask.mean()),
                    "calibration_NDCG@10": float(calibration_adaptive.mean()),
                    "heldout_NDCG@10": float(heldout_adaptive.mean()),
                    "heldout_vs_always_think": comparison,
                    "heldout_vs_direct": comparison_direct,
                    "heldout_reasoning_tokens": int(adaptive_tokens),
                    "heldout_reasoning_token_reduction": float(token_reduction),
                    "random_same_budget": random_baseline(
                        direct_ndcg[heldout_indices],
                        heldout_generated,
                        int(heldout_mask.sum()),
                        seed=seed + len(candidates),
                    ),
                    "accuracy_success": bool(comparison["delta"] >= 0.003),
                    "efficiency_success": bool(token_reduction >= 0.40 and comparison["delta"] >= -0.001),
                }
            )

    selected = sorted(
        candidates,
        key=lambda item: (
            -item["calibration_NDCG@10"],
            item["target_thinking_rate"],
            item["feature"],
        ),
    )[0]
    heldout_oracle = np.maximum(direct_ndcg[heldout_indices], generated_ndcg[heldout_indices])
    return {
        "calibration_seed": seed,
        "calibration_examples": len(calibration_indices),
        "heldout_examples": len(heldout_indices),
        "calibration_row_ids": [row_ids[index] for index in calibration_indices],
        "heldout_row_ids": [row_ids[index] for index in heldout_indices],
        "heldout_direct_NDCG@10": float(direct_ndcg[heldout_indices].mean()),
        "heldout_always_think_NDCG@10": float(generated_ndcg[heldout_indices].mean()),
        "heldout_oracle_NDCG@10": float(heldout_oracle.mean()),
        "candidates": candidates,
        "selected_on_calibration": selected,
        "success": bool(selected["accuracy_success"] or selected["efficiency_success"]),
    }


def comparison_summary(
    baseline: list[dict],
    candidate: list[dict],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    baseline_values = metric_values(baseline)
    candidate_values = metric_values(candidate)
    return {
        "metrics": summarize_mode(candidate),
        "NDCG@10_vs_generated": paired_bootstrap_delta(
            baseline_values,
            candidate_values,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def markdown_report(report: dict) -> str:
    causal = report["causal"]
    lines = [
        "# Adaptive Reasoning Analysis",
        "",
        "## Causal Reasoning Test",
        "",
        "| Mode | NDCG@10 | HR@10 | Exact@1 | Mean reasoning tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("direct", "generated"):
        metrics = causal[name]
        lines.append(
            f"| {name} | {metrics['NDCG@10']:.4f} | {metrics['HR@10']:.4f} | "
            f"{metrics['exact_match@1']:.4f} | {metrics['reasoning_tokens_mean']:.1f} |"
        )
    comparison = causal["generated_vs_direct"]
    lines.extend(
        [
            "",
            f"Generated reasoning changes NDCG@10 by `{comparison['delta']:+.4f}` "
            f"(95% CI `[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}]`).",
            "",
            f"The per-example oracle improves over the better fixed mode by "
            f"`{causal['oracle_improvement_over_best_fixed']:+.4f}`. "
            f"Adaptive gating is {'supported' if causal['continue_to_adaptive_gate'] else 'not supported'} "
            "by the predefined `0.003` continuation gate.",
            "",
        ]
    )
    if report.get("controls"):
        lines.extend(
            [
                "## Reasoning Controls",
                "",
                "| Control | NDCG@10 | Delta from generated | 95% CI |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for name, summary in report["controls"].items():
            comparison = summary["NDCG@10_vs_generated"]
            lines.append(
                f"| {name} | {summary['metrics']['NDCG@10']:.4f} | {comparison['delta']:+.4f} | "
                f"[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] |"
            )
        lines.append("")
    if report.get("adaptive_gate"):
        gate = report["adaptive_gate"]
        selected = gate["selected_on_calibration"]
        lines.extend(
            [
                "## Adaptive Gate",
                "",
                f"Selected `{selected['feature']}` with target thinking rate "
                f"`{selected['target_thinking_rate']:.0%}` using calibration only.",
                "",
                f"Held-out NDCG@10 is `{selected['heldout_NDCG@10']:.4f}` at an actual thinking rate of "
                f"`{selected['heldout_thinking_rate']:.1%}`, with "
                f"`{selected['heldout_reasoning_token_reduction']:.1%}` fewer reasoning tokens than always thinking.",
                "",
                f"Against always-direct prediction, the held-out NDCG@10 change is "
                f"`{selected['heldout_vs_direct']['delta']:+.4f}` "
                f"(95% CI `[{selected['heldout_vs_direct']['ci95'][0]:+.4f}, "
                f"{selected['heldout_vs_direct']['ci95'][1]:+.4f}]`).",
                "",
                f"Predefined success criterion: `{'met' if gate['success'] else 'not met'}`.",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-results", type=Path, required=True)
    parser.add_argument("--generated-results", type=Path, required=True)
    parser.add_argument("--control", action="append", default=[], help="Optional NAME=PATH causal control")
    parser.add_argument("--self-consistency", action="append", default=[], help="Optional NAME=PATH result")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--thinking-budgets", default="0.25,0.5,0.75")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [float(value) for value in args.thinking_budgets.split(",") if value.strip()]
    if not budgets or any(not 0.0 < budget < 1.0 for budget in budgets):
        raise SystemExit("Thinking budgets must be comma-separated values in (0, 1)")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be in (0, 1)")

    direct_raw = load_results(args.direct_results)
    generated_raw = load_results(args.generated_results)
    row_ids, aligned = align_results(direct_raw, generated_raw)
    direct, generated = aligned

    report = {
        "direct_results": str(args.direct_results),
        "generated_results": str(args.generated_results),
        "causal": causal_summary(direct, generated, args.bootstrap_samples, args.calibration_seed),
        "controls": {},
        "self_consistency": {},
    }

    for name, path in parse_named_paths(args.control).items():
        control_raw = load_results(path)
        _, aligned = align_results(generated, control_raw)
        generated_aligned, control = aligned
        report["controls"][name] = comparison_summary(
            generated_aligned,
            control,
            args.bootstrap_samples,
            args.calibration_seed,
        )

    if report["causal"]["continue_to_adaptive_gate"]:
        report["adaptive_gate"] = adaptive_gate_summary(
            row_ids,
            direct,
            generated,
            calibration_fraction=args.calibration_fraction,
            budgets=budgets,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.calibration_seed,
        )
    else:
        report["adaptive_gate"] = None

    for name, path in parse_named_paths(args.self_consistency).items():
        candidate_raw = load_results(path)
        _, aligned = align_results(generated, candidate_raw)
        generated_aligned, candidate = aligned
        report["self_consistency"][name] = comparison_summary(
            generated_aligned,
            candidate,
            args.bootstrap_samples,
            args.calibration_seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "adaptive_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "adaptive_analysis.md").write_text(markdown_report(report) + "\n")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
