#!/usr/bin/env python
"""Analyze the frozen paired Office Products test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_reasoning_utils import paired_bootstrap_delta, per_example_rank_metric


EXPECTED_EXAMPLES = 4866
EXPECTED_TEST_SHA256 = "c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"
PAPER_NDCG10 = 0.1208
PAPER_RECALL10 = 0.1648
FROZEN_GATES = {
    "paper_stage3": {
        "feature": "normalized_margin",
        "direction": "low",
        "threshold": 0.24685489670011115,
    },
    "lora_no_constrained": {
        "feature": "normalized_entropy",
        "direction": "high",
        "threshold": 0.8546990742765369,
    },
}


def load_json(path: Path):
    return json.loads(path.read_text())


def metric_values(results: list[dict], metric: str) -> np.ndarray:
    return np.asarray([per_example_rank_metric(row, metric) for row in results], dtype=np.float64)


def align_results(result_sets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    maps = {name: {int(row["row_index"]): row for row in rows} for name, rows in result_sets.items()}
    row_sets = [set(mapping) for mapping in maps.values()]
    if not row_sets or any(rows != row_sets[0] for rows in row_sets[1:]):
        raise ValueError("Final-test result files do not contain identical row IDs")
    row_ids = sorted(row_sets[0])
    aligned = {name: [mapping[row_id] for row_id in row_ids] for name, mapping in maps.items()}
    targets = [[row["target_sid"] for row in rows] for rows in aligned.values()]
    if any(values != targets[0] for values in targets[1:]):
        raise ValueError("Final-test targets are not paired across result files")
    return aligned


def verify_metrics(name: str, metrics: dict, run_kind: str) -> None:
    failures = []
    if metrics.get("total") != EXPECTED_EXAMPLES or metrics.get("evaluated") != EXPECTED_EXAMPLES:
        failures.append("example count")
    if metrics.get("errors") != 0:
        failures.append("evaluation errors")
    if metrics.get("catalog_valid@1") != 1.0:
        failures.append("catalog validity")
    if metrics.get("data_sha256") != EXPECTED_TEST_SHA256:
        failures.append("test checksum")
    if metrics.get("split") != "test":
        failures.append("split")
    if run_kind == "generated" and metrics.get("generated_reasoning_close_rate", 0.0) < 0.95:
        failures.append("reasoning closure")
    if failures:
        raise ValueError(f"{name} failed verification: {', '.join(failures)}")


def summarize(rows: list[dict]) -> dict:
    tokens = np.asarray([row.get("reasoning_token_count", 0) for row in rows], dtype=np.float64)
    latency = np.asarray([row.get("total_latency_seconds", 0.0) for row in rows], dtype=np.float64)
    return {
        "examples": len(rows),
        "NDCG@10": float(metric_values(rows, "NDCG@10").mean()),
        "HR@10": float(metric_values(rows, "HR@10").mean()),
        "exact_match@1": float(metric_values(rows, "exact_match@1").mean()),
        "reasoning_tokens_total": int(tokens.sum()),
        "reasoning_tokens_mean": float(tokens.mean()),
        "latency_seconds_total": float(latency.sum()),
        "latency_seconds_mean": float(latency.mean()),
    }


def paired_comparison(
    baseline: list[dict],
    candidate: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    return {
        metric: paired_bootstrap_delta(
            metric_values(baseline, metric),
            metric_values(candidate, metric),
            samples=bootstrap_samples,
            seed=seed,
        )
        for metric in ("NDCG@10", "HR@10", "exact_match@1")
    }


def verdict(comparison: dict) -> str:
    low, high = comparison["ci95"]
    if low > 0:
        return "improved"
    if high < 0:
        return "regressed"
    return "inconclusive"


def group_summary(rows: list[dict]) -> dict:
    output = {}
    for group_name, repeat_target in (("repeat_target", True), ("novel_target", False)):
        selected = [row for row in rows if bool(row.get("repeat_target")) is repeat_target]
        output[group_name] = summarize(selected)
    return output


def adaptive_rows(direct: list[dict], generated: list[dict], gate: dict) -> tuple[list[dict], np.ndarray]:
    selected = []
    mask = []
    for direct_row, generated_row in zip(direct, generated):
        confidence = direct_row.get("confidence") or {}
        value = float(confidence[gate["feature"]])
        should_think = value <= gate["threshold"] if gate["direction"] == "low" else value >= gate["threshold"]
        mask.append(should_think)
        source = generated_row if should_think else direct_row
        row = dict(source)
        row["adaptive_selected_reasoning"] = should_think
        selected.append(row)
    return selected, np.asarray(mask, dtype=bool)


def adaptive_summary(
    direct: list[dict],
    generated: list[dict],
    gate: dict,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    adaptive, mask = adaptive_rows(direct, generated, gate)
    generated_tokens = np.asarray(
        [row.get("reasoning_token_count", 0) for row in generated],
        dtype=np.float64,
    )
    adaptive_tokens = float(generated_tokens[mask].sum())
    total_tokens = float(generated_tokens.sum())
    return {
        "gate": gate,
        "thinking_rate": float(mask.mean()),
        "metrics": summarize(adaptive),
        "reasoning_tokens": int(adaptive_tokens),
        "reasoning_token_reduction_vs_always_think": (1.0 - adaptive_tokens / total_tokens if total_tokens else 0.0),
        "vs_direct": paired_comparison(
            direct,
            adaptive,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "vs_always_think": paired_comparison(
            generated,
            adaptive,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def markdown_report(report: dict) -> str:
    modes = report["modes"]
    primary = report["comparisons"]["primary_lora_vs_stage3_generated"]["NDCG@10"]
    secondary = report["comparisons"]["secondary_lora_vs_stage3_direct"]["NDCG@10"]
    lines = [
        "# Final Office Products Test Evaluation",
        "",
        "## Frozen Protocol",
        "",
        f"- Test examples: `{report['verification']['examples']}`.",
        f"- Test SHA-256: `{report['verification']['test_sha256']}`.",
        "- Constrained beam-10 SID ranking, deterministic reasoning, seed `42`.",
        "- Generated reasoning is the primary paper-style endpoint; direct inference is secondary.",
        "",
        "## Results",
        "",
        "| Model | Mode | NDCG@10 | HR@10 | Exact@1 | Mean reasoning tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model_key, label in (
        ("paper_stage3", "Reproduced Stage 3"),
        ("lora_no_constrained", "LoRA extension"),
    ):
        for run_kind, mode_label in (("direct", "Direct"), ("generated", "Generated")):
            values = modes[f"{model_key}_{run_kind}"]
            lines.append(
                f"| {label} | {mode_label} | {values['NDCG@10']:.4f} | "
                f"{values['HR@10']:.4f} | {values['exact_match@1']:.4f} | "
                f"{values['reasoning_tokens_mean']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            f"Primary generated-reasoning comparison: LoRA changes NDCG@10 by "
            f"`{primary['delta']:+.4f}` (95% CI "
            f"`[{primary['ci95'][0]:+.4f}, {primary['ci95'][1]:+.4f}]`), "
            f"classified as **{report['primary_verdict']}**.",
            "",
            f"Secondary direct comparison: LoRA changes NDCG@10 by "
            f"`{secondary['delta']:+.4f}` (95% CI "
            f"`[{secondary['ci95'][0]:+.4f}, {secondary['ci95'][1]:+.4f}]`).",
            "",
            "## Frozen Adaptive Gates",
            "",
            "| Model | Think rate | Adaptive NDCG@10 | Delta vs direct | 95% CI | Token reduction | Verdict vs direct |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for model_key, label in (
        ("paper_stage3", "Reproduced Stage 3"),
        ("lora_no_constrained", "LoRA extension"),
    ):
        adaptive = report["adaptive"][model_key]
        comparison = adaptive["vs_direct"]["NDCG@10"]
        lines.append(
            f"| {label} | {adaptive['thinking_rate']:.1%} | "
            f"{adaptive['metrics']['NDCG@10']:.4f} | {comparison['delta']:+.4f} | "
            f"[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] | "
            f"{adaptive['reasoning_token_reduction_vs_always_think']:.1%} | "
            f"{verdict(comparison)} |"
        )
    lines.extend(
        [
            "",
            "## Repeat and Novel Targets",
            "",
            "| Model | Mode | Repeat NDCG@10 | Novel NDCG@10 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for model_key, label in (
        ("paper_stage3", "Reproduced Stage 3"),
        ("lora_no_constrained", "LoRA extension"),
    ):
        for run_kind, mode_label in (("direct", "Direct"), ("generated", "Generated")):
            groups = report["groups"][f"{model_key}_{run_kind}"]
            lines.append(
                f"| {label} | {mode_label} | "
                f"{groups['repeat_target']['NDCG@10']:.4f} | "
                f"{groups['novel_target']['NDCG@10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paper Context",
            "",
            f"The paper reports Office Products Recall@10 `{PAPER_RECALL10:.4f}` and "
            f"NDCG@10 `{PAPER_NDCG10:.4f}` after Stage-3 RL. These values are context only. "
            "The defensible causal claim is the paired LoRA-versus-local-reproduction result "
            "under this frozen evaluator.",
            "",
            "## Test-Set Caveat",
            "",
            "This test split had previously been evaluated once on the imported Stage-2 checkpoint. "
            "No Stage-3 or LoRA test result was used to choose the frozen checkpoints, gates, or "
            "decoding settings reported here.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = {
        "paper_stage3_direct": ("paper_stage3", "direct"),
        "paper_stage3_generated": ("paper_stage3", "generated"),
        "lora_no_constrained_direct": ("lora_no_constrained", "direct"),
        "lora_no_constrained_generated": ("lora_no_constrained", "generated"),
    }
    raw_results = {}
    metrics = {}
    jobs = {}
    for key, (model_key, run_kind) in keys.items():
        directory = args.result_root / model_key / run_kind
        raw_results[key] = load_json(directory / "result.json")
        metrics[key] = load_json(directory / "metrics.json")
        jobs[key] = load_json(directory / "job.json")
        verify_metrics(key, metrics[key], run_kind)

    results = align_results(raw_results)
    if any(len(rows) != EXPECTED_EXAMPLES for rows in results.values()):
        raise ValueError("Final-test result files do not all contain 4,866 examples")

    modes = {key: summarize(rows) for key, rows in results.items()}
    comparisons = {
        "primary_lora_vs_stage3_generated": paired_comparison(
            results["paper_stage3_generated"],
            results["lora_no_constrained_generated"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "secondary_lora_vs_stage3_direct": paired_comparison(
            results["paper_stage3_direct"],
            results["lora_no_constrained_direct"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "stage3_generated_vs_direct": paired_comparison(
            results["paper_stage3_direct"],
            results["paper_stage3_generated"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "lora_generated_vs_direct": paired_comparison(
            results["lora_no_constrained_direct"],
            results["lora_no_constrained_generated"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    adaptive = {
        model_key: adaptive_summary(
            results[f"{model_key}_direct"],
            results[f"{model_key}_generated"],
            FROZEN_GATES[model_key],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for model_key in FROZEN_GATES
    }
    groups = {key: group_summary(rows) for key, rows in results.items()}
    primary = comparisons["primary_lora_vs_stage3_generated"]["NDCG@10"]
    frozen_manifest = load_json(args.frozen_manifest)
    report = {
        "protocol": frozen_manifest,
        "verification": {
            "examples": EXPECTED_EXAMPLES,
            "test_sha256": EXPECTED_TEST_SHA256,
            "all_errors_zero": True,
            "all_catalog_valid_at_1": True,
            "generated_reasoning_close_rate_at_least_95_percent": True,
        },
        "jobs": jobs,
        "raw_metrics": metrics,
        "modes": modes,
        "comparisons": comparisons,
        "primary_verdict": verdict(primary),
        "adaptive": adaptive,
        "groups": groups,
        "paper_context": {
            "Office_Products_Recall@10": PAPER_RECALL10,
            "Office_Products_NDCG@10": PAPER_NDCG10,
        },
    }
    final_manifest = dict(frozen_manifest)
    final_manifest["status"] = "complete"
    final_manifest["jobs"] = jobs
    final_manifest["independent_test_results"] = {
        "modes": modes,
        "comparisons": comparisons,
        "primary_verdict": report["primary_verdict"],
        "adaptive": adaptive,
        "groups": groups,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "final_test_report.md").write_text(markdown_report(report) + "\n")
    (args.output_dir / "final_manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
