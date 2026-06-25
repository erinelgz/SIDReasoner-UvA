#!/usr/bin/env python
"""Generate auditable, poster-ready copy for the adaptive-reasoning study.

The source evaluation outputs live in shared experiment storage rather than the
repository.  This script turns those frozen outputs into a compact Markdown
content pack and a JSON evidence bundle while checking the claims printed on
the poster.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_EXAMPLES = 4_866
EXPECTED_TEST_SHA256 = "c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"
DEFAULT_RESULT_ROOT = Path(
    "/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/" "final_office_test_20260613"
)
DEFAULT_TEST_CSV = Path("data/Amazon/test/Office_Products_5_2016-10-2018-11.csv")
DEFAULT_INFO_FILE = Path("data/Amazon/info/Office_Products_5_2016-10-2018-11.txt")
DEFAULT_FROZEN_MANIFEST = Path("experiments/final_office_test_20260613/frozen_manifest.json")
DEFAULT_OUTPUT_MARKDOWN = Path("experiments/adaptive_reasoning_20260613/POSTER_CONTENT.md")
DEFAULT_OUTPUT_JSON = Path("experiments/adaptive_reasoning_20260613/poster_test_examples.json")

ADAPTIVE_PROCESS_BULLET = (
    "**Adaptive reasoning:** first make a fast direct beam-10 recommendation; use its score "
    "distribution to detect uncertainty (low top-two margin or high beam entropy); only then "
    "generate a rationale and re-rank. Thresholds are calibrated on validation, so confident "
    "cases stay direct."
)
VALIDATION_BENCHMARK_NOTE = (
    "The earlier `1 helped / 5 harmed / 94 unchanged` (reproduced Stage-3) and "
    "`4 / 1 / 95` (LoRA) figures are from a separate 100-example **validation benchmark**; "
    "they are not held-out test results."
)

MODEL_LABELS = {
    "paper_stage3": "Reproduced Stage-3",
    "lora_no_constrained": "LoRA extension",
}
EXPECTED_CHANGE_COUNTS = {
    "paper_stage3": {"helped": 105, "harmed": 161, "unchanged": 4_600},
    "lora_no_constrained": {"helped": 142, "harmed": 190, "unchanged": 4_534},
}


@dataclass(frozen=True)
class PosterCaseDefinition:
    """A fixed, auditable illustrative case selected from the frozen test set."""

    case_id: str
    model_key: str
    outcome: str
    row_index: int
    history_indices: tuple[int, ...]
    expected_direct_rank: int | None
    expected_generated_rank: int | None
    trace_summary: str


POSTER_CASES = (
    PosterCaseDefinition(
        case_id="stage3_help_flair_pens",
        model_key="paper_stage3",
        outcome="Reasoning helps",
        row_index=3704,
        history_indices=(1, 3, 4),
        expected_direct_rank=None,
        expected_generated_rank=1,
        trace_summary=(
            "The trace shifts attention from art tools to colourful, medium-tip writing instruments, "
            "which moves the matching Flair pen variant to the top."
        ),
    ),
    PosterCaseDefinition(
        case_id="stage3_harm_highlighters",
        model_key="paper_stage3",
        outcome="Reasoning harms",
        row_index=97,
        history_indices=(0,),
        expected_direct_rank=1,
        expected_generated_rank=None,
        trace_summary=(
            "The trace generalises one yellow highlighter into broad office-supply preferences and "
            "falls back to the already-seen yellow item instead of the orange highlighter target."
        ),
    ),
    PosterCaseDefinition(
        case_id="lora_help_flair_pens",
        model_key="lora_no_constrained",
        outcome="Reasoning helps",
        row_index=2908,
        history_indices=(0,),
        expected_direct_rank=None,
        expected_generated_rank=1,
        trace_summary=(
            "The trace explicitly broadens a black felt-tip pen to colour and pack-size variants, "
            "which promotes the assorted-colour Flair pack to rank 1."
        ),
    ),
    PosterCaseDefinition(
        case_id="lora_harm_ink_cartridge",
        model_key="lora_no_constrained",
        outcome="Reasoning harms",
        row_index=2111,
        history_indices=(0, 1, 2),
        expected_direct_rank=1,
        expected_generated_rank=None,
        trace_summary=(
            "The trace blends a relevant black ink cartridge with unrelated phone and hardware purchases, "
            "then shifts the recommendation away from the matching cyan ink cartridge."
        ),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Required artifact is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Artifact is not valid JSON: {path}") from error


def load_result_rows(path: Path) -> dict[int, dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    rows: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object result row")
        if record.get("error"):
            raise ValueError(f"{path} contains an evaluation error at row {record.get('row_index')}")
        if "row_index" not in record:
            raise ValueError(f"{path} contains a row without row_index")
        row_index = int(record["row_index"])
        if row_index in rows:
            raise ValueError(f"{path} contains duplicate row_index {row_index}")
        rows[row_index] = record
    return rows


def load_test_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as error:
        raise ValueError(f"Test CSV is missing: {path}") from error
    required_columns = {"history_item_title", "item_title", "history_item_sid", "item_sid"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"Test CSV does not provide the expected columns: {path}")
    return rows


def load_sid_titles(info_file: Path, test_rows: list[dict[str, str]]) -> dict[str, str]:
    try:
        lines = info_file.read_text().splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"SID catalogue is missing: {info_file}") from error

    titles: dict[str, str] = {}
    for line in lines:
        fields = line.split("\t", 2)
        if len(fields) >= 2 and fields[0] and fields[1]:
            titles[fields[0]] = fields[1]
    for row in test_rows:
        sid = row["item_sid"]
        title = row["item_title"]
        if sid and title:
            titles[sid] = title
    if not titles:
        raise ValueError("No SID-to-title mappings were loaded")
    return titles


def target_rank(row: dict[str, Any]) -> int | None:
    predictions = row.get("predict") or []
    target = row.get("target_sid", row.get("output", ""))
    for index, prediction in enumerate(predictions[:10], start=1):
        if prediction == target:
            return index
    return None


def ndcg_at_10(row: dict[str, Any]) -> float:
    rank = target_rank(row)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def change_kind(direct: dict[str, Any], generated: dict[str, Any]) -> str:
    difference = ndcg_at_10(generated) - ndcg_at_10(direct)
    if difference > 0:
        return "helped"
    if difference < 0:
        return "harmed"
    return "unchanged"


def summarize_changes(
    direct_rows: dict[int, dict[str, Any]], generated_rows: dict[int, dict[str, Any]]
) -> dict[str, int]:
    counts = {"helped": 0, "harmed": 0, "unchanged": 0}
    for row_index in sorted(direct_rows):
        counts[change_kind(direct_rows[row_index], generated_rows[row_index])] += 1
    return counts


def paired_bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> dict[str, float | list[float]]:
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("Paired bootstrap requires equally sized, non-empty arrays")
    differences = candidate - baseline
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 100
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        indices = generator.integers(0, differences.size, size=(size, differences.size))
        means[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "delta": float(differences.mean()),
        "ci95": [float(low), float(high)],
    }


def validate_result_sets(
    results: dict[str, dict[str, dict[int, dict[str, Any]]]], test_rows: list[dict[str, str]]
) -> None:
    if len(test_rows) != EXPECTED_EXAMPLES:
        raise ValueError(f"Expected {EXPECTED_EXAMPLES} test rows, found {len(test_rows)}")
    expected_row_ids = set(range(EXPECTED_EXAMPLES))

    for model_key, modes in results.items():
        direct_rows = modes["direct"]
        generated_rows = modes["generated"]
        if set(direct_rows) != expected_row_ids or set(generated_rows) != expected_row_ids:
            raise ValueError(f"{model_key} does not contain exactly the frozen {EXPECTED_EXAMPLES} test rows")
        for row_index in expected_row_ids:
            direct = direct_rows[row_index]
            generated = generated_rows[row_index]
            expected_target = test_rows[row_index]["item_sid"]
            if direct.get("target_sid") != generated.get("target_sid"):
                raise ValueError(f"{model_key} target mismatch between modes at row {row_index}")
            if direct.get("target_sid") != expected_target:
                raise ValueError(f"{model_key} target does not match test CSV at row {row_index}")
            if direct.get("split") != "test" or generated.get("split") != "test":
                raise ValueError(f"{model_key} includes a non-test result row at {row_index}")

    for row_index in expected_row_ids:
        targets = {
            results[model_key][mode][row_index].get("target_sid")
            for model_key in MODEL_LABELS
            for mode in ("direct", "generated")
        }
        if len(targets) != 1:
            raise ValueError(f"Target mismatch across models at row {row_index}")


def title_for(sid: str, titles: dict[str, str]) -> str:
    if not sid or sid not in titles:
        raise ValueError(f"No catalogue title is available for SID {sid!r}")
    return titles[sid]


def selected_history(test_row: dict[str, str], indices: tuple[int, ...]) -> tuple[list[str], list[str]]:
    try:
        history_titles = list(ast.literal_eval(test_row["history_item_title"]))
        history_sids = list(ast.literal_eval(test_row["history_item_sid"]))
    except (SyntaxError, ValueError) as error:
        raise ValueError("Selected test row has unreadable history data") from error
    if len(history_titles) != len(history_sids):
        raise ValueError("Selected test row has mismatched history titles and SIDs")
    if any(index < 0 or index >= len(history_titles) for index in indices):
        raise ValueError("Poster case refers to a missing history item")
    return [history_titles[index] for index in indices], [history_sids[index] for index in indices]


def validate_case(
    definition: PosterCaseDefinition,
    direct: dict[str, Any],
    generated: dict[str, Any],
) -> None:
    direct_rank = target_rank(direct)
    generated_rank = target_rank(generated)
    if direct_rank != definition.expected_direct_rank or generated_rank != definition.expected_generated_rank:
        raise ValueError(
            f"Poster case {definition.case_id} no longer has the expected rank transition "
            f"({direct_rank!r} -> {generated_rank!r})"
        )
    expected_kind = "helped" if definition.outcome == "Reasoning helps" else "harmed"
    if change_kind(direct, generated) != expected_kind:
        raise ValueError(f"Poster case {definition.case_id} no longer has the expected outcome")
    reasonings = generated.get("reasonings") or []
    if not reasonings or not str(reasonings[0].get("text", "")).strip():
        raise ValueError(f"Poster case {definition.case_id} has no saved generated trace")


def build_case(
    definition: PosterCaseDefinition,
    direct: dict[str, Any],
    generated: dict[str, Any],
    test_row: dict[str, str],
    titles: dict[str, str],
) -> dict[str, Any]:
    validate_case(definition, direct, generated)
    history_titles, history_sids = selected_history(test_row, definition.history_indices)
    direct_rank = target_rank(direct)
    generated_rank = target_rank(generated)
    direct_top_sid = str((direct.get("predict") or [""])[0])
    generated_top_sid = str((generated.get("predict") or [""])[0])
    target_sid = str(direct["target_sid"])
    generated_trace = str(generated["reasonings"][0]["text"])
    direct_ndcg = ndcg_at_10(direct)
    generated_ndcg = ndcg_at_10(generated)
    return {
        "case_id": definition.case_id,
        "model_key": definition.model_key,
        "model_label": MODEL_LABELS[definition.model_key],
        "outcome": definition.outcome,
        "source": {
            "split": "test",
            "row_index": definition.row_index,
            "direct_mode": "direct",
            "generated_mode": "generated",
        },
        "history": {
            "selected_titles": history_titles,
            "selected_sids": history_sids,
            "all_sids": list(ast.literal_eval(test_row["history_item_sid"])),
        },
        "target": {"sid": target_sid, "title": title_for(target_sid, titles)},
        "direct": {
            "top_recommendation": {"sid": direct_top_sid, "title": title_for(direct_top_sid, titles)},
            "target_rank": direct_rank,
            "ndcg_at_10": direct_ndcg,
        },
        "generated_reasoning": {
            "top_recommendation": {"sid": generated_top_sid, "title": title_for(generated_top_sid, titles)},
            "target_rank": generated_rank,
            "ndcg_at_10": generated_ndcg,
            "ndcg_change": generated_ndcg - direct_ndcg,
            "reasoning_token_count": int(generated.get("reasoning_token_count", 0)),
            "full_generated_trace": generated_trace,
            "poster_trace_summary": definition.trace_summary,
        },
    }


def rank_label(rank: int | None) -> str:
    return "not in top 10" if rank is None else f"#{rank}"


def summarize_adaptive_gate(
    direct_rows: dict[int, dict[str, Any]],
    generated_rows: dict[int, dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    row_ids = sorted(direct_rows)
    feature = str(gate["feature"])
    direction = str(gate["direction"])
    threshold = float(gate["threshold"])
    should_think = np.asarray(
        [
            float(direct_rows[row_index]["confidence"][feature]) <= threshold
            if direction == "low"
            else float(direct_rows[row_index]["confidence"][feature]) >= threshold
            for row_index in row_ids
        ],
        dtype=bool,
    )
    direct_ndcg = np.asarray([ndcg_at_10(direct_rows[row_index]) for row_index in row_ids], dtype=np.float64)
    generated_ndcg = np.asarray([ndcg_at_10(generated_rows[row_index]) for row_index in row_ids], dtype=np.float64)
    adaptive_ndcg = np.where(should_think, generated_ndcg, direct_ndcg)
    generated_tokens = np.asarray(
        [generated_rows[row_index].get("reasoning_token_count", 0) for row_index in row_ids], dtype=np.float64
    )
    total_tokens = float(generated_tokens.sum())
    token_reduction = 1.0 - float(generated_tokens[should_think].sum()) / total_tokens
    comparison = paired_bootstrap_delta(direct_ndcg, adaptive_ndcg)
    if comparison["ci95"][0] > 0 or comparison["ci95"][1] < 0:
        raise ValueError("Frozen adaptive gate unexpectedly has a conclusive accuracy change versus direct")
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "thinking_rate": float(should_think.mean()),
        "adaptive_ndcg_at_10": float(adaptive_ndcg.mean()),
        "delta_vs_direct": comparison,
        "reasoning_token_reduction_vs_always_think": token_reduction,
    }


def summarize_model(
    model_key: str,
    direct_rows: dict[int, dict[str, Any]],
    generated_rows: dict[int, dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    counts = summarize_changes(direct_rows, generated_rows)
    if counts != EXPECTED_CHANGE_COUNTS[model_key]:
        raise ValueError(
            f"{model_key} change counts differ from the frozen poster claim: "
            f"expected {EXPECTED_CHANGE_COUNTS[model_key]}, found {counts}"
        )
    row_ids = sorted(direct_rows)
    direct_ndcg = np.asarray([ndcg_at_10(direct_rows[row_index]) for row_index in row_ids], dtype=np.float64)
    generated_ndcg = np.asarray([ndcg_at_10(generated_rows[row_index]) for row_index in row_ids], dtype=np.float64)
    reasoning_comparison = paired_bootstrap_delta(direct_ndcg, generated_ndcg)
    if reasoning_comparison["ci95"][1] >= 0:
        raise ValueError(f"{model_key} generated reasoning no longer has a supported test-set regression")
    return {
        "label": MODEL_LABELS[model_key],
        "reasoning_change_counts": counts,
        "direct_ndcg_at_10": float(direct_ndcg.mean()),
        "generated_ndcg_at_10": float(generated_ndcg.mean()),
        "generated_minus_direct": reasoning_comparison,
        "adaptive_gate": summarize_adaptive_gate(direct_rows, generated_rows, gate),
    }


def render_markdown(evidence: dict[str, Any]) -> str:
    summaries = evidence["model_summaries"]
    stage3 = summaries["paper_stage3"]
    lora = summaries["lora_no_constrained"]
    lines = [
        "# Adaptive Reasoning: Spend Reasoning Tokens Only When Needed",
        "",
        "## How It Works",
        "",
        f"- {ADAPTIVE_PROCESS_BULLET}",
        "- For the frozen test policy, the reproduced Stage-3 model reasons below a validation-set "
        "margin threshold; the LoRA model reasons above a validation-set entropy threshold.",
        "- This is a routing policy: it does not know the target item at inference time, so it can miss "
        "some cases where reasoning would have helped and select some cases where it hurts.",
        "",
        "## Held-Out Office Products Test (n = 4,866)",
        "",
        "| Model | Reasoning helped | Reasoning harmed | Unchanged | Generated − direct NDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Reproduced Stage-3 | "
        f"{stage3['reasoning_change_counts']['helped']} | "
        f"{stage3['reasoning_change_counts']['harmed']} | "
        f"{stage3['reasoning_change_counts']['unchanged']} | "
        f"{stage3['generated_minus_direct']['delta']:+.4f} |",
        "| LoRA extension | "
        f"{lora['reasoning_change_counts']['helped']} | "
        f"{lora['reasoning_change_counts']['harmed']} | "
        f"{lora['reasoning_change_counts']['unchanged']} | "
        f"{lora['generated_minus_direct']['delta']:+.4f} |",
        "",
        "- Always generating reasoning reduced NDCG@10 for both models: reproduced Stage-3 "
        f"{stage3['generated_minus_direct']['delta']:+.4f} "
        f"(95% CI [{stage3['generated_minus_direct']['ci95'][0]:+.4f}, "
        f"{stage3['generated_minus_direct']['ci95'][1]:+.4f}]); LoRA "
        f"{lora['generated_minus_direct']['delta']:+.4f} "
        f"(95% CI [{lora['generated_minus_direct']['ci95'][0]:+.4f}, "
        f"{lora['generated_minus_direct']['ci95'][1]:+.4f}]).",
        "- The frozen confidence gates invoked reasoning for "
        f"{stage3['adaptive_gate']['thinking_rate']:.1%} (Stage-3) and "
        f"{lora['adaptive_gate']['thinking_rate']:.1%} (LoRA) of test cases, saving "
        f"{stage3['adaptive_gate']['reasoning_token_reduction_vs_always_think']:.1%} and "
        f"{lora['adaptive_gate']['reasoning_token_reduction_vs_always_think']:.1%} of reasoning tokens. "
        "Their changes versus always-direct inference were inconclusive, so direct prediction remains "
        "the accuracy default.",
        "",
        f"> **Validation-only note:** {VALIDATION_BENCHMARK_NOTE}",
        "",
        "## Illustrative Held-Out Test Cases",
        "",
        "These cards are concrete examples of individual rank changes. They are illustrative, not representative rates. "
        "The accompanying JSON preserves the complete model-generated trace and source SIDs.",
        "",
    ]
    for case in evidence["cases"]:
        direct = case["direct"]
        generated = case["generated_reasoning"]
        lines.extend(
            [
                f"### {case['model_label']} — {case['outcome']}",
                "",
                f"- **Recent history:** {'; '.join(case['history']['selected_titles'])}",
                f"- **Target next item:** {case['target']['title']}",
                f"- **Direct recommendation:** {direct['top_recommendation']['title']} "
                f"(target {rank_label(direct['target_rank'])})",
                f"- **After generated reasoning:** {generated['top_recommendation']['title']} "
                f"(target {rank_label(generated['target_rank'])}; "
                f"ΔNDCG@10 {generated['ndcg_change']:+.4f})",
                f"- **Trace summary (human-readable summary of the saved model trace):** "
                f"{generated['poster_trace_summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Poster Takeaway",
            "",
            "Reasoning changes only a minority of rankings and can either reveal a useful product variant or "
            "distract the recommender. Confidence-gated reasoning substantially reduces its token cost, but this "
            "test does not establish an accuracy gain over always-direct recommendation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_evidence(
    result_root: Path,
    test_csv: Path,
    info_file: Path,
    frozen_manifest: Path,
) -> dict[str, Any]:
    manifest = load_json(frozen_manifest)
    gates = manifest.get("adaptive_gates")
    if not isinstance(gates, dict) or set(gates) != set(MODEL_LABELS):
        raise ValueError("Frozen manifest does not contain both adaptive-gate definitions")
    expected_sha256 = manifest.get("test_data", {}).get("sha256")
    if expected_sha256 != EXPECTED_TEST_SHA256:
        raise ValueError("Frozen manifest does not contain the expected test checksum")
    actual_sha256 = file_sha256(test_csv)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Test CSV checksum mismatch: expected {expected_sha256}, found {actual_sha256}")

    test_rows = load_test_rows(test_csv)
    results = {
        model_key: {
            mode: load_result_rows(result_root / model_key / mode / "result.json") for mode in ("direct", "generated")
        }
        for model_key in MODEL_LABELS
    }
    validate_result_sets(results, test_rows)
    titles = load_sid_titles(info_file, test_rows)
    summaries = {
        model_key: summarize_model(
            model_key,
            results[model_key]["direct"],
            results[model_key]["generated"],
            gates[model_key],
        )
        for model_key in MODEL_LABELS
    }
    cases = [
        build_case(
            definition,
            results[definition.model_key]["direct"][definition.row_index],
            results[definition.model_key]["generated"][definition.row_index],
            test_rows[definition.row_index],
            titles,
        )
        for definition in POSTER_CASES
    ]
    return {
        "title": "Adaptive-Reasoning Poster Content Evidence",
        "test_protocol": {
            "split": "Office Products held-out test",
            "examples": EXPECTED_EXAMPLES,
            "sha256": actual_sha256,
            "frozen_manifest": str(frozen_manifest),
        },
        "source_artifacts": {
            "result_root": str(result_root),
            "test_csv": str(test_csv),
            "sid_catalogue": str(info_file),
        },
        "benchmark_provenance_note": VALIDATION_BENCHMARK_NOTE,
        "model_summaries": summaries,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--info-file", type=Path, default=DEFAULT_INFO_FILE)
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = build_evidence(args.result_root, args.test_csv, args.info_file, args.frozen_manifest)
    markdown = render_markdown(evidence)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n")
    args.output_json.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Wrote {args.output_markdown}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
