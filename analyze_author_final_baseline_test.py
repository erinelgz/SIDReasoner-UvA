#!/usr/bin/env python
"""Analyze the post-freeze three-model Office Products baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

from adaptive_reasoning_utils import paired_bootstrap_delta, per_example_rank_metric


EXPECTED_EXAMPLES = 4866
EXPECTED_TEST_SHA256 = "c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"
MODEL_ORDER = ("author_stage3", "paper_stage3", "lora_no_constrained")
MODEL_LABELS = {
    "author_stage3": "Author-released Stage 3 (ckpt1000)",
    "paper_stage3": "Team reproduced Stage 3",
    "lora_no_constrained": "LoRA extension",
}
AUTHOR_PUBLIC_DATA_URL = "https://drive.google.com/file/d/1etg1e8oStGOjsg1Vr15vFnjlTMUx4Htz/view?usp=sharing"
OFFICE_PUBLIC_MEMBERS = {
    "test": "Amazon/test/Office_Products_5_2016-10-2018-11.csv",
    "info": "Amazon/info/Office_Products_5_2016-10-2018-11.txt",
    "index": "Amazon/index/Office_Products.index.json",
    "item": "Amazon/index/Office_Products.item.json",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([per_example_rank_metric(row, metric) for row in rows], dtype=np.float64)


def align_results(result_sets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    mappings = {name: {int(row["row_index"]): row for row in rows} for name, rows in result_sets.items()}
    row_sets = [set(mapping) for mapping in mappings.values()]
    if not row_sets or any(row_set != row_sets[0] for row_set in row_sets[1:]):
        raise ValueError("Result files do not have identical row IDs")
    row_ids = sorted(row_sets[0])
    aligned = {name: [mapping[row_id] for row_id in row_ids] for name, mapping in mappings.items()}
    targets = [[row["target_sid"] for row in rows] for rows in aligned.values()]
    if any(target != targets[0] for target in targets[1:]):
        raise ValueError("Result files do not have identical targets")
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


def summarize(rows: list[dict], metrics: dict) -> dict:
    tokens = np.asarray([row.get("reasoning_token_count", 0) for row in rows], dtype=np.float64)
    return {
        "examples": len(rows),
        "NDCG@10": float(metric_values(rows, "NDCG@10").mean()),
        "HR@10": float(metric_values(rows, "HR@10").mean()),
        "exact_match@1": float(metric_values(rows, "exact_match@1").mean()),
        "catalog_valid@1": float(metrics["catalog_valid@1"]),
        "reasoning_tokens_mean": float(tokens.mean()),
        "wall_seconds": float(metrics.get("wall_seconds", 0.0)),
    }


def comparison(baseline: list[dict], candidate: list[dict], *, samples: int, seed: int) -> dict:
    return {
        metric: paired_bootstrap_delta(
            metric_values(baseline, metric),
            metric_values(candidate, metric),
            samples=samples,
            seed=seed,
        )
        for metric in ("NDCG@10", "HR@10", "exact_match@1")
    }


def verdict(result: dict) -> str:
    low, high = result["ci95"]
    if low > 0:
        return "improved"
    if high < 0:
        return "regressed"
    return "inconclusive"


def sid_token_inventory(model_path: str) -> dict:
    """Summarize the SID tokens embedded in a checkpoint tokenizer."""
    tokens = json.loads((Path(model_path) / "added_tokens.json").read_text())
    sid_tokens = {token: token_id for token, token_id in tokens.items() if token.startswith(("<a_", "<b_", "<c_"))}
    return {
        "model_path": model_path,
        "sid_tokens": sid_tokens,
        "counts": {prefix: sum(token.startswith(f"<{prefix}_") for token in sid_tokens) for prefix in "abc"},
    }


def codebook_compatibility(author_model_path: str, reference_model_path: str) -> dict:
    """Check whether the author checkpoint can consume the local SID codebook."""
    author = sid_token_inventory(author_model_path)
    reference = sid_token_inventory(reference_model_path)
    author_tokens = set(author["sid_tokens"])
    reference_tokens = set(reference["sid_tokens"])
    missing_from_author = sorted(reference_tokens - author_tokens)
    missing_from_reference = sorted(author_tokens - reference_tokens)
    return {
        "compatible": not missing_from_author and not missing_from_reference,
        "author": {"model_path": author_model_path, "counts": author["counts"], "total": len(author_tokens)},
        "reference": {
            "model_path": reference_model_path,
            "counts": reference["counts"],
            "total": len(reference_tokens),
        },
        "common_tokens": len(author_tokens & reference_tokens),
        "missing_from_author": missing_from_author,
        "missing_from_reference": missing_from_reference,
        "missing_from_author_a_tokens": [token for token in missing_from_author if token.startswith("<a_")],
    }


def public_data_audit(archive_path: Path, local_data_root: Path, author_model_path: str) -> dict:
    """Verify the authors' public Office assets and their tokenizer compatibility."""
    local_paths = {
        "test": local_data_root / "test" / "Office_Products_5_2016-10-2018-11.csv",
        "info": local_data_root / "info" / "Office_Products_5_2016-10-2018-11.txt",
        "index": local_data_root / "index" / "Office_Products.index.json",
        "item": local_data_root / "index" / "Office_Products.item.json",
    }
    if not archive_path.is_file():
        raise FileNotFoundError(f"Author public data archive is missing: {archive_path}")
    if any(not path.is_file() for path in local_paths.values()):
        missing = [str(path) for path in local_paths.values() if not path.is_file()]
        raise FileNotFoundError(f"Local Office data asset is missing: {missing}")

    with zipfile.ZipFile(archive_path) as archive:
        member_bytes = {key: archive.read(member) for key, member in OFFICE_PUBLIC_MEMBERS.items()}
    public_hashes = {key: sha256_bytes(value) for key, value in member_bytes.items()}
    local_hashes = {key: sha256_bytes(path.read_bytes()) for key, path in local_paths.items()}
    public_index = json.loads(member_bytes["index"])
    index_tokens = {token for sid in public_index.values() for token in sid}
    author_tokens = set(sid_token_inventory(author_model_path)["sid_tokens"])
    missing_from_author = sorted(index_tokens - author_tokens)
    counts = {prefix: sum(token.startswith(f"<{prefix}_") for token in index_tokens) for prefix in "abc"}
    return {
        "source_url": AUTHOR_PUBLIC_DATA_URL,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "public_member_sha256": public_hashes,
        "local_member_sha256": local_hashes,
        "all_office_assets_byte_identical_to_local": public_hashes == local_hashes,
        "index_items": len(public_index),
        "index_sid_tokens": len(index_tokens),
        "index_sid_token_counts": counts,
        "missing_index_tokens_from_author_tokenizer": missing_from_author,
        "missing_index_a_tokens_from_author_tokenizer": [
            token for token in missing_from_author if token.startswith("<a_")
        ],
        "author_can_represent_all_public_index_tokens": not missing_from_author,
    }


def markdown_report(report: dict) -> str:
    compatibility = report["codebook_compatibility"]
    public_data = report["public_data_audit"]
    lines = [
        "# Author Checkpoint SID-Codebook Compatibility Diagnostic",
        "",
        "This audit checked the public Amazon data archive linked by the authors before running the author-released",
        "Office RL `ckpt1000` checkpoint. Its Office assets are byte-identical to the local test assets, but the",
        "released checkpoint tokenizer cannot represent all SID tokens in that public index. The resulting metrics",
        "are technical diagnostics only—not a valid comparison of recommendation quality or reasoning.",
        "",
        "## Public Data Provenance and Compatibility Finding",
        "",
        f"- Author public archive: [Amazon.zip]({public_data['source_url']}); SHA-256 `{public_data['archive_sha256']}`.",
        f"- The public Office test CSV, catalog text, index JSON, and item JSON are byte-identical to the local assets: `{public_data['all_office_assets_byte_identical_to_local']}`.",
        f"- Public Office index: `{public_data['index_items']}` items and `{public_data['index_sid_tokens']}` SID tokens "
        f"(`a={public_data['index_sid_token_counts']['a']}`, `b={public_data['index_sid_token_counts']['b']}`, "
        f"`c={public_data['index_sid_token_counts']['c']}`).",
        f"- Author checkpoint SID tokens: `{compatibility['author']['total']}` "
        f"(`a={compatibility['author']['counts']['a']}`, `b={compatibility['author']['counts']['b']}`, "
        f"`c={compatibility['author']['counts']['c']}`).",
        f"- Local Office checkpoint SID tokens: `{compatibility['reference']['total']}` "
        f"(`a={compatibility['reference']['counts']['a']}`, `b={compatibility['reference']['counts']['b']}`, "
        f"`c={compatibility['reference']['counts']['c']}`).",
        f"- Public-index SID tokens missing from the author tokenizer: `{len(public_data['missing_index_tokens_from_author_tokenizer'])}` "
        f"(`{len(public_data['missing_index_a_tokens_from_author_tokenizer'])}` first-level `<a_…>` tokens).",
        "- Downloading the authors' public data release therefore does not resolve the mismatch: it is exactly the local mapping already used.",
        "- The author checkpoint was forced to rank catalog SIDs that do not share its released tokenizer codebook.",
        "",
        "**Do not interpret any score or paired delta below as model quality, paper replication, or a reasoning effect.**",
        "",
        "## Technical Pipeline Audit",
        "",
        "- 4,866 local Office Products test examples.",
        f"- Test SHA-256: `{report['test_sha256']}`.",
        "- Deterministic seed 42, constrained beam-10 SID ranking, and identical direct/generated modes.",
        "- Every run completed without evaluation errors and produced catalog-valid top-one outputs.",
        "",
        "## Non-Comparable Output Metrics",
        "",
        "| Model | Mode | NDCG@10 | HR@10 | Exact@1 | Valid@1 | Mean reasoning tokens | Wall time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_key in MODEL_ORDER:
        for mode, label in (("direct", "Direct"), ("generated", "Generated")):
            summary = report["modes"][f"{model_key}_{mode}"]
            lines.append(
                f"| {MODEL_LABELS[model_key]} | {label} | {summary['NDCG@10']:.4f} | "
                f"{summary['HR@10']:.4f} | {summary['exact_match@1']:.4f} | "
                f"{summary['catalog_valid@1']:.4f} | {summary['reasoning_tokens_mean']:.1f} | "
                f"{summary['wall_seconds'] / 60:.1f} min |"
            )
    lines.extend(
        [
            "",
            "## Direct-versus-Generated Audit",
            "",
            "| Model | Generated minus direct NDCG@10 | 95% CI | Verdict |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for model_key in MODEL_ORDER:
        result = report["within_model_reasoning"][model_key]["NDCG@10"]
        lines.append(
            f"| {MODEL_LABELS[model_key]} | {result['delta']:+.4f} | "
            f"[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] | {verdict(result)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The generated-author output is retained to show that the technical reasoning pipeline runs, but its ",
            "small direct-to-generated change is not interpretable. The public author data release is byte-identical to ",
            "the local mapping and still lacks compatibility with the released checkpoint. A fair author evaluation ",
            "requires a different, model-matched SID codebook that is not in the public resources audited here. The frozen ",
            "reproduced-Stage-3-versus-LoRA report remains the valid local comparison; this document must not be cited as ",
            "a three-model ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-result-root", type=Path, required=True)
    parser.add_argument("--frozen-result-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--author-public-data-archive", type=Path, required=True)
    parser.add_argument("--local-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_roots = {
        "author_stage3": args.author_result_root,
        "paper_stage3": args.frozen_result_root / "paper_stage3",
        "lora_no_constrained": args.frozen_result_root / "lora_no_constrained",
    }
    raw_results = {}
    raw_metrics = {}
    jobs = {}
    for model_key, root in source_roots.items():
        for mode in ("direct", "generated"):
            key = f"{model_key}_{mode}"
            directory = root / mode
            raw_results[key] = load_json(directory / "result.json")
            raw_metrics[key] = load_json(directory / "metrics.json")
            verify_metrics(key, raw_metrics[key], mode)
            job_path = directory / "job.json"
            if job_path.exists():
                jobs[key] = load_json(job_path)

    aligned = align_results(raw_results)
    if any(len(rows) != EXPECTED_EXAMPLES for rows in aligned.values()):
        raise ValueError("Not every result file contains all 4,866 test examples")
    modes = {key: summarize(rows, raw_metrics[key]) for key, rows in aligned.items()}
    codebook = codebook_compatibility(
        jobs["author_stage3_direct"]["base_model"],
        jobs["paper_stage3_direct"]["base_model"],
    )
    public_data = public_data_audit(
        args.author_public_data_archive,
        args.local_data_root,
        jobs["author_stage3_direct"]["base_model"],
    )
    within_model_reasoning = {
        model_key: comparison(
            aligned[f"{model_key}_direct"],
            aligned[f"{model_key}_generated"],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for model_key in MODEL_ORDER
    }
    between_model = {}
    for mode in ("direct", "generated"):
        for baseline, candidate in itertools.combinations(MODEL_ORDER, 2):
            key = f"{mode}_{baseline}_vs_{candidate}"
            between_model[key] = {
                "mode": mode,
                "baseline": baseline,
                "candidate": candidate,
                "comparison": comparison(
                    aligned[f"{baseline}_{mode}"],
                    aligned[f"{candidate}_{mode}"],
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
            }
    report = {
        "status": "complete",
        "test_sha256": EXPECTED_TEST_SHA256,
        "provenance": load_json(args.provenance),
        "jobs": jobs,
        "raw_metrics": raw_metrics,
        "modes": modes,
        "performance_comparison_valid": public_data["author_can_represent_all_public_index_tokens"],
        "codebook_compatibility": codebook,
        "public_data_audit": public_data,
        "within_model_reasoning": within_model_reasoning,
        "between_model": between_model,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "BASELINE_TEST_REPORT.md").write_text(markdown_report(report))
    print(markdown_report(report), end="")


if __name__ == "__main__":
    main()
