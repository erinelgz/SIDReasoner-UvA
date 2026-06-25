#!/usr/bin/env python
"""Capture, stop, and archive the June 2026 Office Products ablation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "office_products_ablation_20260612"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / RUN_ID
DEFAULT_ARTIFACT_ROOT = Path(
    "/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/office_products_ablation_20260612"
)
ARRAY_JOB_ID = "23681188"
STEP_RE = re.compile(r"training/global_step:(\d+)")
SAVE_RE = re.compile(r"Saved model to (.*?/global_step_(\d+)/actor)/model_world_size_")
VAL_RE = re.compile(
    r"step:(\d+).*?val-aux/rec/Office_Products/reward/mean@1:([0-9.eE+-]+)"
    r".*?val-core/rec/Office_Products/reward/mean@10:([0-9.eE+-]+)"
)


@dataclass(frozen=True)
class Variant:
    task_id: int
    name: str
    log_name: str
    hydra_run: str
    target_step: int
    constrained_decoding: bool
    checkpoint_type: str

    @property
    def job_id(self) -> str:
        return f"{ARRAY_JOB_ID}_{self.task_id}"

    @property
    def log_path(self) -> Path:
        return PROJECT_ROOT / "logs" / self.log_name

    @property
    def slurm_path(self) -> Path:
        return PROJECT_ROOT / "slurm_output" / f"sid-rl-ablation-{ARRAY_JOB_ID}_{self.task_id}.out"

    @property
    def hydra_path(self) -> Path:
        return PROJECT_ROOT / "outputs" / self.hydra_run / ".hydra"


VARIANTS = (
    Variant(0, "control", "Office_Products_stage3_rl_control_Qwen3-1.7B.log", "2026-06-12/04-35-24", 500, True, "fsdp"),
    Variant(
        1,
        "no_validity",
        "Office_Products_stage3_rl_no_validity_Qwen3-1.7B.log",
        "2026-06-12/04-46-22",
        500,
        True,
        "fsdp",
    ),
    Variant(
        2,
        "no_constrained",
        "Office_Products_stage3_rl_no_constrained_Qwen3-1.7B.log",
        "2026-06-12/04-51-54",
        500,
        False,
        "fsdp",
    ),
    Variant(
        3,
        "exact_only",
        "Office_Products_stage3_rl_exact_only_Qwen3-1.7B.log",
        "2026-06-12/05-42-03",
        500,
        True,
        "fsdp",
    ),
    Variant(
        4,
        "lora_no_validity",
        "Office_Products_stage3_rl_lora_no_validity_Qwen3-1.7B.log",
        "2026-06-12/05-51-42",
        200,
        True,
        "lora",
    ),
    Variant(
        5,
        "lora_no_constrained",
        "Office_Products_stage3_rl_lora_no_constrained_Qwen3-1.7B.log",
        "2026-06-12/05-51-44",
        200,
        False,
        "lora",
    ),
)

FAILED_RUNS = (
    {"job_id": "23632570", "cause": "Initial full ablation failed during startup."},
    {"job_id": "23652272-23653996", "cause": "Successive smoke attempts failed during environment/data setup."},
    {"job_id": "23662117", "cause": "Generated shell command contained an unterminated string literal."},
    {"job_id": "23662124", "cause": "Used a Python environment without Transformers installed."},
    {"job_id": "23662135", "cause": "Passed a non-integer pandas random_state."},
    {"job_id": "23662161", "cause": "Quoted PROJECT_DIR prevented shell variable expansion in a data path."},
    {"job_id": "23662164", "cause": "vLLM expected all_special_tokens_extended on Qwen2Tokenizer."},
    {"job_id": "23662214", "cause": "Ray worker could not import the one-off tokenizer patch module."},
    {"job_id": "23680644", "cause": "Smoke run reached its one-hour Slurm time limit."},
    {"job_id": "23686411", "cause": "Completed smoke run; superseded by reproducible run 23688674."},
)


def run_command(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stderr}")
    return result.stdout + result.stderr


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def latest_step(log_text: str) -> int | None:
    matches = STEP_RE.findall(log_text)
    return int(matches[-1]) if matches else None


def validation_series(log_text: str) -> list[dict]:
    return [
        {"step": int(step), "aux_reward_mean_at_1": float(aux), "core_reward_mean_at_10": float(core)}
        for step, aux, core in VAL_RE.findall(log_text)
    ]


def target_actor_path(variant: Variant, log_text: str) -> Path | None:
    saves = [(Path(path), int(step)) for path, step in SAVE_RE.findall(log_text)]
    if not saves:
        return None
    experiment_root = saves[-1][0].parents[1]
    return experiment_root / f"global_step_{variant.target_step}" / "actor"


def checkpoint_status(variant: Variant) -> dict:
    log_text = read_text(variant.log_path)
    actor_path = target_actor_path(variant, log_text)
    status = {
        "latest_step": latest_step(log_text),
        "target_step": variant.target_step,
        "actor_path": str(actor_path) if actor_path else None,
        "complete": False,
        "reason": "target path not discovered",
    }
    if actor_path is None:
        return status
    if not actor_path.is_dir():
        status["reason"] = "target actor directory does not exist"
        return status

    config_path = actor_path / "fsdp_config.json"
    try:
        world_size = int(json.loads(config_path.read_text())["world_size"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        status["reason"] = "missing or invalid fsdp_config.json"
        return status

    model_shards = sorted(actor_path.glob(f"model_world_size_{world_size}_rank_*.pt"))
    hf_files = (actor_path / "huggingface" / "config.json", actor_path / "huggingface" / "tokenizer.json")
    tracker = actor_path.parents[1] / "latest_checkpointed_iteration.txt"
    try:
        tracked_step = int(tracker.read_text().strip())
    except (FileNotFoundError, ValueError):
        tracked_step = -1

    if len(model_shards) != world_size:
        status["reason"] = f"found {len(model_shards)}/{world_size} model shards"
    elif not all(path.is_file() for path in hf_files):
        status["reason"] = "Hugging Face metadata is incomplete"
    elif (
        variant.checkpoint_type == "lora" and not (actor_path / "lora_adapter" / "adapter_model.safetensors").is_file()
    ):
        status["reason"] = "LoRA adapter is missing"
    elif tracked_step < variant.target_step:
        status["reason"] = f"checkpoint tracker is at step {tracked_step}"
    else:
        status["complete"] = True
        status["reason"] = "complete"
    status["world_size"] = world_size
    status["model_shards"] = len(model_shards)
    status["tracked_step"] = tracked_step
    return status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_actor(variant: Variant, actor_path: Path, artifact_root: Path) -> dict:
    destination = artifact_root / variant.name / f"step_{variant.target_step}" / "raw_actor"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(actor_path.glob("model_world_size_*_rank_*.pt")):
        shutil.copy2(source, destination / source.name)
    shutil.copy2(actor_path / "fsdp_config.json", destination / "fsdp_config.json")
    for directory in ("huggingface", "lora_adapter"):
        source = actor_path / directory
        if source.is_dir():
            shutil.copytree(source, destination / directory, dirs_exist_ok=True)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    checksums = {str(path.relative_to(destination)): sha256_file(path) for path in files}
    manifest_path = destination.parent / "raw_actor_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "variant": variant.name,
                "step": variant.target_step,
                "source": str(actor_path),
                "destination": str(destination),
                "files": checksums,
            },
            indent=2,
        )
        + "\n"
    )
    return {"path": str(destination), "manifest": str(manifest_path), "file_count": len(files)}


def cluster_snapshot() -> dict:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "squeue": run_command(["squeue", "-u", subprocess.getoutput("id -un")], check=False),
        "sacct": run_command(
            [
                "sacct",
                "-j",
                ARRAY_JOB_ID,
                "-X",
                "--format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End",
                "-n",
            ],
            check=False,
        ),
    }


def build_manifest(artifact_root: Path) -> dict:
    variants = []
    for variant in VARIANTS:
        log_text = read_text(variant.log_path)
        artifact_dir = artifact_root / variant.name / f"step_{variant.target_step}"
        metrics_dir = artifact_dir / "evaluation"
        raw_manifest_path = artifact_dir / "raw_actor_manifest.json"
        raw_manifest = json.loads(raw_manifest_path.read_text()) if raw_manifest_path.is_file() else None
        variants.append(
            {
                **asdict(variant),
                "job_id": variant.job_id,
                "variant": variant.name,
                "selected_step": variant.target_step,
                "latest_step": latest_step(log_text),
                "online_validation": validation_series(log_text),
                "configuration": {
                    "resolved_hydra": str(artifact_root / "evidence" / "hydra" / variant.name / "config.yaml"),
                    "overrides": str(artifact_root / "evidence" / "hydra" / variant.name / "overrides.yaml"),
                    "constrained_decoding": variant.constrained_decoding,
                    "checkpoint_type": variant.checkpoint_type,
                },
                "artifact_dir": str(artifact_dir),
                "artifact_checksum_manifest": (str(raw_manifest_path) if raw_manifest is not None else None),
                "artifact_checksums": raw_manifest["files"] if raw_manifest is not None else {},
                "merged_model": str(artifact_dir / "model"),
                "merged_model_checksum_manifest": str(artifact_dir / "model.sha256"),
                "independent_metrics": {
                    "common": str(metrics_dir / "common.metrics.json"),
                    "native": str(metrics_dir / "native.metrics.json"),
                    "reasoning_smoke": str(metrics_dir / "reasoning_smoke.metrics.json"),
                },
                "verdict": "pending",
            }
        )
    downstream_path = EXPERIMENT_DIR / "evidence" / "downstream_jobs.tsv"
    downstream_jobs = []
    if downstream_path.is_file():
        with downstream_path.open(newline="") as handle:
            downstream_jobs = list(csv.DictReader(handle, delimiter="\t"))

    return {
        "run_id": RUN_ID,
        "array_job_id": ARRAY_JOB_ID,
        "artifact_root": str(artifact_root),
        "stage2_baseline": str(artifact_root / "stage2_baseline" / "model"),
        "primary_metric": "NDCG@10",
        "smoke_run": {
            "job_id": "23688674",
            "state": "COMPLETED",
            "steps": "62/62",
            "initial_validation_reward": 0.2475000037252903,
            "final_validation_reward": 0.2425000038743019,
            "verdict": "operationally successful; scientifically inconclusive",
        },
        "failed_runs": list(FAILED_RUNS),
        "variants": variants,
        "downstream_jobs": downstream_jobs,
        "cluster": cluster_snapshot(),
    }


def write_snapshot(artifact_root: Path) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_dir = EXPERIMENT_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(artifact_root)
    (EXPERIMENT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (evidence_dir / "squeue.txt").write_text(manifest["cluster"]["squeue"])
    (evidence_dir / "sacct.txt").write_text(manifest["cluster"]["sacct"])

    for variant in VARIANTS:
        if variant.log_path.is_file():
            shutil.copy2(variant.log_path, evidence_dir / variant.log_path.name)
        if variant.slurm_path.is_file():
            shutil.copy2(variant.slurm_path, evidence_dir / variant.slurm_path.name)
        if variant.hydra_path.is_dir():
            shutil.copytree(
                variant.hydra_path,
                evidence_dir / "hydra" / variant.name,
                dirs_exist_ok=True,
            )

    report = [
        "# Office Products Ablation Run",
        "",
        "Run ledger for Slurm array `23681188` and smoke run `23688674`.",
        "",
        "## Current Verdict",
        "",
        "- Smoke run `23688674` completed 62/62 steps but changed validation reward from "
        "`0.2475` to `0.2425`; it is operationally successful and scientifically inconclusive.",
        "- Full ablation verdicts remain pending independent evaluation.",
        "",
        "## Retention Policy",
        "",
        "- Full-weight variants retain step 500.",
        "- LoRA variants retain step 200.",
        "- One merged, independently evaluated model is retained per variant.",
        "",
        "See `manifest.json` for job, metric, artifact, and verdict fields.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run --no-sync python scripts/manage_ablation_run.py snapshot",
        "sbatch --array=0-5 scripts/merge_ablation_checkpoints.sh",
        "sbatch --array=0-5 --export=MODE=common scripts/evaluate_ablation.sh",
        "sbatch --array=0-5 --export=MODE=native scripts/evaluate_ablation.sh",
        "sbatch --array=0-5 --export=MODE=reasoning_smoke scripts/evaluate_ablation.sh",
        "uv run --no-sync python analyze_ablation_results.py \\",
        "  --manifest experiments/office_products_ablation_20260612/manifest.json \\",
        f"  --artifact-root {artifact_root}",
        "```",
    ]
    (EXPERIMENT_DIR / "README.md").write_text("\n".join(report) + "\n")

    durable_evidence = artifact_root / "evidence"
    durable_evidence.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPERIMENT_DIR / "manifest.json", artifact_root / "manifest.json")
    shutil.copy2(EXPERIMENT_DIR / "README.md", artifact_root / "README.md")
    shutil.copytree(evidence_dir, durable_evidence, dirs_exist_ok=True)


def watch_and_stop(artifact_root: Path, poll_seconds: int) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    pending = {variant.name: variant for variant in VARIANTS}
    while pending:
        for name, variant in list(pending.items()):
            status = checkpoint_status(variant)
            print(
                f"{datetime.now().isoformat(timespec='seconds')} {name}: "
                f"step={status.get('latest_step')} target={variant.target_step} {status['reason']}",
                flush=True,
            )
            if not status["complete"]:
                continue

            actor_path = Path(status["actor_path"])
            archived = archive_actor(variant, actor_path, artifact_root)
            print(f"Archived {name}: {archived}", flush=True)
            run_command(["scancel", variant.job_id], check=False)
            print(f"Cancelled {variant.job_id}", flush=True)
            del pending[name]
            write_snapshot(artifact_root)
        if pending:
            time.sleep(poll_seconds)
    write_snapshot(artifact_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "snapshot", "watch"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.command == "status":
        for variant in VARIANTS:
            print(json.dumps({"variant": variant.name, **checkpoint_status(variant)}, indent=2))
    elif args.command == "snapshot":
        write_snapshot(args.artifact_root)
    else:
        watch_and_stop(args.artifact_root, args.poll_seconds)


if __name__ == "__main__":
    main()
