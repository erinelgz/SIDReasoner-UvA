#!/bin/bash
# Download the author-released Office Products RL checkpoint used by the
# post-freeze three-model baseline comparison.

set -euo pipefail

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/author_final_baseline_test_20260623}"
repo="Sober-Clever/Qwen3-1.7B_base_e2e-AmazonMix3-EP2_General_reasoning-activate-ep1_RLonOffice_ckpt1000"
revision="749c36abddf4b40cfedb407279a8ff77324981cb"
archive_name="Qwen3-1.7B_base_e2e-AmazonMix3-EP2_General_reasoning-activate-ep1_RLonOffice_ckpt1000.zip"
archive_sha256="d26a02f324d6b411ddb6d30127d9161c3a3559899468e83e49d393ecfd4a8421"
archive_path="${artifact_root}/${archive_name}"
extract_root="${artifact_root}/author_stage3_extracted"
model_link="${artifact_root}/author_stage3_model"
url="https://huggingface.co/${repo}/resolve/${revision}/${archive_name}"

mkdir -p "$artifact_root"

if [[ ! -f "$archive_path" ]]; then
    curl --fail --location --retry 3 --continue-at - --output "$archive_path" "$url"
fi

actual_archive_sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
[[ "$actual_archive_sha256" == "$archive_sha256" ]] || {
    echo "Unexpected archive checksum: $actual_archive_sha256" >&2
    exit 2
}

if [[ ! -d "$extract_root" ]]; then
    mkdir -p "$extract_root"
    unzip -q "$archive_path" -d "$extract_root"
fi

mapfile -t model_files < <(find "$extract_root" -type f -name model.safetensors -print)
[[ "${#model_files[@]}" == "1" ]] || {
    printf 'Expected exactly one model.safetensors under %s; found %s\n' "$extract_root" "${#model_files[@]}" >&2
    printf '%s\n' "${model_files[@]}" >&2
    exit 3
}

model_dir="$(dirname "${model_files[0]}")"
[[ -f "${model_dir}/config.json" ]] || {
    echo "Model directory does not contain config.json: $model_dir" >&2
    exit 4
}
ln -sfn "$model_dir" "$model_link"

model_sha256="$(sha256sum "${model_dir}/model.safetensors" | awk '{print $1}')"
python3 - "$artifact_root/provenance.json" "$repo" "$revision" "$url" "$archive_name" "$archive_sha256" "$archive_path" "$model_dir" "$model_sha256" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    repo,
    revision,
    url,
    archive_name,
    archive_sha256,
    archive_path,
    model_dir,
    model_sha256,
) = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "source_assumption": "Author-released Office RL ckpt1000; accepted without additional Stage-2 verification.",
            "huggingface_repo": repo,
            "huggingface_revision": revision,
            "download_url": url,
            "archive_name": archive_name,
            "archive_sha256": archive_sha256,
            "archive_path": archive_path,
            "model_path": model_dir,
            "model_safetensors_sha256": model_sha256,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )
    + "\n"
)
PY

printf 'Author model ready at: %s\n' "$model_link"
printf 'Model SHA-256: %s\n' "$model_sha256"
