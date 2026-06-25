#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-setup-env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x-%j.out


set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

WITH_MEGATRON=0
WITH_SGLANG=0
for arg in "$@"; do
    case "$arg" in
        --megatron)
            WITH_MEGATRON=1
            ;;
        --sglang)
            WITH_SGLANG=1
            ;;
        --all)
            WITH_MEGATRON=1
            WITH_SGLANG=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: sbatch scripts/setup_uv_environment.sh [--sglang] [--megatron] [--all]"
            exit 2
            ;;
    esac
done

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
module purge
module load 2023
module load CUDA/12.4.0

# Install uv if not already present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PWD}/.venv}"
export MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p "${UV_CACHE_DIR}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Python requested by uv: 3.10"

uv sync --python 3.10

echo "Installing cuDNN runtime override used by the original VERL setup"
uv pip install --no-deps "nvidia-cudnn-cu12==9.8.0.87"

if [[ "${WITH_SGLANG}" == "1" ]]; then
    echo "Installing optional SGLang dependency without letting it override the pinned FlashInfer runtime"
    uv pip install --no-deps "sglang[all]==0.4.6.post1"
fi

if [[ "${WITH_MEGATRON}" == "1" ]]; then
    echo "Installing optional Megatron and TransformerEngine dependencies"
    uv pip install pip setuptools wheel
    NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1"
    uv pip install --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2"
fi
