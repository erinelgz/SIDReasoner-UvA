# SIDReasoner

This repository contains the code for **"SIDReasoner - Reasoning over Semantic IDs Enhances Generative Recommendation"**.

SIDReasoner is a generative recommendation framework that strengthens recommendation models with reasoning over semantic IDs. The repository provides the training scripts, evaluation scripts, data download workflow, and a Snellius-ready environment setup.

## Snellius Quick Start

Run these commands from a clean checkout.

```bash
git clone <repo-url>
cd SIDReasoner
```

Create the Python environment on a compute node:

```bash
sbatch scripts/setup_uv_environment.sh
```

Download and extract the dataset:

```bash
make data
```

Run the three training stages in order. Submit the next job only after the previous one has finished.

```bash
sbatch scripts/train_sft_qwen3_enrich.sh
sbatch scripts/train_sft_reasoning_activation.sh
sbatch scripts/train_rl.sh
```

Merge the RL checkpoint before thinking-mode evaluation. Adjust `global_step_100` if you want a different checkpoint.

```bash
sbatch scripts/merge_fsdp_checkpoint.sh ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

Run evaluation:

```bash
sbatch scripts/evaluate_qwen3.sh
sbatch scripts/evaluate_qwen3_think.sh
```

Useful monitoring commands:

```bash
squeue -u $USER
tail -f slurm_output/*.out
tail -f logs/*.txt
tail -f logs/*.log
```

## Environment

This repository uses `uv` for dependency management. The Python dependencies and linting tools are declared in `pyproject.toml`.

The base runtime stack is pinned for CUDA 12.4, PyTorch 2.6, vLLM 0.8.5, FlashAttention 2.7.4, and FlashInfer 0.2.2. The cuDNN override from the original VERL setup is installed after uv resolves the base environment because Torch pins a different cuDNN package in its dependency metadata.

The Slurm scripts load Snellius modules through `scripts/snellius_environment.sh`. The setup job uses the Snellius `2023` module stack with `CUDA/12.4.0` and creates `.venv` with Python 3.10 through uv.

Optional SGLang support:

```bash
sbatch scripts/setup_uv_environment.sh --sglang
```

Optional Megatron/TransformerEngine support:

```bash
sbatch scripts/setup_uv_environment.sh --megatron
```

Install both optional stacks:

```bash
sbatch scripts/setup_uv_environment.sh --all
```

## Dataset

Download and extract the dataset with:

```bash
make data
```

This downloads the Google Drive dataset and places it under:

```text
data/Amazon
```

To use a different Google Drive file ID or archive name:

```bash
make data DATA_FILE_ID="..." DATA_ARCHIVE="data/my_dataset.zip"
```

## Training

SIDReasoner follows a three-stage training pipeline.

| Stage | Script |
| --- | --- |
| Stage 1: Supervised Fine-Tuning | `scripts/train_sft_qwen3_enrich.sh` |
| Stage 2: Reasoning Activation | `scripts/train_sft_reasoning_activation.sh` |
| Stage 3: RL Training | `scripts/train_rl.sh` |

Run on Snellius:

```bash
sbatch scripts/train_sft_qwen3_enrich.sh
sbatch scripts/train_sft_reasoning_activation.sh
sbatch scripts/train_rl.sh
```

The scripts write Slurm output to `slurm_output/` and training logs to `logs/`.

Common overrides:

```bash
CATEGORY=Office_Products CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 sbatch scripts/train_sft_qwen3_enrich.sh
N_GPUS_PER_NODE=4 NNODES=1 sbatch scripts/train_rl.sh
```

## Evaluation

Evaluate non-thinking and thinking modes:

```bash
sbatch scripts/evaluate_qwen3.sh
sbatch scripts/evaluate_qwen3_think.sh
```

Override GPU splits:

```bash
CUDA_LIST="0 1" CUDA_LIST_CSV="0,1" sbatch scripts/evaluate_qwen3_think.sh
```

The thinking-mode evaluation expects a merged Hugging Face checkpoint named `actor_merged`. If RL training only produced raw `actor` folders, merge one first:

```bash
sbatch scripts/merge_fsdp_checkpoint.sh ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

## Extensions

In addition to the core SIDReasoner reproduction, this repository contains the extensions described in the accompanying report. These extensions are intended to make the release self-contained for comparison, ablation, and transfer experiments.

### SASRec Baseline

The `sasrec_baseline/` directory contains the discriminative SASRec baseline used for comparison with SIDReasoner.

| File | Purpose |
| --- | --- |
| `sasrec_baseline/convert_data.py` | Converts the SIDReasoner data format for SASRec training. |
| `sasrec_baseline/main_torch.py` | Trains and evaluates the PyTorch SASRec baseline. |
| `sasrec_baseline/sasrec_ce_full.job` | Slurm job for the full SASRec cross-entropy baseline run. |

Run the baseline job from the `sasrec_baseline/` directory:

```bash
cd sasrec_baseline
sbatch sasrec_ce_full.job
```

### Yelp Dataset Extension

The `yelp_dataset_extenstion/` directory contains the Yelp Restaurants transfer experiment. This extension prepares a sampled Yelp Restaurants domain and runs the same SIDReasoner stages used for the Amazon reproduction.

| Stage | Script |
| --- | --- |
| Prepare Yelp data | `yelp_dataset_extenstion/prepare_yelp_data.sh` |
| Stage 1: Supervised Fine-Tuning | `yelp_dataset_extenstion/sft_Qwen3_enrich_yelp.sh` |
| Stage 2: Reasoning Activation | `yelp_dataset_extenstion/sft_reasoning_activation_yelp.sh` |
| Stage 3: RL Training | `yelp_dataset_extenstion/RL_training_script_yelp.sh` |
| Standard evaluation | `yelp_dataset_extenstion/evaluate_Qwen3_yelp.sh` |
| Thinking-mode evaluation | `yelp_dataset_extenstion/evaluate_Qwen3_think_yelp.sh` |

The Yelp preparation script expects the Yelp Open Dataset archive path through `TAR_PATH`:

```bash
TAR_PATH=/path/to/yelp_dataset.tar sbatch yelp_dataset_extenstion/prepare_yelp_data.sh
```

Then run the Yelp training and evaluation scripts in the same stage order as the Amazon pipeline.

### Further Extensions

The `scripts/` directory also includes the Office Products extension experiments from the report. These scripts probe reward design, rollout constraints, LoRA updates, inference-time reasoning, and adaptive reasoning behavior without changing the main three-stage training pipeline.

| Extension | Scripts |
| --- | --- |
| RL reward, rollout, and LoRA ablations | `scripts/train_rl_ablation.sh`, `scripts/train_rl_ablation_small_data.sh`, `scripts/train_rl_ablation_legacy.sh` |
| Ablation checkpoint merging and evaluation | `scripts/merge_ablation_checkpoints.sh`, `scripts/evaluate_ablation.sh`, `scripts/manage_ablation_run.py` |
| Inference-time reasoning comparisons | `scripts/evaluate_final_office.sh`, `scripts/evaluate_author_final_baseline.sh`, `scripts/analyze_final_office.sh`, `scripts/analyze_author_final_baseline.sh` |
| Adaptive reasoning analysis | `scripts/evaluate_adaptive_reasoning.sh`, `scripts/analyze_adaptive_reasoning.sh` |
| Aligned evaluation checks | `scripts/evaluate_aligned.sh` |

These scripts are primarily research and report-reproduction utilities. They assume the same Snellius environment setup as the main pipeline and may require experiment-specific checkpoint paths through environment-variable overrides such as `ARTIFACT_ROOT`, `STAGE2_CHECKPOINT`, `CHECKPOINT_ROOT`, or `RESULT_ROOT`.

## Development

Useful Make targets:

```bash
make data        # download and extract the dataset
make lint        # run ruff checks
make format      # format Python files with ruff
make precommit   # run all pre-commit hooks
```

Install development hooks:

```bash
make install-dev
```

## Checkpoints

Pretrained model checkpoints are available on Hugging Face:

https://huggingface.co/Sober-Clever/SIDReasoner-Models/tree/main

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{SIDReasoner,
  title={Reasoning over Semantic IDs Enhances Generative Recommendation},
  author={Yingzhi He and Yan Sun and Junfei Tan and Yuxin Chen and Xiaoyu Kong and Chunxu Shen and Xiang Wang and An Zhang and Tat-Seng Chua},
  journal={arXiv preprint arXiv:2603.23183},
  year={2026}
}
```

## Acknowledgement

This repo is built upon [MiniOneRec](https://github.com/AkaliKong/MiniOneRec).
