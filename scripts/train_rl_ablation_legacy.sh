#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-rl-ablation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --array=0-5
#SBATCH --output=slurm_output/%x-%A_%a.out

set -euo pipefail
set -x

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"
PROJECT_DIR="$(pwd -P)"

source ./scripts/snellius_environment.sh

n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
nnodes="${NNODES:-1}"
ablations=(control no_validity no_constrained exact_only lora lora_no_constrained)
if [[ -n "${ABLATION:-}" ]]; then
    ablation="${ABLATION}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ablation="${ablations[$SLURM_ARRAY_TASK_ID]}"
else
    ablation=control
fi
stage2_checkpoint="${STAGE2_CHECKPOINT:-/home/scur1249/Office_Products_checkpoint/merged}"
sid_index_file="${SID_INDEX_FILE:-${PROJECT_DIR}/data/Amazon/index/Office_Products.index.json}"
sid_info_file="${SID_INFO_FILE:-${PROJECT_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"
export SID_INFO_FILE="${sid_info_file}"

train_file="${TRAIN_FILE:-${PROJECT_DIR}/data/Amazon/rec_reasoning_verl/Office_Products/train.parquet}"
val_file="${VAL_FILE:-${PROJECT_DIR}/data/Amazon/rec_reasoning_verl/Office_Products/test.parquet}"
reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office.py"
sid_constrained_decoding=True
actor_lr="5e-7"
lora_rank=0
lora_alpha=16
target_modules="all-linear"

case "$ablation" in
    control)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_control_Qwen3-1.7B}"
        ;;
    no_validity)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_no_validity_Qwen3-1.7B}"
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        ;;
    no_constrained)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_no_constrained_Qwen3-1.7B}"
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        sid_constrained_decoding=False
        ;;
    exact_only)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_exact_only_Qwen3-1.7B}"
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_exact_only.py"
        ;;
    lora)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_lora_no_validity_Qwen3-1.7B}"
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        actor_lr="${ACTOR_LR:-5e-6}"
        lora_rank="${LORA_RANK:-32}"
        lora_alpha="${LORA_ALPHA:-64}"
        target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
        ;;
    lora_no_constrained)
        experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_lora_no_constrained_Qwen3-1.7B}"
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        sid_constrained_decoding=False
        actor_lr="${ACTOR_LR:-5e-6}"
        lora_rank="${LORA_RANK:-32}"
        lora_alpha="${LORA_ALPHA:-64}"
        target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
        ;;
    *)
        echo "Unknown ABLATION='$ablation'"
        echo "Choose one of: control, no_validity, no_constrained, exact_only, lora, lora_no_constrained"
        exit 2
        ;;
esac

log_file="${LOG_FILE:-./logs/${experiment_name}.log}"
mkdir -p ./logs
trainer_logger="${TRAINER_LOGGER:-['console']}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-64}"

hydra_args=(
    algorithm.adv_estimator=grpo
    data.train_files="${train_file}"
    data.val_files="${val_file}"
    data.train_batch_size=256
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="${stage2_checkpoint}"
    actor_rollout_ref.actor.optim.lr="${actor_lr}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.lora_rank="${lora_rank}"
    actor_rollout_ref.model.lora_alpha="${lora_alpha}"
    actor_rollout_ref.model.target_modules="${target_modules}"
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_max_num_batched_tokens}"
    actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}"
    actor_rollout_ref.rollout.n=16
    actor_rollout_ref.rollout.sid_constrained_decoding="${sid_constrained_decoding}"
    actor_rollout_ref.rollout.sid_index_file="${sid_index_file}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    "trainer.logger=${trainer_logger}"
    custom_reward_function.path="${reward_path}"
    custom_reward_function.name=rule_base_reward
    trainer.project_name=RecRL_Reasoning
    trainer.experiment_name="${experiment_name}"
    trainer.n_gpus_per_node="${n_gpus_per_node}"
    trainer.nnodes="${nnodes}"
    trainer.save_freq="${SAVE_FREQ:-100}"
    trainer.test_freq="${TEST_FREQ:-50}"
    trainer.total_epochs="${TOTAL_EPOCHS:-10}"
)

{
    echo "Ablation: ${ablation}"
    echo "Experiment: ${experiment_name}"
    echo "Reward: ${reward_path}"
    echo "Constrained rollout: ${sid_constrained_decoding}"
    echo "Actor LR: ${actor_lr}"
    echo "LoRA rank: ${lora_rank}"
    echo "LoRA alpha: ${lora_alpha}"
    echo "Target modules: ${target_modules}"
    echo "Rollout GPU memory utilization: ${rollout_gpu_memory_utilization}"
    echo "Rollout max num batched tokens: ${rollout_max_num_batched_tokens}"
    echo "Rollout max num seqs: ${rollout_max_num_seqs}"
    echo "Train file: ${train_file}"
    echo "Val file: ${val_file}"
    echo "SID index file: ${sid_index_file}"
    echo "SID info file: ${sid_info_file}"
    ${PYTHON_CMD} -m verl.trainer.main_ppo "${hydra_args[@]}" "$@"
} > "${log_file}" 2>&1
