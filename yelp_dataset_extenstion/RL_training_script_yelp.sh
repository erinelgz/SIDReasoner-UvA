#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-yelp-stage3-rl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=18:00:00
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --chdir=/gpfs/home2/scur1223/SIDReasoner-UvA

set -euo pipefail
set -x

source ./scripts/snellius_environment.sh
SCRIPT_DIR="$(pwd)"
export SIDREASONER_PROJECT_DIR="${SIDREASONER_PROJECT_DIR:-${SCRIPT_DIR}}"
export SIDREASONER_YELP_INFO_FILE="${SCRIPT_DIR}/data/Yelp/info/Yelp_Restaurants_5core.txt"
export WANDB_MODE="${WANDB_MODE:-disabled}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

CATEGORY="${CATEGORY:-Yelp_Restaurants}"
DATASET="${DATASET:-Yelp_Restaurants_5core}"
n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
nnodes="${NNODES:-1}"
experiment_name="${EXPERIMENT_NAME:-${CATEGORY}_stage3_rl_Qwen3-1.7B}"
stage2_checkpoint="${STAGE2_CHECKPOINT:-${SCRIPT_DIR}/output_dir/${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint}"
train_files="${TRAIN_FILES:-${SCRIPT_DIR}/data/Yelp/rec_reasoning_verl/Yelp_Restaurants/train.parquet}"
val_files="${VAL_FILES:-${SCRIPT_DIR}/data/Yelp/rec_reasoning_verl/Yelp_Restaurants/test.parquet}"
checkpoint_dir="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/RecRL_Reasoning/${experiment_name}}"
reward_file="${REWARD_FILE:-${SCRIPT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Yelp.py}"
trainer_logger="${TRAINER_LOGGER:-['console']}"
log_file="${LOG_FILE:-./logs/${experiment_name}.log}"

mkdir -p ./logs "${checkpoint_dir}" slurm_output

{
${PYTHON_CMD} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${stage2_checkpoint}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${trainer_logger}" \
    custom_reward_function.path="${reward_file}" \
    custom_reward_function.name="rule_base_reward" \
    trainer.project_name='RecRL_Reasoning' \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 "$@"
} > "${log_file}" 2>&1
