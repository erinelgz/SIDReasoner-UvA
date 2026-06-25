#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-rl-ablation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
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

source ./scripts/snellius_env.sh

variants=(control no_validity no_constrained exact_only lora_no_validity lora_no_constrained)
if [[ -n "${ABLATION:-}" ]]; then
    ablation="${ABLATION}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ablation="${variants[$SLURM_ARRAY_TASK_ID]}"
else
    ablation=control
fi
[[ "$ablation" == "lora" ]] && ablation=lora_no_validity

profile="${PROFILE:-full}"
stage2_checkpoint="${STAGE2_CHECKPOINT:-/home/scur1249/Office_Products_checkpoint/merged}"
checkpoint_root="${CHECKPOINT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/runs}"
sid_index_file="${SID_INDEX_FILE:-${PROJECT_DIR}/data/Amazon/index/Office_Products.index.json}"
sid_info_file="${SID_INFO_FILE:-${PROJECT_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"
source_train_file="${TRAIN_FILE:-${PROJECT_DIR}/data/Amazon/rec_reasoning_verl/Office_Products/train.parquet}"
source_val_file="${VAL_FILE:-${PROJECT_DIR}/data/Amazon/rec_reasoning_verl/Office_Products/test.parquet}"
export SID_INFO_FILE="${sid_info_file}"

reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office.py"
sid_constrained_decoding=True
actor_lr="${ACTOR_LR:-5e-7}"
lora_rank=0
lora_alpha=16
target_modules=all-linear

case "$ablation" in
    control) ;;
    no_validity)
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        ;;
    no_constrained)
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        sid_constrained_decoding=False
        ;;
    exact_only)
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_exact_only.py"
        ;;
    lora_no_validity)
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        actor_lr="${ACTOR_LR:-5e-6}"
        lora_rank="${LORA_RANK:-32}"
        lora_alpha="${LORA_ALPHA:-64}"
        target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
        ;;
    lora_no_constrained)
        reward_path="${PROJECT_DIR}/verl/utils/reward_score/direct_recommendation_StepRule_Office_no_validity.py"
        sid_constrained_decoding=False
        actor_lr="${ACTOR_LR:-5e-6}"
        lora_rank="${LORA_RANK:-32}"
        lora_alpha="${LORA_ALPHA:-64}"
        target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
        ;;
    *)
        echo "Unknown ABLATION='$ablation'. Choose: ${variants[*]} (or lora alias)."
        exit 2
        ;;
esac

case "$profile" in
    full)
        n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
        train_batch_size="${TRAIN_BATCH_SIZE:-256}"
        prompt_length="${MAX_PROMPT_LENGTH:-1024}"
        response_length="${MAX_RESPONSE_LENGTH:-1024}"
        ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-256}"
        micro_batch_size="${MICRO_BATCH_SIZE:-8}"
        rollout_n="${ROLLOUT_N:-16}"
        rollout_memory="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
        rollout_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
        rollout_seqs="${ROLLOUT_MAX_NUM_SEQS:-64}"
        total_epochs="${TOTAL_EPOCHS:-10}"
        save_freq="${SAVE_FREQ:-100}"
        test_freq="${TEST_FREQ:-50}"
        train_file="$source_train_file"
        val_file="$source_val_file"
        ;;
    smoke)
        n_gpus_per_node="${N_GPUS_PER_NODE:-1}"
        train_batch_size="${TRAIN_BATCH_SIZE:-32}"
        prompt_length="${MAX_PROMPT_LENGTH:-256}"
        response_length="${MAX_RESPONSE_LENGTH:-256}"
        ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}"
        micro_batch_size="${MICRO_BATCH_SIZE:-4}"
        rollout_n="${ROLLOUT_N:-4}"
        rollout_memory="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}"
        rollout_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
        rollout_seqs="${ROLLOUT_MAX_NUM_SEQS:-8}"
        total_epochs="${TOTAL_EPOCHS:-2}"
        save_freq="${SAVE_FREQ:-31}"
        test_freq="${TEST_FREQ:-1}"
        sample_seed="${SEED:-42}"
        sample_dir="${SMALL_DATA_DIR:-${PROJECT_DIR}/temp/rl_ablation_small_data}"
        train_samples="${SMALL_TRAIN_SAMPLES:-1000}"
        val_samples="${SMALL_VAL_SAMPLES:-100}"
        mkdir -p "$sample_dir"
        train_file="${sample_dir}/train_${train_samples}_seed${sample_seed}.parquet"
        val_file="${sample_dir}/val_${val_samples}_seed${sample_seed}.parquet"
        ${PYTHON_CMD} - "$source_train_file" "$source_val_file" "$train_file" "$val_file" \
            "$train_samples" "$val_samples" "$sample_seed" <<'PY'
import sys
import pandas as pd

src_train, src_val, out_train, out_val, n_train, n_val, seed = sys.argv[1:]
seed = int(seed)
for src, out, count in ((src_train, out_train, n_train), (src_val, out_val, n_val)):
    frame = pd.read_parquet(src)
    count = int(count)
    if count >= 0:
        frame = frame.sample(n=min(count, len(frame)), random_state=seed)
    frame.to_parquet(out, index=False)
    print(f"Wrote {len(frame)} rows to {out}")
PY
        ;;
    *)
        echo "PROFILE must be full or smoke, got '$profile'."
        exit 2
        ;;
esac

experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_${ablation}_Qwen3-1.7B_${profile}}"
checkpoint_dir="${checkpoint_root}/${experiment_name}"
log_file="${LOG_FILE:-${PROJECT_DIR}/logs/${experiment_name}.log}"
mkdir -p "$(dirname "$log_file")" "$checkpoint_dir"

hydra_args=(
    algorithm.adv_estimator=grpo
    data.train_files="${train_file}"
    data.val_files="${val_file}"
    data.train_batch_size="${train_batch_size}"
    data.max_prompt_length="${prompt_length}"
    data.max_response_length="${response_length}"
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="${stage2_checkpoint}"
    actor_rollout_ref.actor.optim.lr="${actor_lr}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.lora_rank="${lora_rank}"
    actor_rollout_ref.model.lora_alpha="${lora_alpha}"
    actor_rollout_ref.model.target_modules="${target_modules}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${micro_batch_size}"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_memory}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_tokens}"
    actor_rollout_ref.rollout.max_num_seqs="${rollout_seqs}"
    actor_rollout_ref.rollout.n="${rollout_n}"
    actor_rollout_ref.rollout.sid_constrained_decoding="${sid_constrained_decoding}"
    actor_rollout_ref.rollout.sid_index_file="${sid_index_file}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${micro_batch_size}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    "trainer.logger=['console']"
    custom_reward_function.path="${reward_path}"
    custom_reward_function.name=rule_base_reward
    trainer.project_name=SIDReasoner
    trainer.experiment_name="${experiment_name}"
    trainer.n_gpus_per_node="${n_gpus_per_node}"
    trainer.nnodes="${NNODES:-1}"
    trainer.default_local_dir="${checkpoint_dir}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
    trainer.resume_mode="${RESUME_MODE:-disable}"
    trainer.save_freq="${save_freq}"
    trainer.test_freq="${test_freq}"
    trainer.total_epochs="${total_epochs}"
)

{
    printf 'Ablation: %s\nProfile: %s\nExperiment: %s\nCheckpoint dir: %s\n' \
        "$ablation" "$profile" "$experiment_name" "$checkpoint_dir"
    ${PYTHON_CMD} -m verl.trainer.main_ppo "${hydra_args[@]}" "$@"
} >"$log_file" 2>&1
