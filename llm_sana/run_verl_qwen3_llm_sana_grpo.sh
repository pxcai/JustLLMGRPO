#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 || true
ulimit -u 65535 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PY=${PY:-python}
LLM_SANA_DIR=${LLM_SANA_DIR:-${ROOT}/llm_sana}
VERL_DIR=${VERL_DIR:-${ROOT}/third_party/verl}

csv_count() {
  local csv="${1:-}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F, '{n=0; for (i=1; i<=NF; i++) {gsub(/^[ \t]+|[ \t]+$/, "", $i); if ($i != "") n++} print n}' <<<"${csv}"
}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}
SANA_MODEL_PATH=${SANA_MODEL_PATH:-raman07/CheXGenBench-Models-Sana-e20}
TRAIN_CSV=${TRAIN_CSV:-${ROOT}/data/LLAVARAD_ANNOTATIONS_TRAIN.csv}
VAL_CSV=${VAL_CSV:-${ROOT}/data/LLAVARAD_ANNOTATIONS_TEST.csv}
DATA_DIR=${DATA_DIR:-${ROOT}/data/processed/llavarad_prompt_rewrite}
PROMPT_COL=${PROMPT_COL:-annotated_prompt}
LABELS_COL=${LABELS_COL:-chexpert_labels}
BALANCED_VAL_PER_LABEL=${BALANCED_VAL_PER_LABEL:-40}
REBUILD_DATA=${REBUILD_DATA:-1}

REWARD_MODE=${REWARD_MODE:-biovil_label_raddino}
BIOVIL_T_PATH=${BIOVIL_T_PATH:-microsoft/BiomedVLP-BioViL-T}
RAD_DINO_PATH=${RAD_DINO_PATH:-microsoft/rad-dino}
CXR_CLASSIFIER_CHECKPOINT=${CXR_CLASSIFIER_CHECKPOINT:-${ROOT}/artifacts/best_classifier.pt}
RADDINO_REFERENCE_CACHE=${RADDINO_REFERENCE_CACHE:-${ROOT}/artifacts/raddino_train20k_ref.npz}
BIOVIL_WEIGHT=${BIOVIL_WEIGHT:-0.45}
LABEL_WEIGHT=${LABEL_WEIGHT:-0.1}
RADDINO_WEIGHT=${RADDINO_WEIGHT:-0.45}
REWARD_DEVICE=${REWARD_DEVICE:-cuda}
LLM_CUDA_VISIBLE_DEVICES=${LLM_CUDA_VISIBLE_DEVICES:-0,1}
REWARD_CUDA_VISIBLE_DEVICES=${REWARD_CUDA_VISIBLE_DEVICES:-2,3,4}
LLM_NUM_GPUS_DEFAULT=$(csv_count "${LLM_CUDA_VISIBLE_DEVICES}")
REWARD_NUM_GPUS_DEFAULT=$(csv_count "${REWARD_CUDA_VISIBLE_DEVICES}")
SANA_MAX_CONCURRENT=${SANA_MAX_CONCURRENT:-${REWARD_NUM_GPUS_DEFAULT}}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-${SANA_MAX_CONCURRENT}}
REWARD_USE_DISPATCHER=${REWARD_USE_DISPATCHER:-1}
SANA_BATCH_SIZE=${SANA_BATCH_SIZE:-32}
SANA_BATCH_MAX_WAIT_MS=${SANA_BATCH_MAX_WAIT_MS:-50}
REWARD_GPU_WORKER_MAX_CONCURRENCY=${REWARD_GPU_WORKER_MAX_CONCURRENCY:-$((SANA_BATCH_SIZE * 4))}
REWARD_PRINT_DENOMINATOR=${REWARD_PRINT_DENOMINATOR:-32}
REWARD_MAX_PRINT_CHARS=${REWARD_MAX_PRINT_CHARS:-2000}
DEBUG_IMAGE_DIR=${DEBUG_IMAGE_DIR:-}

SANA_HEIGHT=${SANA_HEIGHT:-512}
SANA_WIDTH=${SANA_WIDTH:-512}
SANA_STEPS=${SANA_STEPS:-20}
SANA_GUIDANCE_SCALE=${SANA_GUIDANCE_SCALE:-4.5}
SEED=${SEED:-42}

RUN_TAG=${RUN_TAG:-justllmgrpo_qwen3_4b}
PROJECT_NAME=${PROJECT_NAME:-justllmgrpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_prompt_policy_${REWARD_MODE}}
RUN_DIR=${RUN_DIR:-${ROOT}/outputs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}

N_GPUS=${N_GPUS:-${LLM_NUM_GPUS_DEFAULT}}
TP_SIZE=${TP_SIZE:-${N_GPUS}}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING:-False}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-768}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-$((14 * BALANCED_VAL_PER_LABEL))}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2816}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
VAL_ROLLOUT_TEMPERATURE=${VAL_ROLLOUT_TEMPERATURE:-0}
VAL_ROLLOUT_TOP_P=${VAL_ROLLOUT_TOP_P:-1.0}
VAL_ROLLOUT_TOP_K=${VAL_ROLLOUT_TOP_K:--1}
VAL_ROLLOUT_DO_SAMPLE=${VAL_ROLLOUT_DO_SAMPLE:-False}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-8}

ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-True}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.001}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
ACTOR_FSDP_MODEL_DTYPE=${ACTOR_FSDP_MODEL_DTYPE:-bf16}
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-False}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ=${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-False}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
REF_LOG_PROB_USE_DYNAMIC_BSZ=${REF_LOG_PROB_USE_DYNAMIC_BSZ:-False}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
REF_USE_TORCH_COMPILE=${REF_USE_TORCH_COMPILE:-False}
REF_FSDP_PARAM_OFFLOAD=${REF_FSDP_PARAM_OFFLOAD:-True}

TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","file"]'}
SAVE_FREQ=${SAVE_FREQ:-400}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-4}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VERL_ENTRYPOINT=${VERL_ENTRYPOINT:-verl.trainer.main_ppo}

export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_llm_sana_grpo_${USER:-user}_$$}
export VERL_RAY_USAGE_STATS_ONLY=${VERL_RAY_USAGE_STATS_ONLY:-1}
export PYTHONPATH="${ROOT}:${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export LLMSANA_LOCAL_FILES_ONLY=${LLMSANA_LOCAL_FILES_ONLY:-0}
export DISABLE_XFORMERS=${DISABLE_XFORMERS:-1}
export LLMSANA_REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES}"
export LLMSANA_REWARD_USE_DISPATCHER="${REWARD_USE_DISPATCHER}"
export LLMSANA_SANA_BATCH_SIZE="${SANA_BATCH_SIZE}"
export LLMSANA_SANA_BATCH_MAX_WAIT_MS="${SANA_BATCH_MAX_WAIT_MS}"
export LLMSANA_REWARD_GPU_WORKER_MAX_CONCURRENCY="${REWARD_GPU_WORKER_MAX_CONCURRENCY}"
export CUDA_VISIBLE_DEVICES="${LLM_CUDA_VISIBLE_DEVICES}"

mkdir -p "${DATA_DIR}" "${RUN_DIR}" "${ROOT}/outputs"
export VERL_FILE_LOGGER_PATH="${RUN_DIR}/metrics.jsonl"

echo "==> Preflight"
PREFLIGHT_ARGS=(
  --model "${MODEL_PATH}" \
  --sana-model "${SANA_MODEL_PATH}" \
  --reward-mode "${REWARD_MODE}" \
  --biovil-path "${BIOVIL_T_PATH}" \
  --classifier-checkpoint "${CXR_CLASSIFIER_CHECKPOINT}" \
  --raddino-path "${RAD_DINO_PATH}" \
  --raddino-cache "${RADDINO_REFERENCE_CACHE}" \
  --rollout-backend "${ROLLOUT_BACKEND}"
)
if [[ "${LLMSANA_LOCAL_FILES_ONLY}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--local-files-only)
fi
"${PY}" "${LLM_SANA_DIR}/check_llm_sana_grpo_env.py" "${PREFLIGHT_ARGS[@]}"

if [[ "${REBUILD_DATA}" == "1" || ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "==> Prepare LLaVA-Rad prompt parquet data"
  PREP_ARGS=(
    --train_csv "${TRAIN_CSV}"
    --output_dir "${DATA_DIR}"
    --prompt_col "${PROMPT_COL}"
    --labels_col "${LABELS_COL}"
    --max_train_samples "${TRAIN_MAX_SAMPLES}"
    --max_val_samples "${VAL_MAX_SAMPLES}"
    --balanced_val_per_label "${BALANCED_VAL_PER_LABEL}"
    --seed "${SEED}"
  )
  if [[ -n "${VAL_CSV}" ]]; then
    PREP_ARGS+=(--val_csv "${VAL_CSV}")
  fi
  "${PY}" "${LLM_SANA_DIR}/data/prepare_llavarad_prompt_parquet.py" "${PREP_ARGS[@]}"
fi

DATA_TRAIN_ROWS=$("${PY}" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "${DATA_DIR}/train.parquet")
DATA_VAL_ROWS=$("${PY}" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "${DATA_DIR}/val.parquet")
if [[ "${TRAIN_MAX_SAMPLES}" -gt 0 && "${TRAIN_MAX_SAMPLES}" -lt "${DATA_TRAIN_ROWS}" ]]; then
  EFFECTIVE_TRAIN_SAMPLES="${TRAIN_MAX_SAMPLES}"
  TRAIN_MAX_SAMPLES_FOR_VERL="${TRAIN_MAX_SAMPLES}"
else
  EFFECTIVE_TRAIN_SAMPLES="${DATA_TRAIN_ROWS}"
  TRAIN_MAX_SAMPLES_FOR_VERL=-1
fi
if [[ "${VAL_MAX_SAMPLES}" -gt 0 && "${VAL_MAX_SAMPLES}" -lt "${DATA_VAL_ROWS}" ]]; then
  VAL_MAX_SAMPLES_FOR_VERL="${VAL_MAX_SAMPLES}"
else
  VAL_MAX_SAMPLES_FOR_VERL=-1
fi
TOTAL_TRAINING_STEPS_DEFAULT=$(((EFFECTIVE_TRAIN_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}

echo "==> Run Qwen3 LLM-Sana GRPO"
echo "model=${MODEL_PATH}"
echo "sana_model=${SANA_MODEL_PATH}"
echo "data=${DATA_DIR}"
echo "reward_mode=${REWARD_MODE}"
echo "reward_weights=biovil:${BIOVIL_WEIGHT},label:${LABEL_WEIGHT},raddino:${RADDINO_WEIGHT}"
echo "llm_cuda_visible_devices=${LLM_CUDA_VISIBLE_DEVICES}; n_gpus=${N_GPUS}; tp_size=${TP_SIZE}"
echo "reward_device=${REWARD_DEVICE}; reward_cuda_visible_devices=${REWARD_CUDA_VISIBLE_DEVICES}"
echo "sana_max_concurrent=${SANA_MAX_CONCURRENT}; reward_num_workers=${REWARD_NUM_WORKERS}; reward_use_dispatcher=${REWARD_USE_DISPATCHER}"
echo "sana_batch_size=${SANA_BATCH_SIZE}; sana_batch_max_wait_ms=${SANA_BATCH_MAX_WAIT_MS}; reward_gpu_worker_max_concurrency=${REWARD_GPU_WORKER_MAX_CONCURRENCY}"
echo "rollout_temperature=${ROLLOUT_TEMPERATURE}; rollout_top_p=${ROLLOUT_TOP_P}; rollout_top_k=${ROLLOUT_TOP_K}; max_response_length=${MAX_RESPONSE_LENGTH}; max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
echo "train_batch_size=${TRAIN_BATCH_SIZE}; val_batch_size=${VAL_BATCH_SIZE}; ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}; rollout_n=${ROLLOUT_N}; micro_batch_size=${MICRO_BATCH_SIZE}"
echo "val_rollout_do_sample=${VAL_ROLLOUT_DO_SAMPLE}; val_rollout_temperature=${VAL_ROLLOUT_TEMPERATURE}; val_rollout_top_p=${VAL_ROLLOUT_TOP_P}; val_rollout_top_k=${VAL_ROLLOUT_TOP_K}"
echo "output=${RUN_DIR}"
echo "data_rows_train=${DATA_TRAIN_ROWS}; data_rows_val=${DATA_VAL_ROWS}; balanced_val_per_label=${BALANCED_VAL_PER_LABEL}"
echo "train_max_samples=${TRAIN_MAX_SAMPLES_FOR_VERL}; val_max_samples=${VAL_MAX_SAMPLES_FOR_VERL}; total_training_steps=${TOTAL_TRAINING_STEPS}"

cd "${VERL_DIR}"
"${PY}" -m "${VERL_ENTRYPOINT}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES_FOR_VERL}" \
  data.val_max_samples="${VAL_MAX_SAMPLES_FOR_VERL}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  data.dataloader_num_workers=0 \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="${MODEL_USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_FSDP_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_ROLLOUT_TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_ROLLOUT_DO_SAMPLE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${REF_LOG_PROB_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.use_torch_compile="${REF_USE_TORCH_COMPILE}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${ACTOR_FSDP_MODEL_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile="${REF_USE_TORCH_COMPILE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_FSDP_PARAM_OFFLOAD}" \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  reward.custom_reward_function.path="${LLM_SANA_DIR}/rewards/llm_sana_online_reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.sana_model_path="${SANA_MODEL_PATH}" \
  +reward.custom_reward_function.reward_kwargs.reward_mode="${REWARD_MODE}" \
  +reward.custom_reward_function.reward_kwargs.device="${REWARD_DEVICE}" \
  +reward.custom_reward_function.reward_kwargs.biovil_t_path="${BIOVIL_T_PATH}" \
  +reward.custom_reward_function.reward_kwargs.classifier_checkpoint="${CXR_CLASSIFIER_CHECKPOINT}" \
  +reward.custom_reward_function.reward_kwargs.raddino_path="${RAD_DINO_PATH}" \
  +reward.custom_reward_function.reward_kwargs.raddino_reference_cache="${RADDINO_REFERENCE_CACHE}" \
  +reward.custom_reward_function.reward_kwargs.biovil_weight="${BIOVIL_WEIGHT}" \
  +reward.custom_reward_function.reward_kwargs.label_weight="${LABEL_WEIGHT}" \
  +reward.custom_reward_function.reward_kwargs.raddino_weight="${RADDINO_WEIGHT}" \
  +reward.custom_reward_function.reward_kwargs.height="${SANA_HEIGHT}" \
  +reward.custom_reward_function.reward_kwargs.width="${SANA_WIDTH}" \
  +reward.custom_reward_function.reward_kwargs.num_inference_steps="${SANA_STEPS}" \
  +reward.custom_reward_function.reward_kwargs.guidance_scale="${SANA_GUIDANCE_SCALE}" \
  +reward.custom_reward_function.reward_kwargs.seed="${SEED}" \
  +reward.custom_reward_function.reward_kwargs.debug_image_dir="${DEBUG_IMAGE_DIR}" \
  +reward.custom_reward_function.reward_kwargs.print_denominator="${REWARD_PRINT_DENOMINATOR}" \
  +reward.custom_reward_function.reward_kwargs.max_print_chars="${REWARD_MAX_PRINT_CHARS}" \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="'${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_REWARD_CUDA_VISIBLE_DEVICES="'${REWARD_CUDA_VISIBLE_DEVICES}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_REWARD_USE_DISPATCHER="'${REWARD_USE_DISPATCHER}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_SANA_BATCH_SIZE="'${SANA_BATCH_SIZE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_SANA_BATCH_MAX_WAIT_MS="'${SANA_BATCH_MAX_WAIT_MS}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_REWARD_GPU_WORKER_MAX_CONCURRENCY="'${REWARD_GPU_WORKER_MAX_CONCURRENCY}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="'${HF_HUB_OFFLINE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="'${TRANSFORMERS_OFFLINE}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LLMSANA_LOCAL_FILES_ONLY="'${LLMSANA_LOCAL_FILES_ONLY}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.DISABLE_XFORMERS="'${DISABLE_XFORMERS}'" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "Done. Logs: ${RUN_DIR}"
