#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PY:-python}"
VERL_DIR="${VERL_DIR:-${ROOT}/third_party/verl}"

RUN_DIR="${RUN_DIR:-${LLM_SANA_RUN_DIR:-${ROOT}/outputs/justllmgrpo_qwen3_4b}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
TARGET_ROOT="${TARGET_ROOT:-${RUN_DIR}/merged_hf}"
BACKEND="${BACKEND:-fsdp}"

# Merge one step by default. Set STEPS=300,350,400 to merge several.
if [[ -n "${STEPS:-}" ]]; then
  IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
elif [[ -n "${STEP:-}" ]]; then
  STEP_LIST=("${STEP}")
else
  if [[ -f "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
    STEP_LIST=("$(<"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")")
  else
    echo "No STEP/STEPS provided and ${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt is missing." >&2
    exit 1
  fi
fi

mkdir -p "${TARGET_ROOT}"

echo "==> RUN_DIR: ${RUN_DIR}"
echo "==> CHECKPOINT_ROOT: ${CHECKPOINT_ROOT}"
echo "==> TARGET_ROOT: ${TARGET_ROOT}"
echo "==> BACKEND: ${BACKEND}"
echo "==> STEPS: ${STEP_LIST[*]}"

merge_one() {
  local step="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/global_step_${step}/actor"
  local out_dir="${TARGET_ROOT}/global_step_${step}"

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "Missing actor checkpoint directory: ${ckpt_dir}" >&2
    return 1
  fi

  if ! find "${ckpt_dir}" -maxdepth 1 -type f \( -name 'model_world_size_*.pt' -o -name 'model.pt' \) | grep -q .; then
    echo "No model weights found under ${ckpt_dir}." >&2
    echo "This step cannot be merged because only metadata/data.pt was kept." >&2
    return 1
  fi

  if [[ -e "${out_dir}" && -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" && "${FORCE_MERGE:-0}" != "1" ]]; then
    echo "Target directory already exists and is not empty: ${out_dir}" >&2
    echo "Set FORCE_MERGE=1 to overwrite it." >&2
    return 1
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  echo "==> Merging global_step_${step}"
  "${PY}" "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend "${BACKEND}" \
    --local_dir "${ckpt_dir}" \
    --target_dir "${out_dir}"

  echo "==> Done: ${out_dir}"
}

for step in "${STEP_LIST[@]}"; do
  merge_one "${step}"
done
