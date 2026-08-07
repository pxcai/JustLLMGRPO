#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-python}"

if [[ -f "${ROOT}/configs/paths.env" ]]; then
  set -a
  source "${ROOT}/configs/paths.env"
  set +a
fi

: "${INPUT_CSV:?Set INPUT_CSV to a CSV containing source prompts.}"
: "${PLANNER_MODEL:?Set PLANNER_MODEL to the merged JustLLMGRPO checkpoint.}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/inference}"
SANA_MODEL_PATH="${SANA_MODEL_PATH:-raman07/CheXGenBench-Models-Sana-e20}"
PROMPT_COL="${PROMPT_COL:-annotated_prompt}"
ID_COL="${ID_COL:-id}"
PLANNER_BACKEND="${PLANNER_BACKEND:-auto}"
PLANNER_TP_SIZE="${PLANNER_TP_SIZE:-1}"

exec "${PY}" -m inference.run_inference \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --planner-model "${PLANNER_MODEL}" \
  --sana-model "${SANA_MODEL_PATH}" \
  --prompt-column "${PROMPT_COL}" \
  --id-column "${ID_COL}" \
  --planner-backend "${PLANNER_BACKEND}" \
  --tensor-parallel-size "${PLANNER_TP_SIZE}" \
  "$@"
