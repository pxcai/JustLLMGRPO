#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${ROOT}/configs/paper_experiment.env"
if [[ -f "${ROOT}/configs/paths.env" ]]; then
  source "${ROOT}/configs/paths.env"
fi
set +a

export ROOT
export VERL_DIR="${VERL_DIR:-${ROOT}/third_party/verl}"
export TRAIN_CSV="${TRAIN_CSV:-${ROOT}/data/LLAVARAD_ANNOTATIONS_TRAIN.csv}"
export VAL_CSV="${VAL_CSV:-${ROOT}/data/LLAVARAD_ANNOTATIONS_TEST.csv}"
export CXR_CLASSIFIER_CHECKPOINT="${CXR_CLASSIFIER_CHECKPOINT:-${ROOT}/artifacts/best_classifier.pt}"
export RADDINO_REFERENCE_CACHE="${RADDINO_REFERENCE_CACHE:-${ROOT}/artifacts/raddino_train20k_ref.npz}"
export TRAINER_LOGGER='["console","file"]'
export SAVE_FREQ=400
export TEST_FREQ=10
export VAL_BEFORE_TRAIN=True

exec bash "${ROOT}/llm_sana/run_verl_qwen3_llm_sana_grpo.sh" "$@"
