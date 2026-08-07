#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PY=${PY:-python}
if [[ -f "${ROOT}/configs/paths.env" ]]; then
  set -a
  source "${ROOT}/configs/paths.env"
  set +a
fi
REAL_CSV=${REAL_CSV:-${REAL_CXR_CSV:-${ROOT}/data/training_data_20K.csv}}
REAL_IMAGE_DIR=${REAL_IMAGE_DIR:-${REAL_CXR_ROOT:-${ROOT}/data/mimic-cxr-jpg}}
RAD_DINO_PATH=${RAD_DINO_PATH:-microsoft/rad-dino}
OUTPUT=${OUTPUT:-${ROOT}/artifacts/raddino_train20k_ref.npz}
MAX_IMAGES=${MAX_IMAGES:-20000}
BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-cuda}
CAPTION_COL=${CAPTION_COL:-__no_dedup__}

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$(dirname "${OUTPUT}")"

cd "${ROOT}"
"${PY}" -m llm_sana.rewards.chexgenbench_reward build-raddino-cache \
  --real_csv "${REAL_CSV}" \
  --real_image_dir "${REAL_IMAGE_DIR}" \
  --raddino_path "${RAD_DINO_PATH}" \
  --output "${OUTPUT}" \
  --caption_col "${CAPTION_COL}" \
  --max_images "${MAX_IMAGES}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}"

echo "Saved RadDINO cache: ${OUTPUT}"
