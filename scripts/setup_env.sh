#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-python}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0+cu118}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.21.0+cu118}"

"${PY}" -m pip install --upgrade pip setuptools wheel packaging ninja
"${PY}" -m pip install \
  "${TORCH_SPEC}" "${TORCHVISION_SPEC}" \
  --index-url "${TORCH_INDEX_URL}"
"${PY}" -m pip install -r "${ROOT}/requirements.txt"
"${PY}" -m pip install -e "${ROOT}"

PY="${PY}" bash "${ROOT}/scripts/install_third_party.sh"

"${PY}" - <<'PY'
import diffusers
import ray
import torch
import transformers
import verl
import vllm

print("Environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("ray", ray.__version__)
print("vllm", vllm.__version__)
print("verl", getattr(verl, "__version__", "editable"))
PY
