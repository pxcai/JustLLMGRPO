#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
VERL_REVISION="${VERL_REVISION:-4cd50e69b73b4ff0df750264f89e49c94c112c15}"
PY="${PY:-python}"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${THIRD_PARTY}/verl/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "${THIRD_PARTY}/verl"
fi
git -C "${THIRD_PARTY}/verl" fetch --all --tags
git -C "${THIRD_PARTY}/verl" checkout "${VERL_REVISION}"
cp -R "${THIRD_PARTY}/verl_overrides/verl/." "${THIRD_PARTY}/verl/verl/"

"${PY}" -m pip install -e "${THIRD_PARTY}/verl"

echo "Installed pinned VERL with the JustLLMGRPO overrides."
