#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
GENESIS_DIR="${GENESIS_REPO_DIR:-${THIRD_PARTY_DIR}/Genesis}"
GENESIS_ASSETS_ROOT_DEFAULT="${GENESIS_DIR}/genesis/assets"
ROBOT_REL_PATH="xml/franka_emika_panda/panda.xml"

mkdir -p "${THIRD_PARTY_DIR}"

echo "[sim-bootstrap] repo root: ${REPO_ROOT}"
echo "[sim-bootstrap] target Genesis dir: ${GENESIS_DIR}"

if [[ -d "${GENESIS_DIR}" ]]; then
  echo "[sim-bootstrap] Genesis directory already exists, skip clone."
else
  if command -v git >/dev/null 2>&1; then
    echo "[sim-bootstrap] Genesis directory not found."
    echo "[sim-bootstrap] Trying to clone public Genesis repository..."
    if git clone https://github.com/Genesis-Embodied-AI/Genesis.git "${GENESIS_DIR}"; then
      echo "[sim-bootstrap] Clone done."
    else
      echo "[sim-bootstrap] Clone failed (network/auth)."
      echo "[sim-bootstrap] Please clone manually to: ${GENESIS_DIR}"
    fi
  else
    echo "[sim-bootstrap] git is not installed. Please clone Genesis manually to: ${GENESIS_DIR}"
  fi
fi

cat <<EOF
[sim-bootstrap] Next steps:
1) Install Genesis python package (editable mode example):
   pip install -e "${GENESIS_DIR}"

2) Export environment variables (if custom paths are used):
   export GENESIS_REPO_DIR="${GENESIS_DIR}"
   export GENESIS_ASSETS_ROOT="${GENESIS_ASSETS_ROOT_DEFAULT}"

3) Verify robot asset exists:
   test -f "${GENESIS_ASSETS_ROOT_DEFAULT}/${ROBOT_REL_PATH}" && echo "ok" || echo "missing"

4) Run minimal simulation command:
   python experiments/04_sim_exp/run_e2e_sim.py --instruction "移动到方块上方并张开夹爪"
EOF
