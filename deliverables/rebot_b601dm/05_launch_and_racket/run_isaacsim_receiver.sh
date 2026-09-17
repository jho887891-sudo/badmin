#!/bin/bash
# Launch the reBot B601-DM Isaac Sim receiver on THIS host.
#
# Why a wrapper at all: upstream (README_EN.md:144-147) expects the official Isaac Sim python.sh from a
# standalone install. This host has Isaac Sim as a pip package inside the project environment instead, so the
# documented command has to be translated rather than followed.
#
# What this script does NOT do, deliberately:
#   - it installs nothing and upgrades nothing (no pip/apt/conda); the Isaac Sim, PyTorch, CUDA and driver
#     stack is left exactly as found
#   - it kills nothing; other users' processes (pt_main_thread jobs were running when this was written) are
#     not touched, and neither is any vLLM
#   - it writes nothing into the upstream repositories
#
# Usage:
#   scripts/simulation/run_isaacsim_receiver.sh [--dry-run] [--port 5005]
#
# Environment:
#   REBOT_ASSET_DIR   patched asset tree (default: outputs/simulation/rebot_b601dm/real_limits)
#   REBOT_UPSTREAM    upstream repository root (default: <repo>/reBot-Isaacsim)
#   ISAACLAB_ROOT     Isaac Lab checkout (default: /home/T7/ojh/robot_sim/IsaacLab)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/T7/ojh/robot_sim/IsaacLab}"
ENV_PY="${ISAACLAB_ROOT%/IsaacLab}/env_isaaclab/bin/python"

if [ ! -x "$ENV_PY" ]; then
  echo "[error] project python not found at $ENV_PY" >&2
  echo "        set ISAACLAB_ROOT to the Isaac Lab checkout" >&2
  exit 2
fi

# Isaac Lab is importable only with its source subdirectories on PYTHONPATH.
ISAACLAB_PATH=""
for d in "$ISAACLAB_ROOT"/source/*; do
  [ -d "$d" ] && ISAACLAB_PATH="${ISAACLAB_PATH:+${ISAACLAB_PATH}:}$d"
done
export PYTHONPATH="${ISAACLAB_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Isaac Sim writes caches; keep them inside the project rather than filling the root partition.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export REBOT_ASSET_DIR="${REBOT_ASSET_DIR:-$REPO_ROOT/outputs/simulation/rebot_b601dm/real_limits}"
export REBOT_UPSTREAM="${REBOT_UPSTREAM:-$REPO_ROOT/reBot-Isaacsim}"

echo "[launcher] python      : $ENV_PY"
echo "[launcher] IsaacLab    : $ISAACLAB_ROOT"
echo "[launcher] asset dir   : $REBOT_ASSET_DIR"
echo "[launcher] upstream    : $REBOT_UPSTREAM"

# -u matters: redirected to a file, Python block-buffers stdout, so a driver that prints a few hundred bytes
# and then spends ten minutes starting Isaac Sim looks IDENTICAL to one that hung immediately. That is exactly
# how the first attempt at this read.
exec "$ENV_PY" -u "$REPO_ROOT/scripts/simulation/isaacsim_receiver_driver.py" "$@"
