#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ISAACLAB_ROOT="$(printenv ISAACLAB_ROOT || echo /home/T7/ojh/robot_sim/IsaacLab)"
ENV_PY="$(dirname "$ISAACLAB_ROOT")/env_isaaclab/bin/python"

if [ ! -x "$ENV_PY" ]; then
  echo "[error] project Isaac Python not found at $ENV_PY" >&2
  echo "        set ISAACLAB_ROOT to the active Isaac Lab checkout" >&2
  exit 2
fi

ISAACLAB_PATH=""
for directory in "$ISAACLAB_ROOT"/source/*; do
  [ -d "$directory" ] || continue
  if [ -z "$ISAACLAB_PATH" ]; then
    ISAACLAB_PATH="$directory"
  else
    ISAACLAB_PATH="$ISAACLAB_PATH:$directory"
  fi
done
OLD_PYTHONPATH="$(printenv PYTHONPATH || true)"
if [ -n "$OLD_PYTHONPATH" ]; then
  export PYTHONPATH="$ISAACLAB_PATH:$OLD_PYTHONPATH"
else
  export PYTHONPATH="$ISAACLAB_PATH"
fi

export OMNI_KIT_ACCEPT_EULA="$(printenv OMNI_KIT_ACCEPT_EULA || echo YES)"
export BADMINTON_COURT_ASSET_USD="$(printenv BADMINTON_COURT_ASSET_USD || echo "$REPO_ROOT/assets/court/badminton_court.usd")"

echo "[launcher] Python: $ENV_PY"
echo "[launcher] Isaac Lab: $ISAACLAB_ROOT"
echo "[launcher] court asset: $BADMINTON_COURT_ASSET_USD"
exec "$ENV_PY" -u "$REPO_ROOT/scripts/simulation/run_isaac_rgbd_yolo_probe.py" "$@"
