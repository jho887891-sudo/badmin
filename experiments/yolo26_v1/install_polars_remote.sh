#!/usr/bin/env bash
set -u
REPO=/home/T7/ojh/robot_sim
DEPS=$REPO/experiments/yolo26_p2_ab/_deps
cd "$REPO" || exit 1
env_isaaclab/bin/python -m pip install --no-deps --quiet --target "$DEPS" polars-runtime-64 2>&1 | tail -3
PYTHONPATH="$DEPS" env_isaaclab/bin/python -c "import polars as pl; print('polars', pl.__version__); import polars._utils.polars_version as v; print('runtime', v.get_polars_version())" 2>&1 | tail -4