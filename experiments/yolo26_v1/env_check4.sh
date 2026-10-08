#!/usr/bin/env bash
cd /home/T7/ojh/robot_sim || exit 1
echo "== manifest remote =="
ls -l outputs/shuttle_capability/v1_dataset/ 2>&1 | head -n 8
echo "== torch import timing =="
start=$(date +%s)
timeout 400 env_isaaclab/bin/python -c "import torch, ultralytics, PIL; print('torch', torch.__version__, 'ultra', ultralytics.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -n 5
rc=$?
echo "rc=$rc elapsed=$(( $(date +%s) - start ))s"