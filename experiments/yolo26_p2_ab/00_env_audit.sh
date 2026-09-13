#!/usr/bin/env bash
# 00_env_audit.sh - read-only environment audit for the YOLO26s vs P2 A/B experiment.
# Usage: bash 00_env_audit.sh [OUTPUT_DIR]   (no env modification; nothing is installed)
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:-$REPO/outputs/yolo26_p2_ab_test}
mkdir -p "$OUT"
ENVF=$OUT/environment.txt
PY=$REPO/env_isaaclab/bin/python
{
  echo "# YOLO26s vs YOLO26s-P2 A/B - environment audit"
  echo "# generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')  (UTC)"
  echo "# host: $(hostname)"
  echo
  echo '## OS'
  ( . /etc/os-release 2>/dev/null; echo "PRETTY_NAME=$PRETTY_NAME"; echo "VERSION=$VERSION" )
  echo "KERNEL=$(uname -r)"
  echo "ARCH=$(uname -m)"
  echo
  echo '## Python / PyTorch / CUDA (from project venv, read-only)'
  "$PY" - <<'PY'
import platform, sys
print('python_version =', sys.version.split()[0], '| executable =', sys.executable)
try:
    import torch
    print('torch_version =', torch.__version__)
    print('torch_cuda_build =', torch.version.cuda)
    print('cuda_available =', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('torch_cuda_device_count =', torch.cuda.device_count())
        print('torch_cuda_device_0 =', torch.cuda.get_device_name(0))
except Exception as exc:
    print('torch_import_error =', repr(exc))
for mod in ('numpy','cv2','PIL','yaml','scipy','matplotlib','onnxruntime','tensorrt','ultralytics'):
    try:
        m = __import__(mod)
        print(f'{mod}_version = ' + str(getattr(m, '__version__', 'present')))
    except Exception as exc:
        print(f'{mod}_version = NOT_INSTALLED ({type(exc).__name__})')
PY
  echo
  echo '## NVIDIA driver / GPU (nvidia-smi, read-only)'
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,compute_mode --format=csv 2>/dev/null || echo 'nvidia-smi unavailable'
  echo
  echo '## CUDA runtime reported by driver'
  nvidia-smi 2>/dev/null | grep -i 'CUDA Version' || true
  echo
  echo '## Current GPU processes (READ-ONLY snapshot; nothing is terminated)'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo 'no compute apps query'
  echo
  echo '## Disk'
  df -h "$REPO" | tail -1
} > "$ENVF" 2>&1
echo "wrote $ENVF"
echo '--- environment.txt (head) ---'
sed -n '1,60p' "$ENVF"