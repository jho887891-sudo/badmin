#!/bin/bash
# wait_for_clean_window.sh - read-only gate for YOLO26S_INFERENCE_SPEED_OPT_V1 latency runs.
#
# Why this exists: the A6000 is shared. Track A measured at GPU 0% gave 15.73/16.07 ms;
# Tracks B/C measured at 33-99% background utilisation gave 18.1-20.9 ms and are UNUSABLE.
# The spec section 17 record schema has no field for background load, so this gate is what
# decides whether a run is admissible at all.
#
# Usage:  bash wait_for_clean_window.sh --check        # evaluate once, print verdict
#         bash wait_for_clean_window.sh                # wait for 3 consecutive clean polls
# Env:    MAX_WAIT_S (default 7200)  POLL_S (default 60)  NEEDED (default 3)
set -u
GPU_UTIL_MAX=${GPU_UTIL_MAX:-10}
GPU_MEM_MAX=${GPU_MEM_MAX:-1200}
LOAD1_MAX=${LOAD1_MAX:-4.0}
# spec requires compute_apps == 0; this script used to omit the check, which let a foreign job through.
GPU_APPS_MAX=${GPU_APPS_MAX:-0}
POLL_S=${POLL_S:-60}
NEEDED=${NEEDED:-3}
MAX_WAIT_S=${MAX_WAIT_S:-7200}
MODE=${1:-wait}

evaluate() {
  local util mem load1 apps ok
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
  load1=$(awk "{print \$1}" /proc/loadavg)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  ok=YES
  awk -v a="$util" -v b="$GPU_UTIL_MAX" "BEGIN{exit !(a>b)}" && ok=NO
  awk -v a="$mem"  -v b="$GPU_MEM_MAX"  "BEGIN{exit !(a>b)}" && ok=NO
  awk -v a="$load1" -v b="$LOAD1_MAX"   "BEGIN{exit !(a>b)}" && ok=NO
  [ "$apps" -le "$GPU_APPS_MAX" ] || ok=NO
  printf "%-20s gpu_util=%-4s mem=%-6s load1=%-6s compute_apps=%s  -> %s\n" "$(date -Is)" "$util" "$mem" "$load1" "$apps" "$ok"
  [ "$ok" = "YES" ]
}

if [ "$MODE" = "--check" ]; then
  evaluate && { echo "VERDICT: CLEAN"; exit 0; } || { echo "VERDICT: BUSY"; exit 1; }
fi

echo "waiting for $NEEDED consecutive clean polls (poll ${POLL_S}s, max ${MAX_WAIT_S}s)"
echo "thresholds: gpu_util<=$GPU_UTIL_MAX  gpu_mem<=${GPU_MEM_MAX}MiB  load1<=$LOAD1_MAX"
waited=0; streak=0
while [ "$waited" -lt "$MAX_WAIT_S" ]; do
  if evaluate; then streak=$((streak+1)); else streak=0; fi
  if [ "$streak" -ge "$NEEDED" ]; then echo "CLEAN WINDOW CONFIRMED after ${waited}s"; exit 0; fi
  sleep "$POLL_S"; waited=$((waited+POLL_S))
done
echo "TIMEOUT after ${MAX_WAIT_S}s without $NEEDED consecutive clean polls"; exit 1