#!/usr/bin/env bash
# run_window_queue.sh - use ONE clean window for both remaining workstreams, in order:
#   1) finish the YOLO26 chain: gate E -> E -> gate A4 -> A4 -> analyze
#   2) fair comparison:        gate -> export ETH YOLOv8s TRT FP16 static -> gate -> bench (same harness)
# Nothing runs while the gate is BUSY. wait_gate NEVER gives up: a 2 h cap made the queue waste 4 h on
# 2026-10-08 without producing anything, so a timeout now re-enters the wait instead of exiting.
set -u
RUN_DIR=${RUN_DIR:-/home/T7/dgut/trt_env}
PY=${PY:-/home/T7/ojh/robot_sim/env_isaaclab/bin/python}
WHEEL=${WHEEL:-/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract}
ETH_CKPT=${ETH_CKPT:-/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/runs/final-model/best.pt}
IMGSZ=1024; WARMUP=50; ITERS=500; GATE_NEED=3; GATE_POLL=60
GATE_MAX_WAIT=${GATE_MAX_WAIT:-86400}
export PYTHONPATH="$RUN_DIR/pkgs:$WHEEL:/home/T7/ojh/robot_sim/tools"
export LD_LIBRARY_PATH="$RUN_DIR/pkgs/tensorrt_libs:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled YOLO_AUTOINSTALL=false CUDA_VISIBLE_DEVICES=0
export TMPDIR="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" PIP_CACHE_DIR="$RUN_DIR/pipcache"
H="$RUN_DIR/speed_chain_helpers.py"; X="$RUN_DIR/speed_harness.py"
LOG="$RUN_DIR/queue.log"; STATE="$RUN_DIR/chain_state"; mkdir -p "$STATE"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
wait_gate() {
  while :; do
    log "gate before $1 (max_wait=${GATE_MAX_WAIT}s, need $GATE_NEED consecutive clean polls)"
    "$PY" "$H" gate --need "$GATE_NEED" --poll "$GATE_POLL" --max-wait "$GATE_MAX_WAIT" \
        --out "$STATE/gate_$1.json" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then log "gate $1 PASSED"; return 0; fi
    log "gate $1 timed out after ${GATE_MAX_WAIT}s - re-entering the wait, NOT giving up"
    sleep 60
  done
}
stage2() {
  log "stage 2: ETH YOLOv8s TensorRT FP16 static"
  wait_gate ETH_EXPORT || return 20
  "$PY" "$RUN_DIR/speed_eth_export.py" --ckpt "$ETH_CKPT" --imgsz "$IMGSZ" --batch 1 \
    --engine-out "$RUN_DIR/eth_yolov8s_static.engine" --out "$STATE/eth_export.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  log "eth export rc=$rc"
  [ "$rc" -eq 0 ] || return 41
  [ -f "$RUN_DIR/eth_yolov8s_static.engine" ] || return 40
  wait_gate ETH_BENCH || return 20
  "$PY" "$X" --track ETH_v8_TRT --backend tensorrt --engine "$RUN_DIR/eth_yolov8s_static.engine" \
    --imgsz "$IMGSZ" --batch 1 --precision fp16 --warmup "$WARMUP" --iters "$ITERS" \
    --monitor-csv "$RUN_DIR/speed_ETH_v8_TRT.monitor.csv" --gate-passed yes --gate-streak "$GATE_NEED" \
    --out "$RUN_DIR/eth_yolov8_trt_fp16_static.json" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  log "eth bench rc=$rc"
  return "$rc"
}
cd "$RUN_DIR" || exit 1
log "QUEUE START (patient mode, max_wait=${GATE_MAX_WAIT}s per gate)"
log "=== stage 1: finish YOLO26 chain (E, A4, analyze) ==="
bash "$RUN_DIR/run_speed_opt_chain_part3.sh" 2>&1 | tee -a "$LOG"
log "stage 1 rc=${PIPESTATUS[0]}"
stage2
log "stage 2 rc=$?"
log "QUEUE DONE"