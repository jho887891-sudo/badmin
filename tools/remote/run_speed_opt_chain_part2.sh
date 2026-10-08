#!/usr/bin/env bash
# run_speed_opt_chain_part2.sh - resume after the D failure: gate D -> D -> gate E -> E -> gate A4 -> A4 -> analyze.
# A3/B/C already have VALID records, so they are deliberately NOT re-run (no need to burn the window).
set -u
RUN_DIR=${RUN_DIR:-/home/T7/dgut/trt_env}
PY=${PY:-/home/T7/ojh/robot_sim/env_isaaclab/bin/python}
WHEEL=${WHEEL:-/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract}
CKPT=${CKPT:-$HOME/.dsh-bench/eth_only_v1_best.pt}
IMGSZ=1024; WARMUP=50; ITERS=500; GATE_NEED=3; GATE_POLL=60
export PYTHONPATH="$RUN_DIR/pkgs:$WHEEL"
export LD_LIBRARY_PATH="$RUN_DIR/pkgs/tensorrt_libs:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled YOLO_AUTOINSTALL=false CUDA_VISIBLE_DEVICES=0
export TMPDIR="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" PIP_CACHE_DIR="$RUN_DIR/pipcache"
H="$RUN_DIR/speed_chain_helpers.py"; X="$RUN_DIR/speed_harness.py"
LOG="$RUN_DIR/chain_part2.log"; STATE="$RUN_DIR/chain_state"
mkdir -p "$STATE"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
gate() {
  log "gate before $1"
  "$PY" "$H" gate --need "$GATE_NEED" --poll "$GATE_POLL" --out "$STATE/gate_$1.json" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { log "GATE FAILED before $1"; exit 20; }
}
cd "$RUN_DIR" || exit 1
[ -f "$CKPT" ] || { log "FATAL checkpoint not found: $CKPT"; exit 2; }
log "part2 start ckpt=$CKPT"

ENG_D="$HOME/.dsh-bench/eth_only_v1_best.engine"
if [ ! -f "$ENG_D" ]; then
  log "D engine absent -> exporting static"
  "$PY" "$H" export --ckpt "$CKPT" --imgsz "$IMGSZ" --batch 1 --precision fp16 --dynamic no --out "$STATE/export_D.json" 2>&1 | tee -a "$LOG" || exit 31
fi
log "reusing D engine $(stat -c%s "$ENG_D") bytes"
gate D
"$PY" "$X" --track D --backend tensorrt --engine "$ENG_D" --imgsz "$IMGSZ" --batch 1 --precision fp16 \
  --warmup "$WARMUP" --iters "$ITERS" --monitor-csv "$RUN_DIR/speed_D.monitor.csv" \
  --gate-passed yes --gate-streak "$GATE_NEED" --out "$RUN_DIR/speed_tensorrt_fp16_static.json" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "track D FAILED"; exit 32; }
log "track D done"

gate E
"$PY" "$H" export --ckpt "$CKPT" --imgsz "$IMGSZ" --batch 1 --precision fp16 --dynamic yes --out "$STATE/export_E.json" 2>&1 | tee -a "$LOG" || exit 33
ENG_E=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(chr(101)+chr(110)+chr(103)+chr(105)+chr(110)+chr(101)))" "$STATE/export_E.json")
log "E engine=$ENG_E"
"$PY" "$X" --track E --backend tensorrt --engine "$ENG_E" --imgsz "$IMGSZ" --batch 1 --precision fp16 \
  --warmup "$WARMUP" --iters "$ITERS" --monitor-csv "$RUN_DIR/speed_E.monitor.csv" \
  --gate-passed yes --gate-streak "$GATE_NEED" --out "$RUN_DIR/speed_tensorrt_fp16_dynamic.json" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "track E FAILED"; exit 34; }
log "track E done"

gate A4
"$PY" "$X" --track A4 --backend pytorch --ckpt "$CKPT" --imgsz "$IMGSZ" --batch 1 --precision fp32 --compile no \
  --warmup "$WARMUP" --iters "$ITERS" --monitor-csv "$RUN_DIR/speed_A4.monitor.csv" \
  --gate-passed yes --gate-streak "$GATE_NEED" --out "$RUN_DIR/speed_pytorch_fp32_A4.json" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "track A4 FAILED"; exit 35; }
log "track A4 done"

log "analyze"
"$PY" "$RUN_DIR/speed_chain_analyze.py" --records-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
log "part2 done"