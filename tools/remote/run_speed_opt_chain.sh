#!/usr/bin/env bash
# run_speed_opt_chain.sh - one-click chain for YOLO26S_INFERENCE_SPEED_OPT_V1.
#
# Order (each Track is preceded by its OWN clean-window gate, not one gate for the whole chain):
#   gate -> A3 FP32 -> gate -> B FP16 -> gate -> C FP16+compile -> gate -> D TRT FP16 static
#   -> gate -> E TRT FP16 dynamic -> gate -> A4 FP32 -> analyze -> summary
#
# Never kills other processes, never changes GPU clocks, checkpoint, imgsz, conf/NMS/max_det.
# Any failure keeps its log and stops the chain; nothing is silently retried under another config.
# Track F (E2E / NMS-free) is deliberately NOT run - see speed_trt_capability.json.
#
# Usage:  RUN_DIR=/home/T7/dgut/trt_env bash run_speed_opt_chain.sh
set -u

RUN_DIR=${RUN_DIR:-/home/T7/dgut/trt_env}
PY=${PY:-/home/T7/ojh/robot_sim/env_isaaclab/bin/python}
WHEEL=${WHEEL:-/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract}
# The chain cd-s into RUN_DIR, so a bare filename does not resolve: the checkpoint lives in
# $HOME/.dsh-bench.  This was the first chain failure (FileNotFoundError at A3, 2026-10-06 15:54Z).
CKPT=${CKPT:-$HOME/.dsh-bench/eth_only_v1_best.pt}
IMGSZ=${IMGSZ:-1024}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-500}
GATE_NEED=${GATE_NEED:-3}
GATE_POLL=${GATE_POLL:-60}

export PYTHONPATH="$RUN_DIR/pkgs:$WHEEL"
export LD_LIBRARY_PATH="$RUN_DIR/pkgs/tensorrt_libs:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled YOLO_AUTOINSTALL=false CUDA_VISIBLE_DEVICES=0
export TMPDIR="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" PIP_CACHE_DIR="$RUN_DIR/pipcache"

H="$RUN_DIR/speed_chain_helpers.py"
X="$RUN_DIR/speed_harness.py"
LOG="$RUN_DIR/chain.log"
STATE="$RUN_DIR/chain_state"
mkdir -p "$STATE"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

gate() {
  local tag="$1"
  log "gate before $tag: need $GATE_NEED consecutive clean polls"
  "$PY" "$H" gate --need "$GATE_NEED" --poll "$GATE_POLL" \
      --out "$STATE/gate_${tag}.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then log "GATE FAILED before $tag (rc=$rc) - chain stops"; exit 20; fi
  grep -q "\"passed\": true" "$STATE/gate_${tag}.json" || { log "gate json missing pass flag"; exit 21; }
}

run_pytorch() {
  local tag="$1" prec="$2" comp="$3"
  log "track $tag start (pytorch $prec compile=$comp)"
  "$PY" "$X" --track "$tag" --backend pytorch --ckpt "$CKPT" --imgsz "$IMGSZ" --batch 1 \
      --precision "$prec" --compile "$comp" --warmup "$WARMUP" --iters "$ITERS" \
      --monitor-csv "$RUN_DIR/speed_${tag}.monitor.csv" --gate-passed yes --gate-streak "$GATE_NEED" \
      --out "$RUN_DIR/$(out_name "$tag")" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { log "track $tag FAILED rc=$rc - chain stops, artifacts kept"; exit 30; }
  log "track $tag done"
}

run_tensorrt() {
  local tag="$1" dyn="$2"
  log "export for $tag (dynamic=$dyn)"
  "$PY" "$H" export --ckpt "$CKPT" --imgsz "$IMGSZ" --batch 1 --precision fp16 --dynamic "$dyn" \
      --out "$STATE/export_${tag}.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { log "export for $tag FAILED rc=$rc"; exit 31; }
  local eng
  eng=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[\"engine\"])" "$STATE/export_${tag}.json")
  log "track $tag start (tensorrt engine=$eng)"
  "$PY" "$X" --track "$tag" --backend tensorrt --engine "$eng" --imgsz "$IMGSZ" --batch 1 \
      --precision fp16 --warmup "$WARMUP" --iters "$ITERS" \
      --monitor-csv "$RUN_DIR/speed_${tag}.monitor.csv" --gate-passed yes --gate-streak "$GATE_NEED" \
      --out "$RUN_DIR/$(out_name "$tag")" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { log "track $tag FAILED rc=$rc - chain stops, artifacts kept"; exit 32; }
  log "track $tag done"
}

out_name() {
  case "$1" in
    A3) echo speed_pytorch_fp32_A3.json ;;
    A4) echo speed_pytorch_fp32_A4.json ;;
    B)  echo speed_pytorch_fp16_clean.json ;;
    C)  echo speed_torch_compile_fp16_clean.json ;;
    D)  echo speed_tensorrt_fp16_static.json ;;
    E)  echo speed_tensorrt_fp16_dynamic.json ;;
  esac
}

cd "$RUN_DIR" || exit 1
log "chain start: run_dir=$RUN_DIR ckpt=$CKPT imgsz=$IMGSZ warmup=$WARMUP iters=$ITERS"
# Fail fast with a clear message instead of dying ~1 s into A3 with a torch stack trace.
if [ ! -f "$CKPT" ]; then log "FATAL checkpoint not found: $CKPT"; exit 2; fi
log "checkpoint ok: $(stat -c%s "$CKPT") bytes"

gate A3; run_pytorch A3 fp32 no
gate B;  run_pytorch B  fp16 no
gate C;  run_pytorch C  fp16 yes
gate D;  run_tensorrt D no
gate E;  run_tensorrt E yes
gate A4; run_pytorch A4 fp32 no

log "validity + summary"
"$PY" "$RUN_DIR/speed_chain_analyze.py" --records-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
log "chain done"