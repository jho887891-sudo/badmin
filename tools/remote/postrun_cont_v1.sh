#!/bin/bash
# Post-run chain for the continuation experiment: wait for the 20-epoch run to finish, then strip the epoch-20
# checkpoint to inference-only weights and record its hashes. Transfer + local evaluation happen afterwards.
# Gate logic dry-run before arming (lesson from the 2026-10-08 launcher incident).
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=$B/runs/detect/runs_cont_v1/full_cont_e20
LOG=$L/postrun.log
echo "=== postrun armed $(date -Is)" >> "$LOG"
for i in $(seq 1 240); do
  pgrep -f "train_eth_only_v1[.]py" > /dev/null || break
  sleep 30
done
ROWS=$(wc -l < "$R/results.csv" 2>/dev/null); ROWS=${ROWS:-0}
echo "$(date -Is) trainer exited; results rows=$ROWS (header + epochs)" >> "$LOG"
if [ "$ROWS" -lt 21 ]; then echo "$(date -Is) GATE FAIL: only $ROWS rows, expected 21" >> "$LOG"; exit 4; fi
if [ ! -s "$R/weights/last.pt" ]; then echo "$(date -Is) GATE FAIL: last.pt missing or empty" >> "$LOG"; exit 4; fi
if [ ! -s "$L/full_cont_e20_manifest.json" ]; then echo "$(date -Is) GATE FAIL: run manifest missing" >> "$LOG"; exit 4; fi
echo "$(date -Is) GATES PASS -> stripping" >> "$LOG"
/usr/bin/python3 "$L/strip_cont_v1_ckpt.py" \
  --src "$R/weights/last.pt" --dst "$B/cont_v1/e20_infer.pt" \
  --provenance "$B/cont_v1/e20_infer_provenance.json" --label "epoch20_primary_endpoint" >> "$LOG" 2>&1
echo "$(date -Is) strip rc=$?" >> "$LOG"
ls -l "$R/weights/" >> "$LOG" 2>&1
echo "$(date -Is) postrun done" >> "$LOG"
