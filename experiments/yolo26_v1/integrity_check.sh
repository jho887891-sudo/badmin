#!/usr/bin/env bash
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
echo "=== run files (integrity) ==="
ls -l "$RUN/results.csv" "$RUN/lr_probe.jsonl" "$RUN/mem_probe.jsonl" "$RUN/lr_config.json" "$RUN/weights/best.pt" "$RUN/weights/last.pt" 2>/dev/null | while read -r line; do echo "$line"; done
echo "=== line counts ==="
wc -l "$RUN/results.csv" "$RUN/lr_probe.jsonl" "$RUN/mem_probe.jsonl" 2>/dev/null
echo "=== latest epochs ==="
tail -2 "$RUN/results.csv"
echo "=== current progress ==="
grep -a "1024:" "$RUN.log" | tail -1 | tr -d "\r" | cut -c1-130
echo "=== process ==="
pgrep -fa "[t]rain_yolo26_v1" | head -1 | cut -c1-120