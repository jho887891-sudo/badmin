#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_smoke_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== proc ==="
pgrep -fa "[t]rain_yolo26_v1" | head -1 || echo NONE
echo "=== lr_config.json ==="
cat "$RUN/lr_config.json" 2>/dev/null || echo "(not written yet)"
echo "=== first batch probe ==="
head -1 "$RUN/lr_probe.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(json.dumps({'epoch':d['epoch'],'step':d['step'],'lrs':[g['lr'] for g in d['groups']]}, indent=1))" 2>/dev/null || head -c 300 "$RUN/lr_probe.jsonl" 2>/dev/null
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-150