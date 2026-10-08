#!/bin/bash
# Wait for the 2-epoch smoke, evaluate the spec-12 pass conditions mechanically, and only then launch the full run.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=/home/dgut/runs/detect/runs_cont_v1/smoke_b16
for i in $(seq 1 24); do pgrep -f "train_eth_only_v1[.]py" > /dev/null || break; sleep 20; done
echo "=== $(date -Is) smoke alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- results.csv:"; cut -d, -f1-8 "$R/results.csv" 2>/dev/null
ROWS=$(wc -l < "$R/results.csv" 2>/dev/null || echo 0)
NAN=$(grep -ci nan "$R/results.csv" 2>/dev/null || echo 1)
PT=$(ls "$R/weights/"*.pt 2>/dev/null | wc -l)
MAN=$L/smoke_b16_manifest.json
echo "rows=$ROWS nan=$NAN pt_files=$PT manifest=$([ -f "$MAN" ] && echo yes || echo no)"
echo "--- manifest summary:"
[ -f "$MAN" ] && /usr/bin/python3 -c "
import json; m=json.load(open('$MAN'))
print(json.dumps({k:m.get(k) for k in ('smoke','diagnostic_only','eligible_for_final_report','diagnostic_reasons','batch','batch_attempts_tried','runtime','endpoint','steps','save_dir')}, indent=1))"
PASS=no
if [ "$ROWS" -ge 3 ] && [ "$NAN" -eq 0 ] && [ "$PT" -ge 2 ] && [ -f "$MAN" ]; then PASS=yes; fi
echo "SMOKE_PASS=$PASS"
if [ "$PASS" = "yes" ]; then
  cd "$B" || exit 3
  echo "=== launching the full 20-epoch run $(date -Is)"
  setsid nohup /usr/bin/python3 "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
    --recipe-json "$B/eth_only_v1_official_recipe.json" \
    --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
    --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
    --project runs_cont_v1 --name full_cont_e20 --out-manifest "$L/full_cont_e20_manifest.json" \
    --batch 16 --mem-cap-gib 24 --workers 8 --device 0 \
    > "$L/full_cont_e20.log" 2>&1 < /dev/null &
  sleep 30
  echo "full run started: alive=$(pgrep -fc 'train_eth_only_v1[.]py') log=$(stat -c%s $L/full_cont_e20.log) B"
fi
