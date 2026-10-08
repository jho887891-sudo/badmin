#!/bin/bash
# Keep waiting for the 2-epoch smoke to finish, check the spec-12 pass conditions, and then launch the 20-epoch
# full run automatically. Runs detached so a dropped SSH session cannot stop the experiment.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=/home/dgut/runs/detect/runs_cont_v1/smoke_b16
LOG=$L/auto_launcher.log
echo "=== auto launcher started $(date -Is)" >> "$LOG"
for i in $(seq 1 120); do
  pgrep -f "train_eth_only_v1[.]py" > /dev/null || break
  sleep 30
done
ROWS=$(wc -l < "$R/results.csv" 2>/dev/null || echo 0)
NAN=$(grep -ci nan "$R/results.csv" 2>/dev/null || echo 1)
PT=$(ls "$R/weights/"*.pt 2>/dev/null | wc -l)
MAN=$L/smoke_b16_manifest.json
echo "$(date -Is) smoke done: rows=$ROWS nan=$NAN pt=$PT manifest=$([ -f "$MAN" ] && echo yes || echo no)" >> "$LOG"
if [ "$ROWS" -ge 3 ] && [ "$NAN" -eq 0 ] && [ "$PT" -ge 2 ] && [ -f "$MAN" ] && [ ! -f "$L/full_cont_e20_manifest.json" ]; then
  cd "$B" || exit 3
  echo "$(date -Is) SMOKE_PASS -> launching full 20-epoch run" >> "$LOG"
  setsid nohup /usr/bin/python3 "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
    --recipe-json "$B/eth_only_v1_official_recipe.json" \
    --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
    --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
    --project runs_cont_v1 --name full_cont_e20 --out-manifest "$L/full_cont_e20_manifest.json" \
    --batch 16 --mem-cap-gib 24 --workers 8 --device 0 \
    > "$L/full_cont_e20.log" 2>&1 < /dev/null &
  sleep 20
  echo "$(date -Is) full run pid=$(pgrep -f 'train_eth_only_v1[.]py' | head -1)" >> "$LOG"
else
  echo "$(date -Is) SMOKE_PASS=no (or full run already exists) - NOT launching" >> "$LOG"
fi
