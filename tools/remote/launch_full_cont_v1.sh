#!/bin/bash
# Validate the smoke against the spec-12 conditions and launch the 20-epoch full run.
# Fix (2026-10-08): `grep -c` prints 0 AND exits 1 when nothing matches, so `|| echo 1` turned NAN into "0\n1"
# and `[ "$NAN" -eq 0 ]` failed -> SMOKE_PASS=no -> the full run was never launched. Counts are now normalised.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=/home/dgut/runs/detect/runs_cont_v1/smoke_b16
MAN=$L/smoke_b16_manifest.json
ROWS=$(wc -l < "$R/results.csv" 2>/dev/null); ROWS=${ROWS:-0}
NAN=$(grep -ci nan "$R/results.csv" 2>/dev/null); NAN=${NAN:-0}
PT=$(ls "$R/weights/"*.pt 2>/dev/null | wc -l); PT=${PT:-0}
echo "rows=$ROWS nan=$NAN pt=$PT manifest=$([ -f "$MAN" ] && echo yes || echo no)"
PASS=no
if [ "$ROWS" -ge 3 ] && [ "$NAN" -eq 0 ] && [ "$PT" -ge 2 ] && [ -f "$MAN" ]; then PASS=yes; fi
echo "SMOKE_PASS=$PASS"
[ "$PASS" = "yes" ] || exit 4
if [ -f "$L/full_cont_e20_manifest.json" ]; then echo "full run manifest already exists - refusing"; exit 3; fi
cd "$B" || exit 3
echo "=== launching full 20-epoch run $(date -Is) batch=16 workers=8"
setsid nohup /usr/bin/python3 "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
  --recipe-json "$B/eth_only_v1_official_recipe.json" \
  --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
  --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
  --project runs_cont_v1 --name full_cont_e20 --out-manifest "$L/full_cont_e20_manifest.json" \
  --batch 16 --mem-cap-gib 24 --workers 8 --device 0 \
  > "$L/full_cont_e20.log" 2>&1 < /dev/null &
sleep 45
echo "alive=$(pgrep -fc 'train_eth_only_v1[.]py') log=$(stat -c%s $L/full_cont_e20.log) B"
tr '\r' '\n' < "$L/full_cont_e20.log" | tail -4
