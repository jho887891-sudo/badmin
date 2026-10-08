#!/bin/bash
B=/home/dgut/.dsh-bench
L=$B/cont_v1
for i in $(seq 1 26); do pgrep -f "train_eth_only_v1[.]py" > /dev/null || break; sleep 20; done
echo "=== $(date -Is) alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- final log lines:"
tr '\r' '\n' < "$L/smoke_b16.log" | grep -vE "Scanning|it/s|s/it" | tail -12
echo "--- results.csv:"
CSV=$(find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name results.csv 2>/dev/null | head -1)
echo "file=$CSV"
if [ -n "$CSV" ]; then head -1 "$CSV" | cut -d, -f1-8; tail -3 "$CSV" | cut -d, -f1-8; echo "nan_count=$(grep -ci nan "$CSV")"; fi
echo "--- weights:"
find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name "*.pt" 2>/dev/null | head -8
echo "--- manifest:"
[ -f "$L/smoke_b16_manifest.json" ] && /usr/bin/python3 -c "
import json; m=json.load(open('$L/smoke_b16_manifest.json'))
print(json.dumps({k:m.get(k) for k in ('experiment','smoke','diagnostic_only','eligible_for_final_report','diagnostic_reasons','batch','batch_attempts_tried','runtime','endpoint','steps','gpu_cap','save_dir','selection')}, indent=1)[:1600])" || echo "manifest not written yet"
