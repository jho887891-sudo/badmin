#!/bin/bash
B=/home/dgut/.dsh-bench
for i in $(seq 1 60); do
  pgrep -f "train_eth_only_v1[.]py" > /dev/null || break
  sleep 20
done
echo "=== $(date -Is) trainer alive: $(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- log (last non- lines):"
tr '\r' '\n' < "$B/cont_v1/smoke_xfs.log" | grep -vE "^(train|val):.*Scanning" | tail -25
echo "--- results.csv:"
CSV=$(find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name results.csv 2>/dev/null | head -1)
echo "file=$CSV"
[ -n "$CSV" ] && head -1 "$CSV" | tr ',' '\n' | head -12 | tr '\n' ' ' && echo && tail -3 "$CSV"
echo "--- weights:"
find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name "*.pt" 2>/dev/null | head -10
echo "--- manifest:"
ls -l "$B/cont_v1/smoke_e2_manifest.json" 2>/dev/null && /usr/bin/python3 -c "
import json; m=json.load(open('$B/cont_v1/smoke_e2_manifest.json'))
print(json.dumps({k:m.get(k) for k in ('experiment','smoke','diagnostic_only','eligible_for_final_report','diagnostic_reasons','batch','batch_attempts_tried','gpu_cap','runtime','endpoint','steps','save_dir')}, indent=1))" 2>/dev/null
