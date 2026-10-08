#!/bin/bash
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=/home/dgut/runs/detect/runs_cont_v1/smoke_b16
for i in $(seq 1 25); do pgrep -f "train_eth_only_v1[.]py" > /dev/null || break; sleep 20; done
echo "=== $(date -Is) alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- results.csv:"; [ -f "$R/results.csv" ] && cut -d, -f1-8 "$R/results.csv" && echo "nan=$(grep -ci nan "$R/results.csv")"
echo "--- weights:"; ls -l "$R/weights/" 2>/dev/null
echo "--- final log lines:"; tr '\r' '\n' < "$L/smoke_b16.log" | grep -E "all |Results saved|epochs completed|optimizer stripped|val:" | tail -8
echo "--- manifest:"; [ -f "$L/smoke_b16_manifest.json" ] && /usr/bin/python3 -c "
import json; m=json.load(open('$L/smoke_b16_manifest.json'))
print(json.dumps({k:m.get(k) for k in ('smoke','diagnostic_only','eligible_for_final_report','batch','batch_attempts_tried','runtime','endpoint','steps','save_dir')}, indent=1))
print('selection:', json.dumps(m.get('selection'))[:300])
" || echo "no manifest yet"
echo "--- reload check:"; /usr/bin/python3 -c "
import torch, glob
w = sorted(glob.glob('$R/weights/*.pt'))
print('files', [x.split('/')[-1] for x in w])
if w:
    ck = torch.load(w[-1], map_location='cpu', weights_only=False)
    sd = ck['model'].state_dict() if hasattr(ck.get('model'), 'state_dict') else {}
    print('reload OK, tensors', len(sd), 'epoch', ck.get('epoch'), 'best_fitness', ck.get('best_fitness'))
" 2>&1 | tail -3
