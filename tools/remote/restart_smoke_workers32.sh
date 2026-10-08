#!/bin/bash
# Restart the smoke with 32 dataloader workers: mosaic means 4 file reads per sample, so 32 samples/iteration is
# ~128 reads/iteration; at ~40 ms FUSE latency 8 workers cannot hide the stalls (GPU sat at ~0%).
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
pkill -9 -f "train_eth_only_v1[.]py" 2>/dev/null
sleep 5
echo "alive after kill: $(pgrep -fc 'train_eth_only_v1[.]py')"
rm -rf "$L/runs/detect/runs_cont_v1" "$B/runs_cont_v1"
sed -i "s/--workers 8/--workers 32/" "$L/run_smoke_cont_v1_xfs.sh" "$L/run_full_cont_v1_xfs.sh"
grep -H "workers" "$L/run_smoke_cont_v1_xfs.sh" "$L/run_full_cont_v1_xfs.sh" | tail -2
echo "=== relaunch"
setsid nohup bash "$L/run_smoke_cont_v1_xfs.sh" smoke_e2 > "$L/smoke_xfs2.log" 2>&1 < /dev/null &
echo launched
sleep 240
echo "=== $(date -Is) snapshot"
tr '\r' '\n' < "$L/smoke_xfs2.log" | grep -vE "Scanning" | tail -6
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
