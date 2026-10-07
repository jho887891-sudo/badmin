#!/bin/bash
set -u
L=/home/dgut/.dsh-bench/cont_v1
R=/home/T7/dgut/robot_sim/eth_shuttle_detection
rm -f /tmp/t2_manifest.csv /tmp/t2_audit.json
echo "=== start $(date -Is)"
timeout 900 /usr/bin/python3 "$L/build_cont_v1_pool.py" --root "$R"   --out-manifest /tmp/t2_manifest.csv --out-audit /tmp/t2_audit.json --limit 400 --workers 12 2>&1 | tail -8
echo "rc=$?"
/usr/bin/python3 - <<'EOF'
import json
a = json.load(open('/tmp/t2_audit.json'))
print("rows", a["manifest_rows"], "elapsed_s", a["elapsed_s"], "rows/s", round(a["manifest_rows"]/max(a["elapsed_s"],1e-9),2))
print("gates", {k: a[k] for k in a if k.startswith("GATE")})
print("boxes", a["boxes_total"], "buckets", a["size_bucket_counts_images"])
EOF
echo "=== end $(date -Is)"
