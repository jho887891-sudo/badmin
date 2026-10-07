#!/bin/bash
set -u
L=/home/dgut/.dsh-bench/cont_v1
R=/home/T7/dgut/robot_sim/eth_shuttle_detection
rm -f /tmp/test_manifest.csv /tmp/test_audit.json
echo "=== start $(date -Is)"
timeout 900 /usr/bin/python3 "$L/build_cont_v1_pool.py" --root "$R"   --out-manifest /tmp/test_manifest.csv --out-audit /tmp/test_audit.json --limit 200
echo "rc=$? (4 = expected for a --limit run, because rows != plan)"
echo "--- head of test manifest:"
head -3 /tmp/test_manifest.csv
echo "--- audit gates:"
/usr/bin/python3 -c "import json; a=json.load(open('/tmp/test_audit.json')); print(json.dumps({k:v for k,v in a.items() if k.startswith('GATE') or k in ('manifest_rows','positive_images','negative_images','boxes_total','size_bucket_counts_boxes','duplicate_sha256_groups','missing_labels','prohibited_source_hits','excluded_sets')}, indent=1)[:1400])"
echo "=== end $(date -Is)"
