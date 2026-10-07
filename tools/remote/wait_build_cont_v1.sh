#!/bin/bash
# Wait for the pool build to finish, then print the final summary and audit gates.
L=/home/dgut/.dsh-bench/cont_v1
B=/home/dgut/.dsh-bench
for i in $(seq 1 40); do
  if ! pgrep -f "build_cont_v1_pool[.]py" > /dev/null; then break; fi
  sleep 30
done
echo "=== $(date -Is) build process alive: $(pgrep -fc 'build_cont_v1_pool[.]py')"
tail -6 "$L/build_cont_v1.log"
echo "--- manifest:"
wc -l < "$B/yolo26s_v2_full_eth_cont_v1_train_manifest.csv"
ls -l "$B/yolo26s_v2_full_eth_cont_v1_train_manifest.csv" "$B/yolo26s_v2_full_eth_cont_v1_data_audit.json" 2>/dev/null
echo "--- lists:"
ls -l "$B/data/yolo26s_v2_full_eth_cont_v1/" 2>/dev/null
wc -l "$B/data/yolo26s_v2_full_eth_cont_v1/train.txt" "$B/data/yolo26s_v2_full_eth_cont_v1/val.txt" 2>/dev/null
echo "--- audit gates:"
/usr/bin/python3 - <<'EOF'
import json
p = "/home/dgut/.dsh-bench/yolo26s_v2_full_eth_cont_v1_data_audit.json"
try:
    a = json.load(open(p))
except Exception as exc:
    print("audit not readable yet:", exc); raise SystemExit
keys = ("manifest_rows", "positive_images", "negative_images", "boxes_total", "difficulty_counts",
        "duplicate_sha256_groups", "duplicate_sha256_extra_rows", "missing_labels", "unreadable_images",
        "non_empty_negative_labels", "prohibited_source_hits", "GATE_all_pass", "GATE_manifest_complete",
        "size_bucket_counts_images", "elapsed_s", "manifest_sha256")
print(json.dumps({k: a.get(k) for k in keys}, indent=1))
print("locations:", sorted(a.get("location_counts", {})))
print("lists:", a.get("lists"))
EOF
