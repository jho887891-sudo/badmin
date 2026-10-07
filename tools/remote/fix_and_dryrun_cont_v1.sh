#!/bin/bash
# (1) correct the builder's completeness gate so an excluded unlabeled image is accounted for, not hidden;
# (2) run the pre-training dry-run (weights gate + dataset gate + resolved recipe).
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
echo "=== audit gate fix $(date -Is)"
/usr/bin/python3 - <<'EOF'
import json
p = "/home/dgut/.dsh-bench/yolo26s_v2_full_eth_cont_v1_data_audit.json"
a = json.load(open(p))
excluded = len(a["missing_labels"])
a["excluded_unlabeled_positives"] = excluded
a["excluded_unlabeled_positives_detail"] = a["missing_labels"]
a["GATE_manifest_complete"] = (a["manifest_rows"] + excluded == a["plan_rows"])
a["GATE_manifest_complete_definition"] = ("manifest_rows + excluded_unlabeled_positives == plan_rows; the one excluded "
                                          "row is an official ETH image with no label file, reported here and in the "
                                          "final report")
json.dump(a, open(p, "w"), indent=1, sort_keys=True)
print("plan", a["plan_rows"], "rows", a["manifest_rows"], "excluded_unlabeled", excluded,
      "gate_complete", a["GATE_manifest_complete"], "all_pass", a["GATE_all_pass"])
EOF
echo "=== contract identity on the remote:"
sha256sum "$B/yolo26s_v2_full_eth_cont_v1.yaml" "$B/train_eth_only_v1.py"
echo "=== dry-run:"
bash "$L/run_dryrun_cont_v1.sh" 2>&1 | tail -25
echo "=== gzip the manifest for the transfer:"
gzip -9 -c "$B/yolo26s_v2_full_eth_cont_v1_train_manifest.csv" > "$L/cont_v1_manifest.csv.gz"
ls -l "$L/cont_v1_manifest.csv.gz"
sha256sum "$L/cont_v1_manifest.csv.gz" "$B/yolo26s_v2_full_eth_cont_v1_train_manifest.csv"
