#!/bin/bash
/usr/bin/python3 - <<'EOF'
import json
a = json.load(open('/tmp/test_audit.json'))
keys = ('manifest_rows','positive_images','negative_images','boxes_total','images_by_set','difficulty_counts',
        'location_counts','size_bucket_counts_boxes','duplicate_sha256_groups','missing_labels',
        'unreadable_images','non_empty_negative_labels','prohibited_source_hits','GATE_missing_images_zero',
        'GATE_unreadable_images_zero','GATE_non_empty_negative_labels_zero','GATE_prohibited_hits_zero',
        'GATE_all_pass','elapsed_s','plan_rows')
print(json.dumps({k: a.get(k) for k in keys}, indent=1))
EOF
echo "--- manifest head:"
head -2 /tmp/test_manifest.csv
echo "--- row count: $(($(wc -l < /tmp/test_manifest.csv) - 1))"
