"""Read-only probe: characterise candidate ETH manifests before any inventory run."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

CANDIDATES = [
    "data/eth_real_hardneg_v2_train_manifest.csv",
    "data/eth_real_hardneg_v2_val_manifest.csv",
    "data/eth_only_v1_train_manifest.csv",
    "data/eth_only_v1_val_manifest.csv",
    "data/yolo26s_v2_full_eth_cont_v1_train_manifest.csv",
    "data/clean_outdoor_cp_v1/frozen_eval_index.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    report = {}
    for rel in CANDIDATES:
        p = Path(rel)
        entry = {"exists": p.is_file()}
        if not entry["exists"]:
            report[rel] = entry
            continue
        entry["bytes"] = p.stat().st_size
        entry["sha256"] = sha256(p)
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        entry["columns"] = fields
        entry["n_data_rows"] = len(rows)
        for col in ("split", "source", "role", "is_positive", "is_negative",
                    "in_official_training", "set_dir", "difficulty"):
            if col in fields:
                entry["count_" + col] = dict(Counter(str(r.get(col, "")).strip() for r in rows).most_common(12))
        # image path resolvability, sampled across the whole file
        key = "image" if "image" in fields else ("path" if "path" in fields else None)
        if key:
            total = len(rows)
            step = max(1, total // 400)
            sample = rows[::step]
            missing = [str(r.get(key)) for r in sample if not Path(str(r.get(key))).is_file()]
            entry["image_key"] = key
            entry["image_sample_checked"] = len(sample)
            entry["image_sample_missing"] = len(missing)
            entry["image_sample_missing_examples"] = missing[:3]
        lkey = "label" if "label" in fields else ("label_path" if "label_path" in fields else None)
        if lkey:
            step = max(1, len(rows) // 400)
            sample = rows[::step]
            missing = [str(r.get(lkey)) for r in sample if not Path(str(r.get(lkey))).is_file()]
            entry["label_key"] = lkey
            entry["label_sample_checked"] = len(sample)
            entry["label_sample_missing"] = len(missing)
            entry["label_sample_missing_examples"] = missing[:3]
        report[rel] = entry
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
