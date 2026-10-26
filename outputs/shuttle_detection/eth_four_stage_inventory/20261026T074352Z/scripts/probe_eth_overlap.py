"""Read-only probe 2: resolve val/train overlap, the role-vs-is_positive off-by-one, and frozen index schema."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

TRAIN = Path("data/eth_real_hardneg_v2_train_manifest.csv")
VAL = Path("data/eth_real_hardneg_v2_val_manifest.csv")
FROZEN = Path("data/clean_outdoor_cp_v1/frozen_eval_index.csv")


def load(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main():
    tcols, train = load(TRAIN)
    vcols, val = load(VAL)
    fcols, frozen = load(FROZEN)
    out = {}

    tr_paths = {r["image"] for r in train}
    tr_sha = {r["sha256"] for r in train}
    tr_stem = {Path(r["image"]).stem for r in train}
    va_paths = {r["image"] for r in val}
    va_sha = {r["sha256"] for r in val}
    va_stem = {Path(r["image"]).stem for r in val}
    out["train_rows"] = len(train)
    out["train_distinct_images"] = len(tr_paths)
    out["train_distinct_sha"] = len(tr_sha)
    out["val_rows"] = len(val)
    out["val_distinct_images"] = len(va_paths)
    out["overlap_val_in_train_by_path"] = len(va_paths & tr_paths)
    out["overlap_val_in_train_by_sha"] = len(va_sha & tr_sha)

    # off-by-one hunt
    odd = [r for r in train if r["role"] == "official_positives" and str(r["is_positive"]).strip() != "True"]
    out["official_positives_but_not_is_positive"] = len(odd)
    out["examples"] = [{k: r[k] for k in ("image", "role", "is_positive", "is_negative",
                                         "source", "set_dir", "n_boxes")} for r in odd[:3]]

    # duplicate images inside train manifest (repeat_index column hints at repeats)
    dup = Counter(r["image"] for r in train)
    out["train_images_with_multiplicity_gt1"] = sum(1 for v in dup.values() if v > 1)
    out["max_multiplicity"] = max(dup.values()) if dup else 0
    out["repeat_index_distribution"] = dict(Counter(str(r.get("repeat_index", "")) for r in train).most_common(6))

    # cross-tab source x set_dir to see where coco rows sit
    xt = Counter((r["source"], r["set_dir"]) for r in train)
    out["source_x_setdir_top"] = [list(k) + [n] for k, n in xt.most_common(20)]

    # positives only, by set_dir
    pos = [r for r in train if str(r.get("is_positive")).strip().lower() in {"true", "1", "yes"}]
    out["positives_rows"] = len(pos)
    out["positives_by_set_dir"] = dict(Counter(r["set_dir"] for r in pos).most_common(30))
    out["positives_by_location"] = dict(Counter(r["location"] for r in pos).most_common(30))
    out["positives_n_boxes_sum"] = sum(int(float(r["n_boxes"] or 0)) for r in pos)

    # frozen eval index schema
    out["frozen_columns"] = fcols
    out["frozen_rows"] = len(frozen)
    out["frozen_first_row"] = {k: frozen[0][k] for k in fcols} if frozen else {}
    fp = {r.get("image", "") for r in frozen}
    fs = {r.get("sha256", "") for r in frozen if r.get("sha256")}
    fst = {Path(x).stem for x in fp if x}
    out["frozen_distinct_paths"] = len(fp)
    out["frozen_distinct_sha"] = len(fs)
    out["frozen_overlap_train_by_path"] = len(fp & tr_paths)
    out["frozen_overlap_train_by_sha"] = len(fs & tr_sha)
    out["frozen_overlap_train_by_stem"] = len(fst & tr_stem)
    out["frozen_sample_paths"] = sorted(fp)[:3]
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
