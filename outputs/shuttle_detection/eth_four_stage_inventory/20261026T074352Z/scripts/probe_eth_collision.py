"""Read-only probe 3: identify the train/frozen sha256 collisions and the 8x-repeated train images."""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = Path("data/eth_real_hardneg_v2_train_manifest.csv")
FROZEN = Path("data/clean_outdoor_cp_v1/frozen_eval_index.csv")


def load(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main():
    _, train = load(TRAIN)
    _, frozen = load(FROZEN)
    out = {}

    by_sha_train = defaultdict(list)
    for r in train:
        by_sha_train[r["sha256"]].append(r)

    frozen_by_sha = defaultdict(list)
    for r in frozen:
        frozen_by_sha[r["sha256"]].append(r)

    collide = sorted(set(by_sha_train) & set(frozen_by_sha))
    out["collide_sha_count"] = len(collide)
    detail = []
    for sha in collide[:12]:
        t = by_sha_train[sha]
        fz = frozen_by_sha[sha]
        detail.append({
            "sha256": sha,
            "train_rows": len(t),
            "train_image": t[0]["image"],
            "train_set_dir": t[0]["set_dir"],
            "train_role": t[0]["role"],
            "train_repeat_indexes": [r.get("repeat_index") for r in t],
            "frozen_rows": len(fz),
            "frozen_source_id": fz[0].get("source_id"),
            "frozen_origin": fz[0].get("origin"),
            "frozen_role": fz[0].get("role"),
        })
    out["collide_detail_first12"] = detail

    # are the colliding train images the same ones repeated 8x?
    mult = Counter(r["image"] for r in train)
    repeated = {i for i, n in mult.items() if n > 1}
    collide_images = {by_sha_train[s][0]["image"] for s in collide}
    out["train_images_repeated_gt1"] = len(repeated)
    out["collide_images"] = len(collide_images)
    out["collide_images_subset_of_repeated"] = collide_images <= repeated

    # which origins does the frozen index pull from, and do any originate from ETH itself
    out["frozen_origins"] = dict(Counter(r.get("origin", "") for r in frozen).most_common(15))
    out["frozen_roles"] = dict(Counter(r.get("role", "") for r in frozen).most_common(10))

    # repeated images: what role/source do they carry
    rep_rows = [r for r in train if r["image"] in repeated]
    out["repeated_rows"] = len(rep_rows)
    out["repeated_role_dist"] = dict(Counter(r["role"] for r in rep_rows).most_common(10))
    out["repeated_source_dist"] = dict(Counter(r["source"] for r in rep_rows).most_common(10))
    out["repeated_set_dir_dist"] = dict(Counter(r["set_dir"] for r in rep_rows).most_common(10))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
