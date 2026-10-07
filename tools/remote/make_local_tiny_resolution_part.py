"""Build the full-coverage (105 tiny GT) part list for the local resolution-response probe.

Every row carries its absolute image/label path plus the already-measured V2@1024 reference values, so the probe run
can be cross-checked against outputs/shuttle_capability/metrics/tiny_representability_v1.csv.
"""
import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAN = REPO / "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv"
TINY = REPO / "outputs/shuttle_capability/metrics/tiny_representability_v1.csv"
OUT = REPO / "tools/remote/local_tiny_part.csv"


def main() -> int:
    man = {r["image"]: r for r in csv.DictReader(open(MAN, encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(TINY, encoding="utf-8")):
        if float(r["equiv_size_640"]) >= 8.0:
            continue
        m = man.get(r["image"]) or {}
        img = Path(r["image"].replace("\\", "/"))
        lab = Path((m.get("label") or "").replace("\\", "/"))
        assert img.is_file(), "missing image %s" % img
        assert lab.is_file(), "missing label %s" % lab
        rows.append({
            "basename": img.name, "abs_image": str(img), "abs_label": str(lab),
            "source": m.get("source", ""), "bucket": r["bucket"],
            "equiv_size_640": r["equiv_size_640"], "net_px_1024": r["net_px_1024"],
            "local_ref_matched_op": r["matched_op"], "local_ref_best_iou": r["best_iou_weak"],
            "local_ref_best_conf": r["best_conf_weak"],
        })
    rows.sort(key=lambda d: (d["bucket"], d["source"], d["basename"]))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("rows %d -> %s (%d B)" % (len(rows), OUT, OUT.stat().st_size))
    print("by bucket", dict(Counter(r["bucket"] for r in rows)))
    print("by source", dict(Counter(r["source"] for r in rows)))
    print("by (source,bucket)", dict(Counter((r["source"], r["bucket"]) for r in rows)))
    print("V2@1024 reference hits:", sum(1 for r in rows if r["local_ref_matched_op"] == "True"), "/", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
