#!/usr/bin/env python3
"""Read-only false-negative (FN) error analysis for the ETH-only YOLO26s V1 model.

Scope: the fair real subset ``val|eth_unseen`` of the V1 eth-only checkpoint, measured at the operating
confidence (0.25) and IoU (0.5) -- the same operating point as the canonical comparison table
``eth_vs_ours_eth_only_v1.csv`` (GT 495, TP 81, FN 414).

Nothing here is allowed to change any dataset membership, manifest, evaluator or test: it only reads frozen
dumps and joins them against the frozen manifest. Every helper (GT decode, matcher, Poisson/Wilson CI) is
imported from ``tools/analyze_eth_only_v1_errors.py`` -- statistics are never re-implemented here.

Blocks reported:
  * ``by_location``   -- GT/FN per manifest ``location`` (the physical court / recording session)
  * ``by_size``       -- GT/FN per canonical 640-equivalent size bucket, plus the ``<8`` px aggregate
  * ``by_source``     -- GT/FN per manifest ``source`` (the domain of the frame)
  * ``recoverable_fn`` -- GT the ETH official model hits while V1 misses it (the recoverable recall gap),
                          with size/location profiles, plus the symmetric case
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as err  # noqa: E402  (canonical matcher + statistics)

SMALL_BUCKETS = ("<4", "4-6", "6-8")
SMALL_AGG_KEY = "<8"
UNKNOWN = "?"
CSV_COLUMNS = ("scope", "key", "gt", "fn", "fn_rate", "extra")

DEFAULT_V1_DUMP = "_scratch_localization_audit/loc_eth_only_v1_best_val.json"
DEFAULT_ETH_DUMP = "_scratch_localization_audit/loc_eth_official_val.json"
DEFAULT_DATASET_MANIFEST = "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv"
DEFAULT_LEAKAGE_INVENTORY = "outputs/shuttle_capability/metrics/eth_data_leakage_inventory.csv"
DEFAULT_OUT_CSV = "outputs/shuttle_capability/metrics/eth_only_v1_fn_analysis.csv"
DEFAULT_OUT_JSON = "outputs/shuttle_capability/metrics/eth_only_v1_fn_analysis.json"


# --------------------------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------------------------
def _norm(path) -> str:
    """Case- and separator-insensitive image key (Windows manifest rows mix backslashes and case)."""
    return str(path or "").replace("\\", "/").lower()


def _rate(fn: int, gt: int):
    """FN rate, None-safe at gt == 0 (an empty stratum has no measurable rate)."""
    return (fn / gt) if gt else None


def manifest_by_image(dataset_manifest) -> dict:
    """Map normalised image path -> manifest row (first row wins)."""
    idx = {}
    if not dataset_manifest:
        return idx
    for r in err.load_rows(dataset_manifest):
        img = r.get("image")
        if img:
            idx.setdefault(_norm(img), r)
    return idx


def _row(key, gt, fn, extra=None) -> dict:
    gt, fn = int(gt), int(fn)
    return {"key": str(key), "gt": gt, "fn": fn, "fn_rate": _rate(fn, gt),
            "fn_rate_ci95": err.wilson_ci95(fn, gt), "extra": extra or {}}


def _sorted_rows(rows) -> list:
    """Worst first: more FN, then more GT, then alphabetical -- deterministic for tests."""
    return sorted(rows, key=lambda r: (-r["fn"], -r["gt"], r["key"]))


def _group_field(per_gt_rows, dataset_manifest, field) -> list:
    idx = manifest_by_image(dataset_manifest)
    gt_by, fn_by, imgs = Counter(), Counter(), {}
    for r in per_gt_rows:
        man = idx.get(_norm(r.get("image"))) or {}
        k = str(man.get(field) or UNKNOWN)
        gt_by[k] += 1
        imgs.setdefault(k, set()).add(_norm(r.get("image")))
        if not bool(r.get("matched")):
            fn_by[k] += 1
    return _sorted_rows([_row(k, gt_by[k], fn_by.get(k, 0), {"images": len(imgs[k]), "field": field})
                         for k in gt_by])


def _bucket_order(per_gt_rows) -> list:
    """Canonical evaluator bucket order, restricted to buckets actually present in the data."""
    present = set(str(r.get("bucket") or UNKNOWN) for r in per_gt_rows)
    ordered = [b for b in err.ev.BUCKET_LABELS if b in present]
    return ordered + sorted(present - set(ordered))


# --------------------------------------------------------------------------------------------------
# block 1-3: strata of the V1 false negatives
# --------------------------------------------------------------------------------------------------
def fn_by_location(per_gt_rows, dataset_manifest) -> list:
    """GT/FN counts and FN rate per manifest ``location`` (court / recording session)."""
    return _group_field(per_gt_rows, dataset_manifest, "location")


def fn_by_source(per_gt_rows, dataset_manifest) -> list:
    """GT/FN counts and FN rate per manifest ``source`` (the domain of the frame)."""
    return _group_field(per_gt_rows, dataset_manifest, "source")


def fn_by_size(per_gt_rows) -> list:
    """GT/FN counts and FN rate per 640-equivalent size bucket, plus the ``<8`` px aggregate."""
    gt_by, fn_by = Counter(), Counter()
    for r in per_gt_rows:
        b = str(r.get("bucket") or UNKNOWN)
        gt_by[b] += 1
        if not bool(r.get("matched")):
            fn_by[b] += 1
    rows = [_row(b, gt_by[b], fn_by.get(b, 0), {"bucket": b}) for b in _bucket_order(per_gt_rows)]
    gt_small = sum(gt_by.get(b, 0) for b in SMALL_BUCKETS)
    fn_small = sum(fn_by.get(b, 0) for b in SMALL_BUCKETS)
    rows.append(_row(SMALL_AGG_KEY, gt_small, fn_small,
                     {"bucket": SMALL_AGG_KEY, "aggregate": True, "buckets": list(SMALL_BUCKETS)}))
    return rows


# --------------------------------------------------------------------------------------------------
# block 4: the recoverable FN (ETH official hits / V1 misses) and its symmetric counterpart
# --------------------------------------------------------------------------------------------------
def recoverable_fn_profile(eth_rows, v1_rows, dataset_manifest) -> dict:
    """GT that the ETH official model hits and V1 misses: count, FN rate, size and location profile.

    ``fn_rate_of_eth_tp`` = recoverable / (GT the ETH official model hits): the share of ETH's own hits
    that V1 throws away. ``share_of_v1_fn`` = recoverable / all V1 FN: how much of the V1 recall gap a
    detector of ETH's operating quality could recover. The symmetric case (V1 hits, ETH misses) is
    reported with the same denominators swapped so the two directions cannot be confused.
    """
    idx = manifest_by_image(dataset_manifest)
    key = lambda r: (str(r.get("image")), int(r.get("gt_index", 0)))  # noqa: E731
    eth = {key(r): r for r in eth_rows}
    v1 = {key(r): r for r in v1_rows}
    common = sorted(set(eth) & set(v1))
    hit = lambda r: bool(r.get("matched"))  # noqa: E731

    bucket_of = {}
    for k in common:
        bucket_of[k] = str((v1.get(k) or eth.get(k) or {}).get("bucket") or UNKNOWN)

    def location_of(k):
        man = idx.get(_norm(k[0])) or {}
        return str(man.get("location") or UNKNOWN)

    eth_tp = [k for k in common if hit(eth[k])]
    v1_tp = [k for k in common if hit(v1[k])]
    v1_fn = [k for k in common if not hit(v1[k])]
    eth_fn = [k for k in common if not hit(eth[k])]
    eth_tp_v1_fn = [k for k in eth_tp if not hit(v1[k])]
    v1_tp_eth_fn = [k for k in v1_tp if not hit(eth[k])]

    def _profile(keys, denom_label, denom_keys, other_label, other_keys):
        sizes = Counter(bucket_of[k] for k in keys)
        locs = Counter(location_of(k) for k in keys)
        size_denom = Counter(bucket_of[k] for k in denom_keys)
        loc_denom = Counter(location_of(k) for k in denom_keys)
        order = _bucket_order([{"bucket": b} for b in set(size_denom) | set(sizes)])
        size_profile = {b: sizes.get(b, 0) for b in order}
        loc_order = sorted(set(locs) | set(loc_denom), key=lambda x: (-locs.get(x, 0), x))
        loc_profile = {b: locs.get(b, 0) for b in loc_order}
        return {
            "count": len(keys),
            "fn_rate_of_%s" % denom_label: _rate(len(keys), len(denom_keys)),
            "share_of_%s" % other_label: _rate(len(keys), len(other_keys)),
            "size_profile": size_profile,
            "size_denominator": {b: size_denom.get(b, 0) for b in order},
            "size_fn_rate": {b: _rate(size_profile[b], size_denom.get(b, 0)) for b in order},
            "location_profile": loc_profile,
            "location_denominator": {b: loc_denom.get(b, 0) for b in loc_order},
            "location_fn_rate": {b: _rate(loc_profile[b], loc_denom.get(b, 0)) for b in loc_order},
            "examples": ["%s#%d" % k for k in keys[:20]],
        }

    return {
        "common_gt": len(common),
        "eth_tp_total": len(eth_tp),
        "v1_tp_total": len(v1_tp),
        "v1_fn_total": len(v1_fn),
        "eth_fn_total": len(eth_fn),
        "recoverable": _profile(eth_tp_v1_fn, "eth_tp", eth_tp, "v1_fn", v1_fn),
        "symmetric": _profile(v1_tp_eth_fn, "v1_tp", v1_tp, "eth_fn", eth_fn),
    }


# --------------------------------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------------------------------
def _headline(gt_total, fn_total, fn_rate, by_size, by_location, rec, model_tag="the model") -> str:
    if not gt_total:
        return "val|eth_unseen: no GT rows -- nothing to analyse"
    small = next((r for r in by_size if r["key"] == SMALL_AGG_KEY), None)
    worst = by_location[0] if by_location else None
    worst_txt = ("%s %.4f (%d/%d)" % (worst["key"], worst["fn_rate"], worst["fn"], worst["gt"])
                 if worst is not None else "n/a")
    return ("val|eth_unseen (fair real ETH-unseen subset): %s misses %d of %d GT "
            "(FN rate %.4f); worst location %s; <8px %d/%d missed (FN rate %s); of the %d FN, %d are "
            "recoverable (ETH official hits them, %.4f of ETH's hits) while only %d go the other way"
            % (model_tag, fn_total, gt_total, fn_rate or 0.0, worst_txt,
               small["fn"] if small else 0, small["gt"] if small else 0,
               ("%.4f" % small["fn_rate"]) if (small and small["fn_rate"] is not None) else "n/a",
               fn_total, rec["recoverable"]["count"],
               rec["recoverable"]["fn_rate_of_eth_tp"] or 0.0, rec["symmetric"]["count"]))


def build_report(v1_rows, eth_rows, dataset_manifest, inputs=None, model_tag="the model") -> dict:
    """Assemble the four analysis blocks plus the top-level headline for one operating point."""
    summary = err.summarize_false_negatives(v1_rows)
    by_location = fn_by_location(v1_rows, dataset_manifest)
    by_size = fn_by_size(v1_rows)
    by_source = fn_by_source(v1_rows, dataset_manifest)
    rec = recoverable_fn_profile(eth_rows or [], v1_rows or [], dataset_manifest)
    gt_total, fn_total = summary["gt_total"], summary["fn_total"]
    fn_rate = _rate(fn_total, gt_total)
    return {
        "fn_total_ci95": err.poisson_ci95(fn_total) if gt_total else [None, None],
        "fn_rate_ci95": err.wilson_ci95(fn_total, gt_total),
        "scope": ("read-only V1 false-negative error analysis on val|eth_unseen; cannot change any "
                  "dataset membership, manifest, evaluator or artifact"),
        "inputs": inputs or {},
        "gt_total": gt_total,
        "fn_total": fn_total,
        "tp_total": gt_total - fn_total,
        "fn_rate": fn_rate,
        "by_location": by_location,
        "by_size": by_size,
        "by_source": by_source,
        "recoverable_fn": rec,
        "headline": _headline(gt_total, fn_total, fn_rate, by_size, by_location, rec, model_tag),
        "model_tag": model_tag,
    }


def flatten_report(report) -> list:
    """Long-format rows: scope,key,gt,fn,fn_rate,extra (extra is compact JSON)."""
    rows = []

    def emit(scope, key, gt, fn, extra=None):
        rows.append({"scope": scope, "key": str(key), "gt": int(gt), "fn": int(fn),
                     "fn_rate": _rate(int(fn), int(gt)),
                     "extra": json.dumps(extra or {}, sort_keys=True, ensure_ascii=False)})

    emit("summary", "val|eth_unseen", report.get("gt_total", 0), report.get("fn_total", 0),
         {"tp_total": report.get("tp_total"), "fn_rate": report.get("fn_rate"),
          "fn_rate_ci95": report.get("fn_rate_ci95"), "fn_total_ci95": report.get("fn_total_ci95")})
    for scope in ("by_location", "by_size", "by_source"):
        for r in report.get(scope) or []:
            emit(scope, r["key"], r["gt"], r["fn"],
                 dict(r["extra"], fn_rate_ci95=r.get("fn_rate_ci95")))
    rec = report.get("recoverable_fn") or {}
    for side, label in (("recoverable", "eth_tp_v1_fn"), ("symmetric", "v1_tp_eth_fn")):
        d = rec.get(side) or {}
        if not d:
            continue
        denom_label = "fn_rate_of_eth_tp" if side == "recoverable" else "fn_rate_of_v1_tp"
        other = "share_of_v1_fn" if side == "recoverable" else "share_of_eth_fn"
        denom = rec.get("eth_tp_total" if side == "recoverable" else "v1_tp_total", 0)
        emit("recoverable", label, denom, d.get("count", 0),
             {"fn_rate": d.get(denom_label), other: d.get(other), "side": label})
        for b, c in (d.get("size_profile") or {}).items():
            emit("recoverable_size", b, (d.get("size_denominator") or {}).get(b, 0), c,
                 {"side": label, "fn_rate": (d.get("size_fn_rate") or {}).get(b)})
        for b, c in (d.get("location_profile") or {}).items():
            emit("recoverable_location", b, (d.get("location_denominator") or {}).get(b, 0), c,
                 {"side": label, "fn_rate": (d.get("location_fn_rate") or {}).get(b)})
    return rows


def write_fn_csv(path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})
    return path


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only false-negative analysis of the ETH-only V1 model on val|eth_unseen")
    ap.add_argument("--v1-dump", default=DEFAULT_V1_DUMP)
    ap.add_argument("--eth-dump", default=DEFAULT_ETH_DUMP)
    ap.add_argument("--dataset-manifest", default=DEFAULT_DATASET_MANIFEST)
    ap.add_argument("--leakage-inventory", default=DEFAULT_LEAKAGE_INVENTORY)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--conf-thr", type=float, default=0.25)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    manifest = repo / a.dataset_manifest
    v1_dump, eth_dump = repo / a.v1_dump, repo / a.eth_dump
    leakage = repo / a.leakage_inventory
    missing = [str(p) for p in (v1_dump, eth_dump, manifest, leakage) if not p.is_file()]
    if missing:
        print("[fn] missing required input(s): %s" % ", ".join(missing))
        return 2

    v1_rows = err.build_val_match_table(v1_dump, manifest, leakage, err.ETH_UNSEEN_CLASSES, split="val",
                                        iou_thr=a.iou_thr, conf_thr=a.conf_thr)
    eth_rows = err.build_val_match_table(eth_dump, manifest, leakage, err.ETH_UNSEEN_CLASSES, split="val",
                                         iou_thr=a.iou_thr, conf_thr=a.conf_thr)
    inputs = {
        "v1_dump": str(v1_dump.resolve()),
        "eth_dump": str(eth_dump.resolve()),
        "dataset_manifest": str(manifest.resolve()),
        "leakage_inventory": str(leakage.resolve()),
        "split": "val",
        "set": "val|eth_unseen",
        "leakage_classes": list(err.ETH_UNSEEN_CLASSES),
        "iou_thr": a.iou_thr,
        "conf_thr": a.conf_thr,
        "v1_per_gt_rows": len(v1_rows),
        "eth_per_gt_rows": len(eth_rows),
    }
    tag = "the model"
    try:
        tag = str(json.loads(v1_dump.read_text(encoding="utf-8")).get("checkpoint") or tag)
    except (OSError, ValueError):
        pass
    report = build_report(v1_rows, eth_rows, manifest, inputs, model_tag=tag)
    rows = flatten_report(report)

    out_json, out_csv = repo / a.out_json, repo / a.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    write_fn_csv(out_csv, rows)
    print("[fn] " + report["headline"])
    print("[fn] wrote %s (%d rows) and %s" % (a.out_json, len(rows), a.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
