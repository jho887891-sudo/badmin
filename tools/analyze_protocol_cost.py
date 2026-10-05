#!/usr/bin/env python3
"""Quantify the cost of the leakage-free protocol by location (read-only, supplementary to the V2 decision).

The ETH-only V1 protocol had to remove every image that appears in one of our frozen evaluation sets. Three ETH
locations are consumed entirely by our evaluation frames, so V1 never trained on them at all. This tool turns
that into numbers: how much supervision was removed, how much of the fair real subset the removed locations
carry, and how the model behaves on trained versus dropped locations.

Inputs are frozen artifacts only; nothing here can change any dataset, recipe or evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as err  # noqa: E402  (canonical leakage classes and statistics)

DEFAULT_RECIPE = "outputs/shuttle_capability/metrics/eth_only_v1_official_recipe.json"
DEFAULT_AUDIT = "outputs/shuttle_capability/metrics/eth_only_v1_data_audit.json"
DEFAULT_INVENTORY = "outputs/shuttle_capability/metrics/eth_data_leakage_inventory.csv"
DEFAULT_FN = "outputs/shuttle_capability/metrics/eth_only_v1_fn_analysis.json"
DEFAULT_CSV = "outputs/shuttle_capability/metrics/eth_only_v1_protocol_cost.csv"
DEFAULT_JSON = "outputs/shuttle_capability/metrics/eth_only_v1_protocol_cost.json"


def location_inventory(recipe_json) -> dict:
    """ETH easy+medium positive frames per location (the trainable pool before our exclusions)."""
    data = json.loads(Path(recipe_json).read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (data["counts"]["positives_by_location"] or {}).items()}


def evaluation_consumption(v1_audit_json) -> dict:
    """Frames per location that our frozen evaluation sets consume (hence cannot be trained on)."""
    data = json.loads(Path(v1_audit_json).read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (data.get("excluded_eval_by_location") or {}).items()}


def v1_training_usage(v1_audit_json) -> dict:
    data = json.loads(Path(v1_audit_json).read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (data.get("train_location_counts") or {}).items()}


def val_gt_by_location(inventory_csv, classes=err.ETH_UNSEEN_CLASSES) -> dict:
    """GT boxes per location on the fair subset val|eth_unseen, from the leakage inventory."""
    keep = set(classes)
    out = {}
    for r in csv.DictReader(open(inventory_csv, encoding="utf-8")):
        if str(r.get("leakage_class")) not in keep:
            continue
        loc = str(r.get("location") or "unknown")
        try:
            gt = int(float(r.get("GT") or 0))
        except ValueError:
            gt = 0
        out[loc] = out.get(loc, 0) + gt
    return out


def fn_by_location(fn_json) -> dict:
    data = json.loads(Path(fn_json).read_text(encoding="utf-8"))
    return {str(r["key"]): {"gt": int(r["gt"]), "fn": int(r["fn"])} for r in data.get("by_location", [])}


def recoverable_by_location(fn_json) -> dict:
    data = json.loads(Path(fn_json).read_text(encoding="utf-8"))
    prof = ((data.get("recoverable_fn") or {}).get("recoverable") or {})
    return {str(k): int(v) for k, v in (prof.get("location_profile") or {}).items()}


def protocol_cost_table(inventory, consumed, training_usage, val_gt, fn_loc, recoverable) -> list:
    rows = []
    for loc in sorted(inventory, key=lambda k: (-inventory[k], k)):
        total = int(inventory[loc])
        used = int(consumed.get(loc, 0))
        remaining = max(0, total - used)
        fn = fn_loc.get(loc) or {}
        rows.append({
            "location": loc,
            "eth_easy_medium_frames": total,
            "consumed_by_our_evaluation": used,
            "remaining_trainable": remaining,
            "fully_consumed": remaining == 0,
            "v1_train_frames_used": int(training_usage.get(loc, 0)),
            "val_unseen_gt": int(val_gt.get(loc, 0)),
            "v1_fn": int(fn.get("fn", 0)),
            "v1_fn_rate": (fn.get("fn", 0) / fn.get("gt", 0)) if fn.get("gt") else None,
            "recoverable_fn": int(recoverable.get(loc, 0)),
        })
    return rows


def non_eth_domains(fn_loc, eth_locations) -> list:
    """val|eth_unseen GT per non-ETH domain (our synthetic / iPhone frames).

    Source is the FN analysis by_location, which takes locations from the dataset manifest; the leakage
    inventory leaves the location empty for our own frames, so aggregating it there would be meaningless."""
    out = []
    for loc in sorted(set(fn_loc) - set(eth_locations)):
        gt = int((fn_loc.get(loc) or {}).get("gt", 0))
        fn = int((fn_loc.get(loc) or {}).get("fn", 0))
        out.append({"domain": loc, "val_unseen_gt": gt, "v1_fn": fn,
                    "v1_fn_rate": (fn / gt) if gt else None})
    return out


def summarise(rows, recoverable_global=None) -> dict:
    dropped = [r for r in rows if r["fully_consumed"]]
    kept = [r for r in rows if r["v1_train_frames_used"] > 0]
    def agg(sel, key):
        return sum(r[key] for r in sel)
    dropped_gt = agg(dropped, "val_unseen_gt")
    kept_gt = agg(kept, "val_unseen_gt")
    dropped_fn = agg(dropped, "v1_fn")
    kept_fn = agg(kept, "v1_fn")
    return {
        "locations_total": len(rows),
        "locations_fully_consumed": [r["location"] for r in dropped],
        "frames_consumed_total": agg(rows, "consumed_by_our_evaluation"),
        "frames_remaining_trainable_total": agg(rows, "remaining_trainable"),
        "dropped_locations_val_gt": dropped_gt,
        "dropped_locations_fn": dropped_fn,
        "dropped_locations_fn_rate": (dropped_fn / dropped_gt) if dropped_gt else None,
        "trained_locations_val_gt": kept_gt,
        "trained_locations_fn": kept_fn,
        "trained_locations_fn_rate": (kept_fn / kept_gt) if kept_gt else None,
        "recoverable_fn_total_in_eth_locations": agg(rows, "recoverable_fn"),
        "recoverable_fn_total_global": (int(recoverable_global) if recoverable_global is not None
                                        else agg(rows, "recoverable_fn")),
        "recoverable_share_of_eth_locations":
            (agg(dropped, "recoverable_fn") / agg(rows, "recoverable_fn"))
            if agg(rows, "recoverable_fn") else None,
        "recoverable_share_of_global":
            (agg(dropped, "recoverable_fn") / int(recoverable_global)) if recoverable_global else None,
        "note_trained_locations_val_gt_zero":
            "val|eth_unseen contains no GT from any location this model trained on, so its recall is a pure "
            "out-of-location (plus cross-domain) number; a trained-location FN rate does not exist.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quantify the leakage-free protocol cost by location (read-only)")
    ap.add_argument("--recipe", default=DEFAULT_RECIPE)
    ap.add_argument("--v1-audit", default=DEFAULT_AUDIT)
    ap.add_argument("--inventory", default=DEFAULT_INVENTORY)
    ap.add_argument("--fn", default=DEFAULT_FN)
    ap.add_argument("--out-csv", default=DEFAULT_CSV)
    ap.add_argument("--out-json", default=DEFAULT_JSON)
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    inventory = location_inventory(repo / a.recipe)
    consumed = evaluation_consumption(repo / a.v1_audit)
    usage = v1_training_usage(repo / a.v1_audit)
    val_gt = val_gt_by_location(repo / a.inventory)
    fn_loc = fn_by_location(repo / a.fn)
    rec = recoverable_by_location(repo / a.fn)

    rows = protocol_cost_table(inventory, consumed, usage, val_gt, fn_loc, rec)
    global_rec = int((((json.loads((repo / a.fn).read_text(encoding="utf-8")).get("recoverable_fn") or {})
                       .get("recoverable") or {}).get("count") or 0))
    summary = summarise(rows, recoverable_global=global_rec)
    summary["non_eth_domains"] = non_eth_domains(fn_loc, inventory)
    out = {
        "scope": "supplementary read-only analysis: cost of excluding every evaluation image from training",
        "inputs": {"recipe": a.recipe, "v1_audit": a.v1_audit, "inventory": a.inventory, "fn": a.fn},
        "summary": summary,
        "by_location": rows,
        "headline": ("%d of the %d ETH locations are consumed 100%% by our evaluation sets: %d frames are "
                     "unusable for training and only %d trainable frames remain. The fair subset carries %d GT "
                     "from those dropped locations (FN rate %.4f), %d GT from our own synthetic/iPhone domains, "
                     "and exactly %d GT from locations this model trained on - so its recall is an "
                     "out-of-location number and no trained-location FN rate exists. %d of the %d recoverable "
                     "FNs (%.1f%%) sit in the dropped locations."
                     % (len(summary["locations_fully_consumed"]), summary["locations_total"],
                        summary["frames_consumed_total"], summary["frames_remaining_trainable_total"],
                        summary["dropped_locations_val_gt"], summary["dropped_locations_fn_rate"] or 0.0,
                        sum(d["val_unseen_gt"] for d in summary["non_eth_domains"]),
                        summary["trained_locations_val_gt"],
                        summary["recoverable_fn_total_in_eth_locations"],
                        summary["recoverable_fn_total_global"],
                        100.0 * (summary["recoverable_share_of_global"] or 0.0))),
    }
    Path(repo / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (repo / a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = list(rows[0].keys()) if rows else ["location"]
    with (repo / a.out_csv).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[cost] " + out["headline"])
    print("[cost] wrote %s and %s" % (a.out_json, a.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
