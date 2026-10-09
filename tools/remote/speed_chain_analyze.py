#!/usr/bin/env python3
"""speed_chain_analyze.py - validity gate + summary for YOLO26S_INFERENCE_SPEED_OPT_V1.

Pure analysis: reads the per-track JSON records and the per-track 1 s monitor CSVs, decides
VALID / CONTAMINATED_INVALID for each track, computes the A3/A4 drift guard, and writes
speed_chain_manifest.json / speed_chain_validity.json / speed_chain_summary.csv / speed_chain_summary.md.

Contamination rules are PRE-REGISTERED here (see RULES) so they cannot be tuned after seeing results.
No GPU is touched by this file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HISTORICAL_BASELINE_MS = 15.90
USEFUL_MS = 12.72
STRONG_MS = 10.0
EXCELLENT_MS = 8.0
DRIFT_REL_MAX = 0.05

RULES = {
    "gate_gpu_util_max": 10.0,
    "gate_gpu_mem_max": 1200,
    "gate_load1_max": 4.0,
    "gate_compute_apps_max": 0,
    "run_foreign_apps_max": 1,
    "run_mem_ceiling_MiB": 2500,
    "run_mem_sustained_samples": 3,
}

TRACKS = [
    ("A3", "pytorch", "fp32", "no", "speed_pytorch_fp32_A3.json"),
    ("B", "pytorch", "fp16", "no", "speed_pytorch_fp16_clean.json"),
    ("C", "pytorch", "fp16", "yes", "speed_torch_compile_fp16_clean.json"),
    ("D", "tensorrt", "fp16", "n/a", "speed_tensorrt_fp16_static.json"),
    ("E", "tensorrt", "fp16", "n/a", "speed_tensorrt_fp16_dynamic.json"),
    ("A4", "pytorch", "fp32", "no", "speed_pytorch_fp32_A4.json"),
]


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_monitor(path):
    """Monitor CSV columns: t_s,gpu_util,gpu_mem,compute_apps,load1"""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append({"t_s": float(row["t_s"]), "gpu_util": float(row["gpu_util"]),
                            "gpu_mem": float(row["gpu_mem"]), "compute_apps": int(row["compute_apps"]),
                            "load1": float(row["load1"])})
            except (KeyError, ValueError):
                continue
    return out


def contamination(samples):
    """Return (contaminated, reason). Pre-registered rules only."""
    if not samples:
        return True, "NO_MONITOR_DATA"
    reasons = []
    foreign = [s for s in samples if s["compute_apps"] > RULES["run_foreign_apps_max"]]
    if foreign:
        worst = max(s["compute_apps"] for s in foreign)
        reasons.append("foreign_compute_apps=%d (max allowed %d) at t=%ss"
                       % (worst, RULES["run_foreign_apps_max"], foreign[0]["t_s"]))
    over = [s for s in samples if s["gpu_mem"] >= RULES["run_mem_ceiling_MiB"]]
    need = RULES["run_mem_sustained_samples"]
    run = best = 0
    for s in samples:
        run = run + 1 if s in over else 0
        best = max(best, run)
    if best >= need:
        reasons.append("gpu_mem>=%dMiB sustained %d samples (need %d)"
                       % (RULES["run_mem_ceiling_MiB"], best, need))
    return (bool(reasons), "; ".join(reasons) if reasons else "")


def judge(rec, samples):
    """Per-track validity verdict."""
    if rec is None:
        return {"validity": "MISSING", "contamination_reason": "record file absent",
                "monitor_samples": len(samples), "clean_gate_samples": None}
    bad, reason = contamination(samples)
    gate = rec.get("clean_gate") or {}
    gate_ok = bool(gate.get("passed"))
    if not gate_ok:
        bad = True
        reason = (reason + "; " if reason else "") + "clean_gate_not_confirmed"
    return {"validity": "CONTAMINATED_INVALID" if bad else "VALID",
            "contamination_reason": reason,
            "monitor_samples": len(samples),
            "clean_gate_samples": gate,
            "gpu_snapshot_start": rec.get("gpu_snapshot_start"),
            "gpu_snapshot_end": rec.get("gpu_snapshot_end")}


def drift(a3, a4):
    if not a3 or not a4:
        return {"drift_abs_ms": None, "drift_rel": None, "warning": "A3_A4_INCOMPLETE"}
    a, b = float(a3["median_ms"]), float(a4["median_ms"])
    d = abs(b - a)
    rel = d / a if a else None
    return {"A3_median_ms": a, "A4_median_ms": b, "drift_abs_ms": round(d, 4),
            "drift_rel": round(rel, 5) if rel is not None else None,
            "threshold_rel": DRIFT_REL_MAX,
            "warning": "WINDOW_DRIFT_WARNING" if (rel is not None and rel > DRIFT_REL_MAX) else None}


def verdict(median_ms, ref_ms):
    d = median_ms - ref_ms
    rel = d / ref_ms if ref_ms else None
    if median_ms <= EXCELLENT_MS:
        grade = "excellent"
    elif median_ms <= STRONG_MS:
        grade = "strong"
    elif median_ms <= USEFUL_MS:
        grade = "useful"
    else:
        grade = "below_useful"
    return round(d, 4), (round(rel, 5) if rel is not None else None), grade


def build(records_dir, summary_prefix):
    recs, mons, validity = {}, {}, {}
    for tag, _b, _p, _c, fname in TRACKS:
        p = os.path.join(records_dir, fname)
        recs[tag] = read_json(p)
        # The chain writes monitors as speed_<TAG>.monitor.csv while this module names records
        # speed_<backend>_<precision>[_clean].json. On 2026-10-08 that mismatch made every track look
        # like NO_MONITOR_DATA and the whole summary came out CONTAMINATED_INVALID. Accept both names.
        cands = [fname.replace(".json", ".monitor.csv"), "speed_%s.monitor.csv" % tag]
        mon_path = next((os.path.join(records_dir, c) for c in cands
                         if os.path.exists(os.path.join(records_dir, c))), None)
        mons[tag] = read_monitor(mon_path)
        validity[tag] = judge(recs[tag], mons[tag])
        validity[tag]["monitor_file"] = mon_path

    dr = drift(recs.get("A3"), recs.get("A4"))
    # Window validity is a property of the A3/A4 sentinels plus the drift guard, NOT of every track.
    # A transiently contaminated B invalidates B, not the whole window.
    sentinels_ok = (validity["A3"]["validity"] == "VALID" and validity["A4"]["validity"] == "VALID")
    overall = bool(sentinels_ok and dr.get("warning") is None)

    main_rows, audit_rows = [], []
    for tag, backend, prec, comp, _f in TRACKS:
        rec = recs.get(tag)
        if rec is None:
            continue
        ref = recs["A3"]["median_ms"] if recs.get("A3") else HISTORICAL_BASELINE_MS
        d_hist = rec["median_ms"] - HISTORICAL_BASELINE_MS
        d_a3, rel_a3, grade = verdict(float(rec["median_ms"]), float(ref))
        row = {"track": tag, "backend": backend, "precision": prec, "compile": comp,
               "median_ms": round(float(rec["median_ms"]), 4), "p95_ms": round(float(rec["p95_ms"]), 4),
               "p99_ms": round(float(rec["p99_ms"]), 4), "FPS_median": rec.get("FPS_median"),
               "delta_vs_historical_A_ms": round(d_hist, 4), "delta_vs_same_window_A3_ms": d_a3,
               "delta_vs_same_window_A3_rel": rel_a3, "grade_vs_same_window_A3": grade,
               "validity": validity[tag]["validity"],
               "gpu_snapshot_start": rec.get("gpu_snapshot_start"), "gpu_snapshot_end": rec.get("gpu_snapshot_end")}
        if "compile_time_s" in rec:
            row["compile_time_s"] = rec.get("compile_time_s")
            row["graph_break_count"] = rec.get("graph_break_count")
            row["symbolic_shape_warning"] = rec.get("symbolic_shape_warning")
        (main_rows if validity[tag]["validity"] == "VALID" else audit_rows).append(row)

    man = {"experiment": "YOLO26S_INFERENCE_SPEED_OPT_V1", "tracks": [t for t, *_ in TRACKS],
           "records_dir": records_dir, "rules": RULES,
           "historical_reference_ms": HISTORICAL_BASELINE_MS,
           "thresholds_ms": {"useful": USEFUL_MS, "strong": STRONG_MS, "excellent": EXCELLENT_MS},
           "inputs_found": {t: (recs.get(t) is not None) for t, *_ in TRACKS}}
    val = {"each_track_validity": {t: validity[t]["validity"] for t, *_ in TRACKS},
           "clean_gate_samples": {t: validity[t]["clean_gate_samples"] for t, *_ in TRACKS},
           "contamination_reason": {t: validity[t]["contamination_reason"] for t, *_ in TRACKS},
           "monitor_samples": {t: validity[t]["monitor_samples"] for t, *_ in TRACKS},
           "A3_A4_drift": dr, "overall_window_valid": bool(overall),
           "warning": dr.get("warning")}

    os.makedirs(records_dir, exist_ok=True)
    with open(os.path.join(records_dir, "speed_chain_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    with open(os.path.join(records_dir, "speed_chain_validity.json"), "w", encoding="utf-8") as fh:
        json.dump(val, fh, indent=1)

    cols = ["track", "backend", "precision", "compile", "median_ms", "p95_ms", "p99_ms", "FPS_median",
            "delta_vs_historical_A_ms", "delta_vs_same_window_A3_ms", "grade_vs_same_window_A3", "validity"]
    with open(os.path.join(records_dir, "speed_chain_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in main_rows:
            w.writerow(r)

    md = ["# speed chain summary", "",
          "historical reference: %.2f ms | thresholds: useful<=%.2f strong<=%.2f excellent<=%.2f"
          % (HISTORICAL_BASELINE_MS, USEFUL_MS, STRONG_MS, EXCELLENT_MS),
          "A3/A4 drift: %s" % json.dumps(dr),
          "overall_window_valid: %s" % overall, ""]
    md.append("## main table (VALID tracks only)")
    md.append("")
    md.append("| Track | Backend | Precision | Median ms | p95 | p99 | FPS | d vs A3 | Validity |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in main_rows:
        md.append("| %s | %s | %s | %.3f | %.3f | %.3f | %s | %+.3f | %s |"
                  % (r["track"], r["backend"], r["precision"], r["median_ms"], r["p95_ms"],
                     r["p99_ms"], r["FPS_median"], r["delta_vs_same_window_A3_ms"], r["validity"]))
    if not main_rows:
        md.append("| (none) | | | | | | | | |")
    md += ["", "## audit table (contaminated / rejected - NOT part of the main table)", ""]
    md.append("| Track | Backend | Precision | Median ms | Validity | Reason |")
    md.append("|---|---|---|---:|---|---|")
    for r in audit_rows:
        md.append("| %s | %s | %s | %.3f | %s | %s |"
                  % (r["track"], r["backend"], r["precision"], r["median_ms"], r["validity"],
                     validity[r["track"]]["contamination_reason"]))
    if not audit_rows:
        md.append("| (none) | | | | | |")
    with open(os.path.join(records_dir, "speed_chain_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    return man, val, main_rows, audit_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    a = ap.parse_args()
    man, val, main_rows, audit_rows = build(a.records_dir, "speed_chain")
    print(json.dumps({"manifest_inputs_found": man["inputs_found"],
                      "validity": val["each_track_validity"],
                      "A3_A4_drift": val["A3_A4_drift"],
                      "overall_window_valid": val["overall_window_valid"],
                      "main_rows": len(main_rows), "audit_rows": len(audit_rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())