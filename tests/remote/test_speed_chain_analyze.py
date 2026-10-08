#!/usr/bin/env python3
"""Tests for tools/remote/speed_chain_analyze.py - pins the pre-registered chain rules.

These run with no GPU and no network. They fix the contamination rules, the A3/A4 drift guard,
the verdict grades and the main/audit table split, so those cannot be tuned after seeing results.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "tools" / "remote" / "speed_chain_analyze.py"


def load_module():
    spec = importlib.util.spec_from_file_location("speed_chain_analyze", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load_module()


def write_monitor(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "gpu_util", "gpu_mem", "compute_apps", "load1"])
        for r in rows:
            w.writerow(r)


def rec(median_ms, gate_passed=True, **kw):
    d = {"median_ms": median_ms, "p95_ms": median_ms + 1, "p99_ms": median_ms + 2,
         "FPS_median": round(1000.0 / median_ms, 2),
         "clean_gate": {"passed": gate_passed, "streak": 3 if gate_passed else 0},
         "gpu_snapshot_start": "0 %, 38 MiB, 49140 MiB",
         "gpu_snapshot_end": "34 %, 932 MiB, 49140 MiB"}
    d.update(kw)
    return d


def test_rules_are_pre_registered():
    assert M.RULES["gate_gpu_util_max"] == 10.0
    assert M.RULES["gate_gpu_mem_max"] == 1200
    assert M.RULES["gate_load1_max"] == 4.0
    assert M.RULES["gate_compute_apps_max"] == 0
    assert M.RULES["run_mem_ceiling_MiB"] == 2500
    assert M.RULES["run_mem_sustained_samples"] == 3
    assert M.HISTORICAL_BASELINE_MS == 15.90
    assert (M.USEFUL_MS, M.STRONG_MS, M.EXCELLENT_MS) == (12.72, 10.0, 8.0)
    assert M.DRIFT_REL_MAX == 0.05


def test_own_process_is_not_contamination():
    rows = [[float(i), 40.0, 932.0, 1, 3.0] for i in range(5)]
    bad, reason = M.contamination([{"t_s": r[0], "gpu_util": r[1], "gpu_mem": r[2],
                                    "compute_apps": r[3], "load1": r[4]} for r in rows])
    assert bad is False and reason == ""


def test_foreign_compute_app_is_contamination():
    s = [{"t_s": 1.0, "gpu_util": 40.0, "gpu_mem": 932.0, "compute_apps": 1, "load1": 3.0},
         {"t_s": 2.0, "gpu_util": 88.0, "gpu_mem": 7000.0, "compute_apps": 2, "load1": 6.0}]
    bad, reason = M.contamination(s)
    assert bad is True and "foreign_compute_apps" in reason


def test_sustained_memory_is_contamination():
    s = [{"t_s": float(i), "gpu_util": 5.0, "gpu_mem": 6000.0, "compute_apps": 1, "load1": 1.0}
         for i in range(3)]
    bad, reason = M.contamination(s)
    assert bad is True and "sustained" in reason


def test_transient_memory_spike_is_not_contamination():
    s = [{"t_s": 0.0, "gpu_util": 5.0, "gpu_mem": 900.0, "compute_apps": 1, "load1": 1.0},
         {"t_s": 1.0, "gpu_util": 5.0, "gpu_mem": 6000.0, "compute_apps": 1, "load1": 1.0},
         {"t_s": 2.0, "gpu_util": 5.0, "gpu_mem": 6000.0, "compute_apps": 1, "load1": 1.0},
         {"t_s": 3.0, "gpu_util": 5.0, "gpu_mem": 900.0, "compute_apps": 1, "load1": 1.0}]
    bad, _ = M.contamination(s)
    assert bad is False


def test_missing_monitor_is_contamination():
    bad, reason = M.contamination([])
    assert bad is True and reason == "NO_MONITOR_DATA"


def test_missing_gate_confirmation_invalidates():
    v = M.judge(rec(15.9, gate_passed=False), [{"t_s": 0.0, "gpu_util": 1.0, "gpu_mem": 100.0,
                                                "compute_apps": 1, "load1": 0.5}])
    assert v["validity"] == "CONTAMINATED_INVALID"
    assert "clean_gate_not_confirmed" in v["contamination_reason"]


def test_valid_track_passes():
    v = M.judge(rec(15.9), [{"t_s": 0.0, "gpu_util": 1.0, "gpu_mem": 100.0, "compute_apps": 1, "load1": 0.5}])
    assert v["validity"] == "VALID"


def test_missing_record_is_missing():
    assert M.judge(None, [])["validity"] == "MISSING"


def test_drift_within_threshold_has_no_warning():
    d = M.drift(rec(15.90), rec(16.20))
    assert d["warning"] is None and d["drift_rel"] < 0.05


def test_drift_over_threshold_warns():
    d = M.drift(rec(15.90), rec(17.00))
    assert d["warning"] == "WINDOW_DRIFT_WARNING"
    assert round(d["drift_abs_ms"], 2) == 1.10


def test_drift_incomplete_without_a4():
    assert M.drift(rec(15.90), None)["warning"] == "A3_A4_INCOMPLETE"


def test_verdict_grades():
    assert M.verdict(12.70, 15.90)[2] == "useful"
    assert M.verdict(12.72, 15.90)[2] == "useful"
    assert M.verdict(9.99, 15.90)[2] == "strong"
    assert M.verdict(7.99, 15.90)[2] == "excellent"
    assert M.verdict(13.00, 15.90)[2] == "below_useful"


def test_build_splits_main_and_audit(tmp_path):
    names = {"A3": "speed_pytorch_fp32_A3.json", "A4": "speed_pytorch_fp32_A4.json",
             "B": "speed_pytorch_fp16_clean.json"}
    clean_mon = [[0.0, 1.0, 100.0, 1, 0.5], [1.0, 2.0, 120.0, 1, 0.6]]
    dirty_mon = [[0.0, 40.0, 6847.0, 1, 4.8], [1.0, 97.0, 7380.0, 2, 6.0]]
    for tag, fname in names.items():
        (tmp_path / fname).write_text(json.dumps(rec(15.9 if tag != "B" else 20.8)), encoding="utf-8")
    write_monitor(tmp_path / "speed_pytorch_fp32_A3.monitor.csv", clean_mon)
    write_monitor(tmp_path / "speed_pytorch_fp32_A4.monitor.csv", clean_mon)
    write_monitor(tmp_path / "speed_pytorch_fp16_clean.monitor.csv", dirty_mon)
    man, val, main_rows, audit_rows = M.build(str(tmp_path), "speed_chain")
    tags_main = {r["track"] for r in main_rows}
    tags_audit = {r["track"] for r in audit_rows}
    assert tags_main == {"A3", "A4"}
    assert tags_audit == {"B"}
    assert val["each_track_validity"]["B"] == "CONTAMINATED_INVALID"
    assert val["each_track_validity"]["C"] == "MISSING"
    assert val["overall_window_valid"] is True
    assert (tmp_path / "speed_chain_manifest.json").exists()
    assert (tmp_path / "speed_chain_validity.json").exists()
    assert (tmp_path / "speed_chain_summary.csv").exists()
    assert (tmp_path / "speed_chain_summary.md").exists()
    md = (tmp_path / "speed_chain_summary.md").read_text(encoding="utf-8")
    assert "main table (VALID tracks only)" in md
    assert "audit table" in md


def test_summary_csv_excludes_contaminated(tmp_path):
    (tmp_path / "speed_pytorch_fp32_A3.json").write_text(json.dumps(rec(15.9)), encoding="utf-8")
    (tmp_path / "speed_pytorch_fp32_A4.json").write_text(json.dumps(rec(16.1)), encoding="utf-8")
    (tmp_path / "speed_pytorch_fp16_clean.json").write_text(json.dumps(rec(20.8)), encoding="utf-8")
    clean = [[0.0, 1.0, 100.0, 1, 0.5]]
    dirty = [[0.0, 97.0, 7380.0, 2, 6.0]]
    write_monitor(tmp_path / "speed_pytorch_fp32_A3.monitor.csv", clean)
    write_monitor(tmp_path / "speed_pytorch_fp32_A4.monitor.csv", clean)
    write_monitor(tmp_path / "speed_pytorch_fp16_clean.monitor.csv", dirty)
    M.build(str(tmp_path), "speed_chain")
    rows = list(csv.DictReader(open(tmp_path / "speed_chain_summary.csv", encoding="utf-8")))
    assert {r["track"] for r in rows} == {"A3", "A4"}


def test_compile_diagnostics_are_carried_through(tmp_path):
    (tmp_path / "speed_pytorch_fp32_A3.json").write_text(json.dumps(rec(15.9)), encoding="utf-8")
    (tmp_path / "speed_pytorch_fp32_A4.json").write_text(json.dumps(rec(15.9)), encoding="utf-8")
    (tmp_path / "speed_torch_compile_fp16_clean.json").write_text(
        json.dumps(rec(18.1, compile_time_s=92.5, graph_break_count=0, symbolic_shape_warning=True)),
        encoding="utf-8")
    clean = [[0.0, 1.0, 100.0, 1, 0.5]]
    for n in ("speed_pytorch_fp32_A3.monitor.csv", "speed_pytorch_fp32_A4.monitor.csv",
              "speed_torch_compile_fp16_clean.monitor.csv"):
        write_monitor(tmp_path / n, clean)
    _man, _val, main_rows, _audit = M.build(str(tmp_path), "speed_chain")
    c = [r for r in main_rows if r["track"] == "C"][0]
    assert c["compile_time_s"] == 92.5
    assert c["graph_break_count"] == 0
    assert c["symbolic_shape_warning"] is True

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
