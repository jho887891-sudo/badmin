#!/usr/bin/env python3
"""Tests for tools/remote/speed_monitor.py - pure helpers, no GPU required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "tools" / "remote" / "speed_monitor.py"


def load():
    spec = importlib.util.spec_from_file_location("speed_monitor", SPEC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load()
RULES = {"gate_gpu_util_max": 10.0, "gate_gpu_mem_max": 1200, "gate_load1_max": 4.0,
         "gate_compute_apps_max": 0}


class FakeRun:
    def __init__(self, gpu="12, 340", apps=""):
        self.gpu, self.apps = gpu, apps

    def __call__(self, cmd, **kw):
        class R:
            pass
        r = R()
        r.stdout = self.gpu if "memory.used" in " ".join(cmd) else self.apps
        return r


def test_sample_gpu_parses_values():
    s = M.sample_gpu(run=FakeRun(gpu="12 %, 340 MiB", apps=""))
    assert s["gpu_util"] == 12.0 and s["gpu_mem"] == 340.0 and s["compute_apps"] == 0


def test_sample_gpu_counts_compute_apps():
    s = M.sample_gpu(run=FakeRun(gpu="0 %, 35 MiB", apps="286773\n358317\n"))
    assert s["compute_apps"] == 2


def test_sample_gpu_never_raises():
    def boom(*a, **k):
        raise RuntimeError("nvidia-smi missing")
    s = M.sample_gpu(run=boom)
    assert s["gpu_util"] is None and s["compute_apps"] == 0


def test_gate_requires_zero_compute_apps():
    assert M.gate_ok({"gpu_util": 1.0, "gpu_mem": 100.0, "compute_apps": 0}, 0.5, RULES) is True
    assert M.gate_ok({"gpu_util": 1.0, "gpu_mem": 100.0, "compute_apps": 1}, 0.5, RULES) is False


def test_gate_thresholds():
    assert M.gate_ok({"gpu_util": 11.0, "gpu_mem": 100.0, "compute_apps": 0}, 0.5, RULES) is False
    assert M.gate_ok({"gpu_util": 1.0, "gpu_mem": 1300.0, "compute_apps": 0}, 0.5, RULES) is False
    assert M.gate_ok({"gpu_util": 1.0, "gpu_mem": 100.0, "compute_apps": 0}, 4.5, RULES) is False
    assert M.gate_ok({"gpu_util": None, "gpu_mem": 100.0, "compute_apps": 0}, 0.5, RULES) is False


def test_symbolic_warning_detected():
    txt = "W1005 torch/utils/_sympy/interp.py:179] failed while executing pow_by_natural([VR[1, int_oo]])"
    sym, wtext, lines = M.classify_compile_warnings(txt)
    assert sym is True and "pow_by_natural" in wtext and len(lines) == 1


def test_clean_compile_output_has_no_warning():
    sym, wtext, lines = M.classify_compile_warnings("compiled ok, 0 graph breaks")
    assert sym is False and wtext == "" and lines == []


def test_warning_does_not_imply_graph_break():
    d = M.compile_diagnostics(warnings_text="failed while executing pow_by_natural",
                             recompile_count=None, elapsed_s=90.0)
    assert d["symbolic_shape_warning"] is True
    assert d["graph_break_count"] is None
    assert d["graph_break_source"] == "NOT_CAPTURED"


def test_graph_break_from_counters():
    d = M.compile_diagnostics(recompile_count=3, elapsed_s=88.25)
    assert d["graph_break_count"] == 3 and d["compile_time_s"] == 88.25
    assert d["graph_break_source"] == "torch._dynamo.utils.counters"


def test_concrete_shape_static_engine_is_unchanged():
    assert M.concrete_input_shape((1, 3, 1024, 1024), 1, 1024) == (1, 3, 1024, 1024)


def test_concrete_shape_dynamic_engine_resolves_minus_ones():
    # measured failure: "negative dimension -1: [1, 3, -1, -1]"
    assert M.concrete_input_shape((-1, 3, -1, -1), 1, 1024) == (1, 3, 1024, 1024)
    assert M.concrete_input_shape((-1, 3, -1, -1), 8, 1024) == (8, 3, 1024, 1024)


def test_concrete_shape_partial_dynamic():
    assert M.concrete_input_shape((4, 3, -1, -1), 1, 640) == (4, 3, 640, 640)


def test_concrete_shape_empty_falls_back():
    assert M.concrete_input_shape((), 1, 1024) == (1, 3, 1024, 1024)


def test_load_engine_plan_skips_metadata_prefix(tmp_path):
    meta = {"names": {"0": "shuttlecock"}, "imgsz": 1024}
    blob = json.dumps(meta).encode("utf-8")
    plan = b"\x00BINARYPLAN\xff\xfe"
    p = tmp_path / "x.engine"
    p.write_bytes(len(blob).to_bytes(4, "little") + blob + plan)
    got_plan, got_meta = M.load_engine_plan(str(p))
    assert got_plan == plan
    assert got_meta == meta


def test_load_engine_plan_raw_fallback(tmp_path):
    raw = b"\x99\x88\x77\x66not-json-at-all"
    p = tmp_path / "raw.engine"
    p.write_bytes(raw)
    got_plan, got_meta = M.load_engine_plan(str(p))
    assert got_plan == raw and got_meta is None


def test_load_engine_plan_handles_tiny_file(tmp_path):
    p = tmp_path / "tiny.engine"
    p.write_bytes(b"\x01\x02")
    got_plan, got_meta = M.load_engine_plan(str(p))
    assert got_plan == b"\x01\x02" and got_meta is None


def test_monitor_csv_roundtrip(tmp_path):
    p = tmp_path / "m.csv"
    fh, w = M.open_monitor(str(p))
    w.writerow(M.monitor_row(0.0, {"gpu_util": 1.0, "gpu_mem": 100.0, "compute_apps": 1}, 0.4))
    fh.close()
    body = p.read_text(encoding="utf-8").splitlines()
    assert body[0] == "t_s,gpu_util,gpu_mem,compute_apps,load1"
    assert body[1].startswith("0.0,1.0,100.0,1,0.4")

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
