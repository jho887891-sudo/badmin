#!/usr/bin/env python3
"""speed_monitor.py - pure helpers for the speed chain: 1 s GPU sampling + compile diagnostics.

Kept free of torch/numpy so it can be unit-tested anywhere. The harness imports it at run time.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess

MONITOR_COLUMNS = ["t_s", "gpu_util", "gpu_mem", "compute_apps", "load1"]


def _num(text, default=None):
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError):
        return default


def sample_gpu(run=subprocess.run):
    """One read-only nvidia-smi sample. Never raises: a missing field becomes a safe default."""
    out = {"gpu_util": None, "gpu_mem": None, "compute_apps": 0}
    try:
        txt = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if txt:
            parts = [p.strip() for p in txt[0].split(",")]
            out["gpu_util"] = _num(parts[0])
            out["gpu_mem"] = _num(parts[1])
    except Exception:
        pass
    try:
        apps = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=10).stdout.strip()
        out["compute_apps"] = len([a for a in apps.splitlines() if a.strip()])
    except Exception:
        pass
    return out


def sample_load1(path="/proc/loadavg"):
    try:
        with open(path, encoding="utf-8") as fh:
            return _num(fh.read().split()[0])
    except Exception:
        return None


def monitor_row(t_s, gpu, load1):
    return {"t_s": round(float(t_s), 3), "gpu_util": gpu.get("gpu_util"),
            "gpu_mem": gpu.get("gpu_mem"), "compute_apps": gpu.get("compute_apps"), "load1": load1}


def open_monitor(path):
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=MONITOR_COLUMNS)
    w.writeheader()
    return fh, w


def gate_ok(gpu, load1, rules):
    """Pre-registered clean-window predicate. compute_apps must be exactly 0 (spec)."""
    if gpu.get("gpu_util") is None or gpu.get("gpu_mem") is None or load1 is None:
        return False
    return (gpu["gpu_util"] <= rules["gate_gpu_util_max"]
            and gpu["gpu_mem"] <= rules["gate_gpu_mem_max"]
            and load1 <= rules["gate_load1_max"]
            and int(gpu.get("compute_apps") or 0) == rules["gate_compute_apps_max"])


SYMBOLIC_WARNING_PATTERNS = ("pow_by_natural", "symbolic", "dynamic shape")


def classify_compile_warnings(text):
    """Return (symbolic_shape_warning, warning_text, matched_lines).

    A warning is NOT evidence of a graph break - graph_break_count must be read from the dynamo
    counters separately. This function only classifies warning text.
    """
    lines = [ln for ln in (text or "").splitlines()
             if any(p in ln.lower() for p in SYMBOLIC_WARNING_PATTERNS)]
    return (bool(lines), ("\n".join(lines) if lines else ""), lines)


def compile_diagnostics(explain_out="", recompile_count=None, elapsed_s=None, warnings_text=""):
    """Assemble the spec section 6 Track C record fields from raw dynamo output."""
    sym, wtext, _lines = classify_compile_warnings(warnings_text or explain_out)
    return {"compile_time_s": (round(float(elapsed_s), 2) if elapsed_s is not None else None),
            "graph_break_count": (int(recompile_count) if recompile_count is not None else None),
            "graph_break_source": ("torch._dynamo.utils.counters" if recompile_count is not None else "NOT_CAPTURED"),
            "symbolic_shape_warning": bool(sym),
            "warning_text": wtext}


def concrete_input_shape(declared, batch, imgsz):
    """Resolve a possibly-dynamic TensorRT input shape to a concrete one.

    A dynamic engine declares dims as -1, and building a torch tensor from that raises
    "Trying to create tensor with negative dimension -1: [1, 3, -1, -1]" (measured 2026-10-07).
    Positive declared dims are preserved; dynamic ones become batch (position 0) or imgsz.
    """
    dims = [int(d) for d in declared]
    if not dims:
        return (batch, 3, imgsz, imgsz)
    out = []
    for i, d in enumerate(dims):
        if d > 0:
            out.append(d)
        else:
            out.append(int(batch) if i == 0 else int(imgsz))
    return tuple(out)


def load_engine_plan(path):
    """Return (plan_bytes, metadata) from an ultralytics-written TensorRT .engine file.

    ultralytics writes a 4-byte little-endian JSON metadata length, then the JSON, then the serialized
    plan. Handing the whole file to deserialize_cuda_engine reads that length as the plan magic.
    Measured failure 2026-10-06: "incompatible serialization version (582 != 1953657958)" - 582 was
    exactly the JSON length, which is what made this diagnosable.
    """
    import json
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) > 4:
        try:
            n = int.from_bytes(data[:4], "little")
            if 0 < n < len(data) - 4:
                return data[4 + n:], json.loads(data[4:4 + n].decode("utf-8"))
        except Exception:
            pass
    return data, None