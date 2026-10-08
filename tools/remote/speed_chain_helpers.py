#!/usr/bin/env python3
"""speed_chain_helpers.py - clean-window gate + TensorRT engine export for the speed chain."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_monitor as MON  # noqa: E402


def _write_json(path, obj):
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)

GATE_REQUIREMENTS = {"gate_gpu_util_max": 10.0, "gate_gpu_mem_max": 1200,
                     "gate_load1_max": 4.0, "gate_compute_apps_max": 0}


def cmd_gate(a):
    """Wait for N consecutive clean polls. Writes a JSON record for the chain."""
    streak = 0
    samples = []
    t0 = time.time()
    while (time.time() - t0) < a.max_wait:
        gpu = MON.sample_gpu()
        load1 = MON.sample_load1()
        ok = MON.gate_ok(gpu, load1, GATE_REQUIREMENTS)
        samples.append({"t_s": round(time.time() - t0, 1), "ok": ok, "gpu": gpu, "load1": load1})
        print("%s gpu_util=%s gpu_mem=%s apps=%s load1=%s -> %s"
              % (time.strftime("%H:%M:%S"), gpu["gpu_util"], gpu["gpu_mem"],
                 gpu["compute_apps"], load1, "CLEAN" if ok else "BUSY"), flush=True)
        streak = streak + 1 if ok else 0
        if streak >= a.need:
            out = {"passed": True, "streak": streak, "requirements": GATE_REQUIREMENTS,
                   "samples_taken": len(samples), "waited_s": round(time.time() - t0, 1),
                   "first_clean_sample": samples[-1]}
            _write_json(a.out, out)
            print(json.dumps({"gate": "PASSED", "streak": streak, "waited_s": out["waited_s"]}))
            return 0
        time.sleep(a.poll)
    out = {"passed": False, "streak": streak, "requirements": GATE_REQUIREMENTS,
           "samples_taken": len(samples), "waited_s": round(time.time() - t0, 1),
           "reason": "TIMEOUT"}
    _write_json(a.out, out)
    print(json.dumps({"gate": "TIMEOUT", "streak": streak}))
    return 1


def cmd_export(a):
    """Export an ONNX + TensorRT FP16 engine with the tensorrt module alias installed."""
    import tensorrt_bindings as tb
    try:
        import tensorrt as trt
    except Exception:
        trt = tb
    else:
        for name in dir(tb):
            if not name.startswith("_") and not hasattr(trt, name):
                setattr(trt, name, getattr(tb, name))
        trt.__version__ = tb.__version__
    sys.modules["tensorrt"] = trt
    print("alias ok: tensorrt -> %s (Logger=%s)" % (trt.__version__, hasattr(trt, "Logger")), flush=True)
    from ultralytics import YOLO
    m = YOLO(a.ckpt)
    t0 = time.time()
    eng = m.export(format="engine", half=(a.precision == "fp16"), imgsz=a.imgsz,
                   batch=a.batch, dynamic=(a.dynamic == "yes"), device=0, verbose=False)
    rec = {"engine": str(eng), "dynamic": a.dynamic, "batch": a.batch, "imgsz": a.imgsz,
           "precision": a.precision, "export_s": round(time.time() - t0, 1),
           "tensorrt_module_alias": True, "tensorrt": trt.__version__}
    _write_json(a.out, rec)
    print(json.dumps(rec))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--need", type=int, default=3)
    g.add_argument("--poll", type=float, default=60.0)
    g.add_argument("--max-wait", type=float, default=7200.0)
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_gate)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", default="eth_only_v1_best.pt")
    e.add_argument("--imgsz", type=int, default=1024)
    e.add_argument("--batch", type=int, default=1)
    e.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    e.add_argument("--dynamic", default="no", choices=["no", "yes"])
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())