#!/usr/bin/env python3
"""speed_harness.py - ONE unified latency harness for YOLO26S_INFERENCE_SPEED_OPT_V1 (v2).

Protocol (spec 7/8): model-only inference latency, warmup 50, measured 500, cuda sync around the
timed region, median/mean/std/p90/p95/p99 - never fastest-only.

v2 adds:
  * 1 s background GPU sampling into --monitor-csv (gpu_util, gpu_mem, compute_apps, load1)
  * --gate-passed / --gate-streak taken from the chain clean-window gate, stored in the record
  * Track C compile diagnostics (compile_time_s, graph_break_count, symbolic_shape_warning, warning_text)
  * TensorRT backend (raw runtime API, model-only) for Tracks D/E

Never modifies ultralytics source. No graph surgery. No GPU clock changes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_monitor as MON  # noqa: E402

CKPT_SHA = "7a836a2621686affbdd3f1da7c3a0a57c4b86f14432835be2bc562c2899fc6f2"
CKPT_BYTES = 20344069


def install_tensorrt_alias():
    """TRT >=10 ships the API as tensorrt_bindings; ultralytics calls trt.Logger / trt.__version__."""
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
    return trt


def stats(ts):
    a = np.asarray(ts, dtype=float)
    return {"n": int(a.size), "median_ms": float(np.median(a)), "mean_ms": float(a.mean()),
            "std_ms": float(a.std()), "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "min_ms": float(a.min()), "max_ms": float(a.max())}


class Monitor(threading.Thread):
    daemon = True

    def __init__(self, path, interval):
        super().__init__()
        self.interval = float(interval)
        self.stop_flag = threading.Event()
        self.fh, self.writer = MON.open_monitor(path)
        self.t0 = time.perf_counter()
        self.n = 0

    def run(self):
        while not self.stop_flag.is_set():
            gpu = MON.sample_gpu()
            self.writer.writerow(MON.monitor_row(time.perf_counter() - self.t0, gpu, MON.sample_load1()))
            self.fh.flush()
            self.n += 1
            self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)
        self.fh.close()
        return self.n


def bench_pytorch(a):
    m = YOLO(a.ckpt)
    net = m.model.eval().to("cuda")
    if a.precision == "fp16":
        net = net.half()
    x = torch.randn(a.batch, 3, a.imgsz, a.imgsz, device="cuda")
    if a.precision == "fp16":
        x = x.half()
    extra = {}
    if a.compile == "yes":
        t0 = time.perf_counter()
        net = torch.compile(net)
        with torch.no_grad():
            net(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        counters = None
        try:
            from torch._dynamo.utils import counters as _c
            counters = int(sum(_c.get("graph_break", {}).values())) or 0
        except Exception:
            counters = None
        extra = MON.compile_diagnostics(recompile_count=counters, elapsed_s=elapsed,
                                         warnings_text=a.compile_warnings or "")
    with torch.no_grad():
        for _ in range(a.warmup):
            net(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return stats(ts), extra, "model_only (GPU forward + detection head)"


def load_engine_plan(path):
    """Return (plan_bytes, metadata).

    ultralytics writes a 4-byte little-endian JSON metadata length, then the JSON, then the serialized
    plan. Handing the whole file to deserialize_cuda_engine reads that length as the plan magic. Measured
    failure 2026-10-06: "incompatible serialization version (582 != 1953657958)" - 582 was the JSON length.
    """
    return MON.load_engine_plan(path)


def bench_tensorrt(a):
    trt = install_tensorrt_alias()
    logger = trt.Logger(trt.Logger.ERROR)
    plan, meta = load_engine_plan(a.engine)
    with trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(plan)
    if engine is None:
        raise SystemExit("could not deserialize engine " + a.engine)
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    out_name = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
    declared = tuple(int(d) for d in engine.get_tensor_shape(in_name))
    real = MON.concrete_input_shape(declared, a.batch, a.imgsz)
    # The shape must be concrete BEFORE the tensor exists: a dynamic engine declares -1 dims, and
    # deriving the tensor shape from them raised "negative dimension -1" on 2026-10-07 (Track E).
    if any(d <= 0 for d in declared):
        if not ctx.set_input_shape(in_name, real):
            raise SystemExit("set_input_shape failed: %s %s" % (in_name, list(real)))
    x = torch.randn(*real, device="cuda").to(torch.float16 if a.precision == "fp16" else torch.float32)
    ctx.set_tensor_address(in_name, int(x.data_ptr()))
    osize = tuple(ctx.get_tensor_shape(out_name))
    out = torch.empty(osize, device="cuda", dtype=torch.float32)
    ctx.set_tensor_address(out_name, int(out.data_ptr()))
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(a.warmup):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return stats(ts), {"engine_metadata": meta}, "model_only (TensorRT runtime execute_async_v3, no python postprocess)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "tensorrt"])
    ap.add_argument("--ckpt", default="eth_only_v1_best.pt")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--compile", default="no", choices=["no", "yes"])
    ap.add_argument("--compile-warnings", default="")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--monitor-csv", default=None)
    ap.add_argument("--monitor-interval", type=float, default=1.0)
    ap.add_argument("--gate-passed", default="no", choices=["yes", "no"])
    ap.add_argument("--gate-streak", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.backend == "tensorrt":
        install_tensorrt_alias()
    # Sample the pre-run state BEFORE the benchmark. v2 sampled these after the run, so a record could
    # claim "gpu_snapshot_start: 27 %" while that 27 % was measured at the end. The 1 s monitor CSV was
    # always correct and is what the analyzer uses; these two fields are now honest as well.
    pre_row = MON.monitor_row(0.0, MON.sample_gpu(), MON.sample_load1())
    pre_gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    mon = Monitor(a.monitor_csv, a.monitor_interval) if a.monitor_csv else None
    if mon:
        mon.start()
    try:
        if a.backend == "pytorch":
            st, extra, level = bench_pytorch(a)
        else:
            st, extra, level = bench_tensorrt(a)
    finally:
        n = mon.stop() if mon else 0

    rec = {"track": a.track, "backend": a.backend, "precision": a.precision, "compile": a.compile,
           "imgsz": a.imgsz, "batch": a.batch, "latency_level": level,
           "warmup_iterations": a.warmup, "timed_iterations": a.iters,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "checkpoint": a.ckpt, "checkpoint_sha256": CKPT_SHA, "checkpoint_bytes": CKPT_BYTES,
           "engine": a.engine, "git_commit": "NOT_A_GIT_TREE",
           "PyTorch": torch.__version__, "CUDA_build": torch.version.cuda,
           "Ultralytics": __import__("ultralytics").__version__,
           "TensorRT": (__import__("tensorrt_bindings").__version__ if a.backend == "tensorrt" else None),
           "python": platform.python_version(),
           "clean_gate": {"passed": a.gate_passed == "yes", "streak": a.gate_streak,
                          "requirements": {"gate_gpu_util_max": 10.0, "gate_gpu_mem_max": 1200,
                                           "gate_load1_max": 4.0, "gate_compute_apps_max": 0},
                          "gpu_at_start": pre_row},
           "monitor_samples": n, "monitor_csv": a.monitor_csv,
           "gpu_snapshot_start": pre_gpu,
           "VRAM_peak_MB": round(torch.cuda.max_memory_allocated() / 1048576.0, 1)}
    rec.update(st)
    rec.update(extra)
    rec["FPS_median"] = round(1000.0 / rec["median_ms"], 2)
    rec["gpu_snapshot_end"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec.get(k) for k in ("track", "backend", "precision", "compile", "median_ms",
                                              "p95_ms", "p99_ms", "FPS_median", "latency_level",
                                              "graph_break_count", "symbolic_shape_warning",
                                              "gpu_snapshot_start", "gpu_snapshot_end")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())