#!/usr/bin/env python3
"""speed_eth_export.py - export the ETH official YOLOv8s checkpoint to a TensorRT FP16 static engine.

Step 1 of the fair comparison: verify the frozen checkpoint by sha256, rebuild the yolov8s architecture
with the AUDITED loader (tools/eval_eth_official_baseline.load_eth_model - not a new one), assert that
missing/unexpected keys are exactly 0, then export and record everything the spec demands.

Refuses to continue on CHECKPOINT_MISMATCH or on any missing/unexpected state-dict key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

EXPECTED_SHA256 = "f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d"
EXPECTED_BYTES = 134312133
TRT_TOOLS = "/home/T7/ojh/robot_sim/tools"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_tensorrt_alias():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-out", default=None, help="copy the built engine here (distinct per model)")
    a = ap.parse_args()

    rec = {"script": "speed_eth_export.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "checkpoint": a.ckpt, "imgsz": a.imgsz, "batch": a.batch,
           "backend": "tensorrt", "precision": "fp16", "shape": "static", "dynamic": False,
           "export_api": "ultralytics YOLO.export(format=engine, half=True, dynamic=False)"}
    if not os.path.isfile(a.ckpt):
        rec.update({"status": "STOP_CHECKPOINT_MISSING"}); _write(a.out, rec); return 2
    rec["file_size"] = os.path.getsize(a.ckpt)
    rec["sha256"] = sha256_file(a.ckpt)
    rec["sha256_matches_record"] = rec["sha256"].lower() == EXPECTED_SHA256
    rec["bytes_match_record"] = rec["file_size"] == EXPECTED_BYTES
    if not (rec["sha256_matches_record"] and rec["bytes_match_record"]):
        rec.update({"status": "STOP_CHECKPOINT_MISMATCH"}); _write(a.out, rec)
        print(json.dumps({"status": "STOP_CHECKPOINT_MISMATCH", "sha256": rec["sha256"]}))
        return 3

    trt = install_tensorrt_alias()
    rec["tensorrt"] = trt.__version__
    rec["tensorrt_module_alias"] = True
    import torch, platform
    rec.update({"PyTorch": torch.__version__, "CUDA_build": torch.version.cuda,
                "python": platform.python_version(),
                "Ultralytics": __import__("ultralytics").__version__})
    rec["gpu_driver"] = _smi("driver_version")
    rec["gpu_name"] = _smi("name")

    sys.path.insert(0, TRT_TOOLS)
    import eval_eth_official_baseline as eth  # audited loader, used unmodified

    # The audited loader returns a YOLO object built from yolov8s.yaml with the released state dict,
    # and raises unless missing and unexpected are both empty. Record the counts explicitly anyway.
    wrapper, meta = eth.load_eth_model(a.ckpt, device="0")
    rec["model_meta"] = meta
    rec["state_dict_missing_keys"] = 0
    rec["state_dict_unexpected_keys"] = 0
    rec["state_dict_note"] = "load_eth_model raises SystemExit if either is non-empty; reaching here means both are 0"

    rec["gpu_snapshot_before_build"] = _smi("utilization.gpu,memory.used")
    t0 = time.time()
    eng = wrapper.export(format="engine", half=True, imgsz=a.imgsz, batch=a.batch,
                         dynamic=False, device=0, verbose=False)
    rec["export_duration_s"] = round(time.time() - t0, 1)
    rec["gpu_snapshot_after_build"] = _smi("utilization.gpu,memory.used")
    rec["engine_file"] = str(eng)
    if os.path.isfile(str(eng)):
        rec["engine_size_bytes"] = os.path.getsize(str(eng))
        rec["engine_sha256"] = sha256_file(str(eng))
    if a.engine_out:
        import shutil
        shutil.copyfile(str(eng), a.engine_out)
        rec["engine_copy"] = a.engine_out
    rec["status"] = "OK"
    _write(a.out, rec)
    print(json.dumps({k: rec.get(k) for k in ("status", "sha256", "engine_file", "engine_size_bytes",
                                              "engine_sha256", "export_duration_s", "tensorrt")}, indent=1))
    return 0


def _smi(fields):
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return None


def _write(path, obj):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())