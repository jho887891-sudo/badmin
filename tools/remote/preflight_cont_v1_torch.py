#!/usr/bin/env python3
"""Gates 1-3 for YOLO26S_V2_FULL_ETH_CONTINUATION_V1, deliberately torch-only.

The ultralytics runtime lives on a fuseblk (NTFS) mount whose reads currently stall, so this variant reads the
checkpoint with plain torch from the local xfs disk: no ultralytics import, no /home/T7 dependency.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

EXPECT_BYTES = 20344133
EXPECT_SHA = "3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7"


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = Path(a.weights)
    info = {"experiment": "YOLO26S_V2_FULL_ETH_CONTINUATION_V1", "weights": str(w), "exists": w.is_file()}
    if not w.is_file():
        print("STOP: checkpoint not found")
        return 2
    info["bytes"] = w.stat().st_size
    info["sha256"] = sha256(w)
    info["expected_bytes"] = EXPECT_BYTES
    info["expected_sha256"] = EXPECT_SHA
    info["bytes_match"] = info["bytes"] == EXPECT_BYTES
    info["sha256_match"] = info["sha256"] == EXPECT_SHA

    import torch
    info["env"] = {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
                   "torch": torch.__version__, "cuda_build": torch.version.cuda,
                   "cuda_available": bool(torch.cuda.is_available()),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "gpu_total_mib": int(torch.cuda.get_device_properties(0).total_memory / (1 << 20))
                   if torch.cuda.is_available() else None,
                   "ultralytics": "not imported in this variant (see environment json / dist-info)"}

    ck = torch.load(str(w), map_location="cpu", weights_only=False)
    info["checkpoint_keys"] = sorted(ck.keys()) if isinstance(ck, dict) else None
    core = ck["model"] if isinstance(ck, dict) and "model" in ck else None
    if core is None:
        print("STOP: checkpoint has no 'model' entry")
        return 5
    y = core.yaml if isinstance(core.yaml, dict) else {}
    stride = [int(s) for s in core.stride.tolist()] if hasattr(core.stride, "tolist") else list(core.stride)
    names = [type(m).__name__ for m in core.model]
    info["arch"] = {"params": int(sum(p.numel() for p in core.parameters())),
                    "stride": stride, "yaml_scale": y.get("scale"), "yaml_nc": y.get("nc"),
                    "yaml_keys": sorted(y.keys()),
                    "yaml_backbone_len": len(y.get("backbone", []) or []),
                    "yaml_head_len": len(y.get("head", []) or []),
                    "layer_names": names,
                    "layer_names_with_p2": [n for n in names if "p2" in n.lower()],
                    "yaml_head_mentions_p2": [str(x) for x in (y.get("head", []) or []) if "P2" in str(x)],
                    "nc": int(core.nc) if hasattr(core, "nc") else None,
                    "names_sample": ({k: core.names[k] for k in list(core.names)[:5]}
                                     if hasattr(core, "names") and core.names else None)}
    if isinstance(ck, dict) and isinstance(ck.get("train_args"), dict):
        ta = {k: ck["train_args"].get(k) for k in ("model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0",
                                                   "nbs", "seed", "pretrained", "freeze", "rect", "cos_lr")
              if k in ck["train_args"]}
        info["train_args_of_source_run"] = ta
    info["plain_yolo26s"] = stride == [8, 16, 32]
    info["has_p2"] = any(s <= 4 for s in stride)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    print("weights   : %s" % w)
    print("bytes     : %d match=%s" % (info["bytes"], info["bytes_match"]))
    print("sha256    : %s match=%s" % (info["sha256"], info["sha256_match"]))
    print("env       : %s" % json.dumps(info["env"]))
    print("params    : %d" % info["arch"]["params"])
    print("stride    : %s plain_yolo26s=%s has_p2=%s" % (stride, info["plain_yolo26s"], info["has_p2"]))
    print("yaml      : scale=%s nc=%s backbone_len=%s head_len=%s" % (info["arch"]["yaml_scale"], info["arch"]["yaml_nc"],
          info["arch"]["yaml_backbone_len"], info["arch"]["yaml_head_len"]))
    print("layers    : %s" % ",".join(names))
    print("p2 hits   : %s" % info["arch"]["layer_names_with_p2"])
    print("names     : %s" % info["arch"]["names_sample"])
    print("src args  : %s" % json.dumps(info.get("train_args_of_source_run")))
    print("WROTE %s" % a.out)
    if not (info["bytes_match"] and info["sha256_match"]):
        print("STOP: checkpoint identity mismatch")
        return 3
    if not info["plain_yolo26s"] or info["has_p2"]:
        print("STOP: not plain YOLO26s / P2 present")
        return 4
    print("PREFLIGHT PASS (torch-only variant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
