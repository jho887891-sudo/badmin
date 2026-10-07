#!/usr/bin/env python3
"""Pre-run gates 1-3 of YOLO26S_V2_FULL_ETH_CONTINUATION_V1: frozen checkpoint identity + frozen architecture.

Spec refs: section 3 (frozen starting checkpoint), section 4 (frozen architecture), section 27 (environment).
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
        print("STOP: checkpoint not found at %s" % w)
        return 2
    info["bytes"] = w.stat().st_size
    info["sha256"] = sha256(w)
    info["expected_bytes"] = EXPECT_BYTES
    info["expected_sha256"] = EXPECT_SHA
    info["bytes_match"] = info["bytes"] == EXPECT_BYTES
    info["sha256_match"] = info["sha256"] == EXPECT_SHA

    import torch
    import ultralytics
    info["env"] = {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
                   "torch": torch.__version__, "cuda_build": torch.version.cuda,
                   "cuda_available": bool(torch.cuda.is_available()),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "gpu_total_mib": (torch.cuda.get_device_properties(0).total_memory / (1 << 20)
                                     if torch.cuda.is_available() else None),
                   "ultralytics": ultralytics.__version__}

    from ultralytics import YOLO
    m = YOLO(str(w))
    core = m.model
    stride = [int(s) for s in core.stride.tolist()] if hasattr(core.stride, "tolist") else list(core.stride)
    y = core.yaml if isinstance(core.yaml, dict) else {}
    head_names = []
    try:
        head_names = [type(l).__name__ for l in core.model]
    except Exception:
        pass
    info["arch"] = {"params": int(sum(p.numel() for p in core.parameters())),
                    "stride": stride,
                    "yaml_scale": y.get("scale"),
                    "yaml_keys": sorted(y.keys()),
                    "yaml_nc": y.get("nc"),
                    "yaml_backbone_len": len(y.get("backbone", []) or []),
                    "yaml_head_len": len(y.get("head", []) or []),
                    "layer_names": head_names,
                    "layer_names_with_p2": [n for n in head_names if "p2" in n.lower()],
                    "yaml_mentions_p2": [str(x) for x in (y.get("head", []) or []) if "P2" in str(x)]}
    info["plain_yolo26s"] = stride == [8, 16, 32]
    info["has_p2"] = any(s <= 4 for s in stride)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    print("weights      : %s" % w)
    print("bytes        : %d (match=%s)" % (info["bytes"], info["bytes_match"]))
    print("sha256       : %s (match=%s)" % (info["sha256"], info["sha256_match"]))
    print("env          : %s" % json.dumps(info["env"]))
    print("params       : %d" % info["arch"]["params"])
    print("stride       : %s  plain_yolo26s=%s  has_p2=%s" % (stride, info["plain_yolo26s"], info["has_p2"]))
    print("yaml         : scale=%s nc=%s backbone_len=%s head_len=%s" % (info["arch"]["yaml_scale"], info["arch"]["yaml_nc"],
          info["arch"]["yaml_backbone_len"], info["arch"]["yaml_head_len"]))
    print("layer names  : %s" % ",".join(head_names))
    print("WROTE %s" % a.out)
    if not (info["bytes_match"] and info["sha256_match"]):
        print("STOP: checkpoint identity mismatch - do not substitute another checkpoint")
        return 3
    if not info["plain_yolo26s"] or info["has_p2"]:
        print("STOP: architecture is not plain YOLO26s / a P2 stride-4 layer is present")
        return 4
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
