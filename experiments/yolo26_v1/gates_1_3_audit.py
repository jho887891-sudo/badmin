#!/usr/bin/env python3
"""Gates 1-3 of the YOLO26 V1 spec: model identity, official P2 config, weight transfer.

Emits:
  outputs/shuttle_capability/metrics/yolo26_v1_gate1_model_audit.json
  outputs/shuttle_capability/metrics/yolo26_v1_gate2_p2_config.json
  outputs/shuttle_capability/metrics/yolo26_p2_weight_transfer.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "env_isaaclab").is_dir() and (c / "src").is_dir():
            return c
    raise SystemExit("repo root not found")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_stats(model, imgsz: int) -> dict:
    from ultralytics.utils.torch_utils import get_flops
    detect = model.model[-1]
    strides = []
    try:
        strides = [float(x) for x in detect.stride.reshape(-1).tolist()]
    except Exception:
        try:
            strides = [float(x) for x in detect.stride.tolist()]
        except Exception:
            strides = []
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "params_total": n_params,
        "params_trainable": n_train,
        "detect_levels": int(getattr(detect, "nl", len(strides))),
        "detect_strides_px": strides,
        "nc": int(getattr(detect, "nc", -1)),
        "gflops": float(get_flops(model, imgsz)),
    }
    try:
        info["head_attributes"] = [k for k in ("one2many", "one2one", "end2end", "export") if hasattr(detect, k)]
    except Exception:
        pass
    return info


def group_of(key: str) -> str:
    m = re.match(r"model\.(\d+)\.", key)
    return "layer_" + m.group(1) if m else "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--scale", default="s")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    metrics = repo / "outputs" / "shuttle_capability" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo26v1_cfg")

    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else (
        repo / "assets" / "external" / "_staging" / "F_yolo" / "weights" / ("yolo26%s.pt" % args.scale))
    cfgdir = Path(ultralytics.__file__).parent / "cfg" / "models" / "26"

    # ---------------- Gate 1: model identity + mechanism audit ----------------
    gate1 = {
        "gate": 1,
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_file": str(ultralytics.__file__),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "python": sys.version.split()[0],
        "yolo26_cfg_dir": str(cfgdir),
        "yolo26_cfg_files": sorted(p.name for p in cfgdir.glob("*.yaml")) if cfgdir.is_dir() else [],
        "official_p2_config_present": (cfgdir / "yolo26-p2.yaml").is_file(),
        "weight_file": str(weights),
        "weight_exists": weights.is_file(),
    }
    if weights.is_file():
        gate1["weight_bytes"] = weights.stat().st_size
        gate1["weight_sha256"] = sha256_file(weights)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        gate1["ckpt_keys"] = sorted(ckpt.keys()) if isinstance(ckpt, dict) else "not_a_dict"
        gate1["weight_scale_letter"] = args.scale
        gate1["weight_nc"] = int(ckpt["model"].model[-1].nc) if isinstance(ckpt, dict) and "model" in ckpt else None

    # training mechanism (spec section 14): read the local installation, never the web
    mech = {}
    try:
        from ultralytics.cfg import get_cfg
        d = get_cfg()
        mech["defaults"] = {k: d[k] for k in ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                                              "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic", "amp", "seed",
                                              "degrees", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v", "batch") if k in d}
    except Exception as e:
        mech["defaults_error"] = repr(e)
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
        mech["trainer_class"] = DetectionTrainer.__module__ + "." + DetectionTrainer.__name__
    except Exception as e:
        mech["trainer_error"] = repr(e)
    try:
        m = YOLO("yolo26%s.yaml" % args.scale)
        crit = m.model.init_criterion()
        mech["loss_class"] = type(crit).__name__
        mech["loss_module"] = type(crit).__module__
        attrs = {}
        for a in ("assigner", "proj", "overlap", "box", "cls", "dfl", "bce", "one2many", "one2one"):
            if hasattr(crit, a):
                v = getattr(crit, a)
                attrs[a] = type(v).__name__ if not isinstance(v, (int, float, str, bool)) else v
        mech["loss_attrs"] = attrs
        mech["assigner_class"] = type(getattr(crit, "assigner", None)).__name__
        try:
            mech["assigner_source"] = inspect.getsourcefile(type(crit.assigner))
        except Exception:
            pass
    except Exception as e:
        mech["loss_error"] = repr(e)
    # STAL / small-target label assignment: search the installed source only
    srcroot = Path(ultralytics.__file__).parent
    stal_hits = []
    for p in srcroot.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for kw in ("STAL", "stal", "small_target", "SmallTarget"):
            if kw in txt:
                n = txt.count(kw)
                stal_hits.append({"file": str(p.relative_to(srcroot)), "keyword": kw, "count": n})
    mech["stal_search_hits"] = stal_hits[:40]
    mech["stal_present"] = bool([h for h in stal_hits if h["keyword"] in ("STAL", "SmallTarget", "small_target")])
    gate1["training_mechanism"] = mech

    # ---------------- Gate 2: official P2 config loads ----------------
    gate2 = {"gate": 2, "requested_config": "yolo26%s-p2.yaml" % args.scale}
    base_model = YOLO(str(weights)) if weights.is_file() else None
    gate2["baseline"] = model_stats(base_model.model, args.imgsz) if base_model is not None else None
    p2 = YOLO("yolo26%s-p2.yaml" % args.scale)
    gate2["p2"] = model_stats(p2.model, args.imgsz)
    gate2["p2_resolved_yaml"] = getattr(p2, "yaml_file", None)
    gate2["verdict"] = "PASS" if gate2["p2"]["detect_levels"] == 4 else "FAIL"

    # ---------------- Gate 3: weight transfer ----------------
    gate3 = {"gate": 3}
    if base_model is None:
        gate3["verdict"] = "FAIL"
        gate3["reason"] = "pretrained weights missing: " + str(weights)
    else:
        base_sd = base_model.model.state_dict()
        p2_sd = p2.model.state_dict()
        base_keys, p2_keys = set(base_sd), set(p2_sd)
        matched = sorted(base_keys & p2_keys)
        missing = sorted(p2_keys - base_keys)
        unexpected = sorted(base_keys - p2_keys)
        ok = [k for k in matched if tuple(base_sd[k].shape) == tuple(p2_sd[k].shape)]
        mismatch = [k for k in matched if tuple(base_sd[k].shape) != tuple(p2_sd[k].shape)]
        n_ok = sum(base_sd[k].numel() for k in ok)
        n_missing = sum(p2_sd[k].numel() for k in missing)
        n_unexpected = sum(base_sd[k].numel() for k in unexpected)
        n_mismatch = sum(p2_sd[k].numel() for k in mismatch)
        total_p2 = sum(v.numel() for v in p2_sd.values())
        groups = {}
        for k in p2_keys:
            g = group_of(k)
            d = groups.setdefault(g, {"total": 0, "transferred": 0})
            d["total"] += p2_sd[k].numel()
            if k in ok:
                d["transferred"] += base_sd[k].numel()
        gate3.update({
            "source_weights": str(weights),
            "source_scale": args.scale,
            "target_config": "yolo26%s-p2.yaml" % args.scale,
            "p2_total_params": total_p2,
            "transferred_params": n_ok,
            "transferred_tensors": len(ok),
            "shape_mismatch_tensors": len(mismatch),
            "shape_mismatch_params": n_mismatch,
            "missing_tensors": len(missing),
            "missing_params": n_missing,
            "unexpected_tensors": len(unexpected),
            "unexpected_params": n_unexpected,
            "coverage_pct": round(100.0 * n_ok / total_p2, 2) if total_p2 else 0.0,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": mismatch,
            "per_layer": dict(sorted(groups.items(), key=lambda kv: int(kv[0].split("_")[1]) if kv[0].startswith("layer_") else 9999)),
        })
        compatible = {k: v for k, v in base_sd.items() if k in p2_sd and tuple(base_sd[k].shape) == tuple(p2_sd[k].shape)}
        res = p2.model.load_state_dict(compatible, strict=False)
        gate3["strict_false_on_compatible_subset"] = {"submitted_tensors": len(compatible),
                                                      "missing_keys": len(res.missing_keys),
                                                      "unexpected_keys": len(res.unexpected_keys)}
        try:
            native = p2.model.load(str(weights))
            gate3["ultralytics_native_load"] = {"intersected_tensors": len(native) if native else 0}
        except Exception as e:
            gate3["ultralytics_native_load_error"] = repr(e)
        def _cov(lo, hi):
            tot = sum(d["total"] for k, d in groups.items() if k.startswith("layer_") and lo <= int(k.split("_")[1]) <= hi)
            tr = sum(d["transferred"] for k, d in groups.items() if k.startswith("layer_") and lo <= int(k.split("_")[1]) <= hi)
            return (round(100.0 * tr / tot, 2) if tot else None)
        gate3["layer_0_10_coverage_pct"] = _cov(0, 10)
        gate3["layer_11_18_coverage_pct"] = _cov(11, 18)
        gate3["layer_19_99_coverage_pct"] = _cov(19, 99)
        bad_bb = gate3["layer_0_10_coverage_pct"]
        gate3["verdict"] = "PASS" if (gate3["coverage_pct"] >= 40.0 and (bad_bb or 0) >= 80.0) else "FAIL"
        gate3["stop_condition_backbone"] = ("STOP: backbone (layers 0-10) transfer below 80%" if (bad_bb is not None and bad_bb < 80.0) else "OK")

    for name, obj in (("yolo26_v1_gate1_model_audit.json", gate1),
                      ("yolo26_v1_gate2_p2_config.json", gate2),
                      ("yolo26_p2_weight_transfer.json", gate3)):
        (metrics / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
        print("wrote", metrics / name)
    print(json.dumps({"gate1_p2_config": gate1.get("official_p2_config_present"),
                      "gate2_verdict": gate2["verdict"],
                      "gate2_p2": gate2["p2"],
                      "gate3_verdict": gate3["verdict"],
                      "gate3_coverage_pct": gate3.get("coverage_pct"),
                      "gate3_transferred_params": gate3.get("transferred_params")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())