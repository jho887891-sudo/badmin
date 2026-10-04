#!/usr/bin/env python3
"""YOLO26 V1 trainer (spec sections 11-17, 23, 27-29).

Stage A = P2 warm-up, Stage B = main, Stage C = real-domain finish, plus --smoke for Gate 5.
Everything comes from configs/shuttle_detection/yolo26_p2_v1.yaml; the train/val lists are
derived from the frozen manifest so the same split is reused at 640/960/1024/1280.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shuttle_detection/yolo26_p2_v1.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--stage", choices=["A", "B", "C", "smoke"], default="smoke")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None, help="use only N training images (smoke)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--val", type=int, default=1, help="1 = run validation each epoch, 0 = skip (benchmarks)")
    ap.add_argument("--map", action="append", default=[], help="LOCAL_PREFIX=HOST_PREFIX, repeatable; rewrites manifest paths for the training host")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers (default from config)")
    ap.add_argument("--init", default=None, help="initial weights for this stage; default = config pretrained (COCO yolo26s.pt). Stage B/C must pass the previous stage best.pt")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(repo / args.config).read_text(encoding="utf-8"))
    imgsz = args.imgsz or int(cfg["run"]["imgsz"])
    batch = args.batch or int(cfg["run"]["physical_batch"])
    device = args.device or str(cfg["run"]["device"])
    seed = int(cfg["run"]["seed"])
    stage = args.stage
    stage_cfg = cfg["stages"]["A" if stage == "smoke" else stage]
    epochs = args.epochs or (1 if stage == "smoke" else int(stage_cfg["epochs"]))
    run_id = args.run_id or ("%s_imgsz%d_%s" % (stage, imgsz, time.strftime("%Y%m%d-%H%M%S")))

    manifest_path = Path(args.manifest) if args.manifest else (repo / cfg["data"]["manifest"])
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    sampler = json.loads((repo / cfg["data"]["sampler"]).read_text(encoding="utf-8"))
    weights = sampler["bucket_weight"]

    outdir = repo / cfg["outputs"]["runs_dir"] / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    pos = [r for r in train_rows if int(r["n_boxes"]) > 0]
    neg = [r for r in train_rows if int(r["n_boxes"]) == 0]
    hard = [r for r in neg if r["source"] == "hard_negative"]
    plain = [r for r in neg if r["source"] != "hard_negative"]
    weighted = []
    for r in pos:
        weighted.extend([r["image"]] * max(1, int(round(weights.get(r["bucket"], 1.0)))))
    n_neg_slots = int(round(len(weighted) * cfg["data"]["negative_ratio"] / (1.0 - cfg["data"]["negative_ratio"])))
    hw = int(cfg["data"]["hard_negative_weight"])
    hard_slots = min(len(hard) * hw, n_neg_slots // 2)
    plain_slots = max(0, n_neg_slots - hard_slots)
    for i in range(hard_slots):
        weighted.append(hard[i % max(1, len(hard))]["image"])
    for i in range(plain_slots):
        weighted.append(plain[i % max(1, len(plain))]["image"])
    if stage == "smoke" and args.subset:
        weighted = weighted[: args.subset]

    maps = []
    for m in (args.map or []):
        if "=" in m:
            a, b = m.split("=", 1)
            maps.append((a, b))

    def remap(x: str) -> str:
        for a, b in maps:
            if x.startswith(a):
                rest = x[len(a):]
                out = (b + rest).replace("\\", "/")
                while "//" in out:
                    out = out.replace("//", "/")
                return out
        return x.replace(os.sep, "/")

    def write_list(name, paths):
        p = outdir / name
        p.write_text(chr(10).join(remap(x) for x in paths) + chr(10), encoding="utf-8")
        return p

    tr = write_list("train_resolved.txt", weighted)
    va = write_list("val_resolved.txt", [r["image"] for r in val_rows])
    data_yaml = outdir / "dataset.yaml"
    data_yaml.write_text(chr(10).join([
        "path: " + str(outdir).replace(os.sep, "/"),
        "train: " + tr.name,
        "val: " + va.name,
        "nc: 1",
        "names:",
        "  0: shuttlecock",
    ]) + chr(10), encoding="utf-8")

    aug = cfg["augmentation"]
    freeze_raw = stage_cfg.get("freeze", "none")
    if isinstance(freeze_raw, str):
        freeze = 11 if freeze_raw.strip().lower() == "backbone" else 0   # layers 0-10 are the yolo26s-p2 backbone
    else:
        freeze = int(freeze_raw)
    overrides = dict(
        data=str(data_yaml),
        model=str(cfg["model"]["config"]),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        seed=seed,
        amp=bool(cfg["run"]["amp"]),
        nbs=int(cfg["run"]["effective_batch"]),
        freeze=freeze,
        workers=int(args.workers) if args.workers is not None else int(cfg["run"]["workers"]),
        project=str(repo / cfg["outputs"]["runs_dir"]),
        name=run_id,
        exist_ok=True,
        pretrained=str(args.init) if args.init else str(repo / cfg["model"]["pretrained"]),
        optimizer=str(cfg["run"].get("optimizer", "auto")),
        lr0=float(stage_cfg["lr0"]),
        warmup_bias_lr=float(cfg["run"].get("warmup_bias_lr", 0.1)),
        cos_lr=False,
        mosaic=float(stage_cfg["mosaic"]),
        mixup=float(aug["mixup"]),
        cutmix=float(aug["cutmix"]),
        copy_paste=float(aug["copy_paste"]),
        close_mosaic=int(aug["close_mosaic"]),
        degrees=float(aug["degrees"]),
        translate=float(aug["translate"]),
        scale=float(aug["scale"]),
        fliplr=float(aug["fliplr"]),
        hsv_h=float(aug["hsv_h"]),
        hsv_s=float(aug["hsv_s"]),
        hsv_v=float(aug["hsv_v"]),
        plots=True,
        val=bool(args.val),
    )
    (outdir / "resolved_config.yaml").write_text(yaml.safe_dump(overrides, sort_keys=True), encoding="utf-8")
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": __import__("torch").__version__,
        "cuda_available": __import__("torch").cuda.is_available(),
        "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None,
        "ultralytics": __import__("ultralytics").__version__,
        "seed": seed,
        "stage": stage,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "train_images_resolved": len(weighted),
        "val_images": len(val_rows),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    }
    (outdir / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    (outdir / "sampler.json").write_text(json.dumps({**sampler, "resolved_bucket_counts": dict(Counter(
        r["bucket"] for r in pos)), "resolved_epoch_length": len(weighted)}, indent=1), encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(str(cfg["model"]["config"]))

    # ---------------- LR / memory instrumentation (ISSUE-026) ----------------
    import json as _json
    import time as _time

    probe_path = outdir / "lr_probe.jsonl"
    mem_path = outdir / "mem_probe.jsonl"
    name_by_id = {}
    for _n, _p in model.model.named_parameters():
        name_by_id[id(_p)] = _n

    _meminfo_state = {}

    def _meminfo():
        info = {}
        try:
            for line in open("/proc/meminfo", "r"):
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        except Exception:
            pass
        out = {}
        for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if k in info:
                out[k + "_GB"] = round(int(info[k].split()[0]) / 1048576.0, 2)
        try:
            cur = int(open("/sys/fs/cgroup/memory.current").read().strip())
            out["cgroup_current_GB"] = round(cur / 1073741824.0, 2)
            mx = open("/sys/fs/cgroup/memory.max").read().strip()
            if mx != "max":
                out["cgroup_frac"] = round(cur / float(mx), 3)
        except Exception:
            pass
        # swap activity (pages/s -> KB/s) so "swap is busy" is measured, not guessed
        try:
            vm = {}
            for line in open("/proc/vmstat", "r"):
                k, v = line.split()
                if k in ("pswpin", "pswpout"):
                    vm[k] = int(v)
            now = _time.time()
            prev = _meminfo_state.get("vm")
            if prev and now > prev[0]:
                dt = now - prev[0]
                din = (vm.get("pswpin", 0) - prev[1].get("pswpin", 0)) * 4.0 / dt
                dout = (vm.get("pswpout", 0) - prev[1].get("pswpout", 0)) * 4.0 / dt
                out["swap_in_KBps"] = round(din, 1)
                out["swap_out_KBps"] = round(dout, 1)
            _meminfo_state["vm"] = (now, vm)
        except Exception:
            pass
        try:
            mx = open("/sys/fs/cgroup/memory.max").read().strip()
            out["cgroup_max_GB"] = "max" if mx == "max" else round(int(mx) / 1073741824.0, 2)
        except Exception:
            pass
        try:
            for line in open("/proc/self/status", "r"):
                if line.startswith("VmRSS:"):
                    out["trainer_rss_GB"] = round(int(line.split()[1]) / 1048576.0, 2)
        except Exception:
            pass
        return out

    def _groups(opt):
        gs = []
        for gi, g in enumerate(opt.param_groups):
            params = g.get("params", [])
            names = [name_by_id.get(id(p), "?") for p in params[:3]]
            gs.append({"group": gi, "lr": g.get("lr"), "initial_lr": g.get("initial_lr"),
                       "weight_decay": g.get("weight_decay"), "n_tensors": len(params),
                       "examples": names})
        return gs

    n_params_total = sum(p.numel() for p in model.model.parameters())
    n_params_train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    bb = tuple("model.%d." % i for i in range(11))
    n_bb_total = sum(p.numel() for n_, p in model.model.named_parameters() if n_.startswith(bb))
    n_bb_train = sum(p.numel() for n_, p in model.model.named_parameters() if n_.startswith(bb) and p.requires_grad)

    def _on_train_start(trainer):
        a = trainer.args
        cfgrec = {
            "lr0": a.lr0, "lrf": a.lrf, "cos_lr": a.cos_lr, "optimizer_arg": a.optimizer,
            "momentum": a.momentum, "weight_decay": a.weight_decay,
            "warmup_epochs": a.warmup_epochs, "warmup_momentum": a.warmup_momentum,
            "warmup_bias_lr": a.warmup_bias_lr, "nbs": a.nbs, "batch": a.batch,
            "resolved_optimizer": type(trainer.optimizer).__name__ if trainer.optimizer else None,
            "resolved_scheduler": type(trainer.scheduler).__name__ if getattr(trainer, "scheduler", None) else None,
            "accumulate": getattr(trainer, "accumulate", None),
            "freeze": overrides.get("freeze"),
            "params_total": n_params_total, "params_trainable": n_params_train,
            "backbone_layers_0_10_trainable_params": n_bb_train,
            "backbone_layers_0_10_total_params": n_bb_total,
            "backbone_unfrozen": bool(n_bb_train > 0),
            "param_groups": _groups(trainer.optimizer) if trainer.optimizer else [],
        }
        (outdir / "lr_config.json").write_text(_json.dumps(cfgrec, indent=1, default=str), encoding="utf-8")
        print("LR_CONFIG " + _json.dumps({k: cfgrec[k] for k in ("lr0", "warmup_epochs", "warmup_bias_lr",
              "resolved_optimizer", "resolved_scheduler", "backbone_unfrozen", "params_trainable")}, default=str), flush=True)

    probe_n = {"n": 0}

    def _on_batch_end(trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        probe_n["n"] += 1
        rec = {"phase": "batch", "epoch": getattr(trainer, "epoch", -1) + 1, "step": probe_n["n"],
               "groups": _groups(opt)}
        with open(probe_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, default=str) + chr(10))
        if probe_n["n"] % 100 == 0:
            mem = _meminfo()
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps({"epoch": rec["epoch"], "step": probe_n["n"], **mem}, default=str) + chr(10))
            danger = (mem.get("MemAvailable_GB", 99.0) < 5.0
                      or mem.get("swap_in_KBps", 0.0) + mem.get("swap_out_KBps", 0.0) > 20000.0
                      or (isinstance(mem.get("cgroup_frac"), float) and mem["cgroup_frac"] > 0.9))
            if danger:
                with open(outdir / "MEMORY_GUARD.txt", "a", encoding="utf-8") as f:
                    f.write("MemAvailable<5GB epoch=%s step=%s %s" % (rec["epoch"], probe_n["n"], _json.dumps(mem)) + chr(10))
                print("MEMORY_GUARD triggered: " + _json.dumps(mem), flush=True)
                try:
                    trainer.save_model()          # immediate checkpoint before stopping
                except Exception:
                    pass
                try:
                    trainer.stop_training = True
                    trainer.epochs = trainer.epoch + 1
                except Exception:
                    pass

    def _on_epoch_end(trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        rec = {"phase": "epoch_end", "epoch": getattr(trainer, "epoch", -1) + 1, "groups": _groups(opt),
               "mem": _meminfo()}
        with open(probe_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, default=str) + chr(10))

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_batch_end", _on_batch_end)
    model.add_callback("on_train_epoch_end", _on_epoch_end)

    t0 = time.time()
    model.train(**overrides)
    elapsed = time.time() - t0
    try:
        import torch
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        peak_mb = None

    best = outdir / "weights" / "best.pt"
    last = outdir / "weights" / "last.pt"
    summary = {
        "run_id": run_id,
        "stage": stage,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "elapsed_s": round(elapsed, 1),
        "best_pt": str(best) if best.exists() else None,
        "last_pt": str(last) if last.exists() else None,
        "train_images": len(weighted),
        "peak_vram_mb": peak_mb,
        "effective_batch_nbs": overrides.get("nbs"),
        "freeze_layers": freeze,
        "init_weights": overrides.get("pretrained"),
        "physical_batch": batch,
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())