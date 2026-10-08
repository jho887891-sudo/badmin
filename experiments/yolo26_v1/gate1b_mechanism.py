"""Gate 1 follow-up: training-mechanism details read from the local ultralytics install."""
import json, os, re, sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\_yolo26v1\cfg")
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT as D

keys = ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
        "cos_lr", "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic", "amp", "seed",
        "degrees", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v", "batch", "imgsz",
        "multi_scale", "rect", "cache", "workers", "patience", "pretrained", "val", "plots"]
defaults = {k: D[k] for k in keys if k in D}

m = YOLO("yolo26s.yaml")
crit = m.model.init_criterion()
out = {"loss_class": type(crit).__name__, "loss_module": type(crit).__module__, "defaults": defaults}
sub = {}
for name in ("one2many", "one2one"):
    c = getattr(crit, name, None)
    if c is None:
        continue
    a = getattr(c, "assigner", None)
    sub[name] = {
        "class": type(c).__name__,
        "assigner_class": type(a).__name__ if a is not None else None,
        "assigner_module": type(a).__module__ if a is not None else None,
        "assigner_kwargs": {k: getattr(a, k, None) for k in ("topk", "num_classes", "alpha", "beta", "eps")} if a is not None else None,
        "hyp": {k: (getattr(c, k) if isinstance(getattr(c, k), (int, float, str, bool, type(None))) else type(getattr(c, k)).__name__) for k in ("box", "cls", "dfl", "bce") if hasattr(c, k)},
    }
out["losses"] = sub
out["head"] = {"end2end": getattr(m.model, "end2end", None), "names": getattr(m.model, "names", None)}

srcroot = Path(ultralytics.__file__).parent
stal = []
for p in srcroot.rglob("*.py"):
    try:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(r"\bSTAL\b", line):
                stal.append({"file": str(p.relative_to(srcroot)), "line": i, "text": line.strip()[:200]})
    except Exception:
        continue
out["stal_lines"] = stal
out["stal_enabled_by_default"] = bool([s for s in stal if "STAL" in s["text"] and "False" not in s["text"]])
txt = json.dumps(out, indent=1, ensure_ascii=False, default=str)
print(txt)
Path(r"E:\具身智能\badmin_project\outputs\shuttle_capability\metrics\yolo26_v1_gate1b_mechanism.json").write_text(txt, encoding="utf-8")