#!/usr/bin/env python3
"""Stage B gate checker (spec: STAGE_B_EPOCH5_GATE / smoke acceptance).

Reads a run directory produced by tools/train_yolo26_v1.py and answers, with data only:
  1 box_loss trend   2 recall collapse   3 mAP50 drop   4 real LR per param group
  5 NaN/Inf          6 init weights      7 backbone unfrozen   8 memory
Verdict is one of PASS_CONTINUE_STAGE_B / FAIL_STOP_STAGE_B / UNKNOWN_NEED_MORE_EVIDENCE.

Gate logic v2 (2026-09-28) closes three holes of v1:
  H1 collapse: a catastrophic metric collapse FAILs on its own (box_loss need not rise);
     the box_loss-rise path now only needs the metrics to fall below 0.8x baseline.
  H2 memory: MemAvailable / swap I/O / cgroup now participate in the verdict, with the same
     thresholds as the trainer runtime guard; SwapFree==0 alone never fails (shared machine).
  H3 LR: limits come from the *design* targets in lr_config.json (initial_lr) x1.10, not from
     a loose constant and never from warmup-observed values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# thresholds (must stay in sync with tools/train_yolo26_v1.py runtime guard)
MEM_AVAILABLE_MIN_GB = 5.0
SWAP_IO_MAX_KBPS = 20.0 * 1024.0        # 20 MB/s expressed in KB/s
CGROUP_FRAC_MAX = 0.90
# catastrophic metric collapse = below half of the Stage A baseline
CATASTROPHIC_FRAC = 0.5
# trend collapse = box_loss rising for this many epochs AND metrics below this fraction
TREND_RISE_EPOCHS = 3
TREND_METRIC_FRAC = 0.8
# LR tolerance around the design target
LR_TOLERANCE = 1.10
# plan B fallback targets when lr_config.json cannot be trusted
FALLBACK_NORMAL_LR = 0.0011
FALLBACK_HEAD_LR = 0.0033


def load_results(run: Path):
    p = run / "results.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            k2 = (k or "").strip()
            try:
                clean[k2] = float(v)
            except (TypeError, ValueError):
                clean[k2] = v
        out.append(clean)
    return out


def series_from_results(res):
    return {
        "epoch": [int(r["epoch"]) for r in res],
        "box_loss": [r.get("train/box_loss") for r in res],
        "recall": [r.get("metrics/recall(B)") for r in res],
        "map50": [r.get("metrics/mAP50(B)") for r in res],
    }


def group_lrs(run: Path):
    """agg[epoch] = {"start": [...], "end": [...], "max": [...]}"""
    p = run / "lr_probe.jsonl"
    agg = {}
    if not p.exists():
        return agg
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ep = rec.get("epoch")
        lrs = [g.get("lr") for g in rec.get("groups", [])]
        if ep is None or not lrs:
            continue
        d = agg.setdefault(ep, {"start": None, "end": None, "max": [0.0] * len(lrs)})
        if d["start"] is None:
            d["start"] = lrs
        if rec.get("phase") == "epoch_end":
            d["end"] = lrs
        for i, v in enumerate(lrs):
            if v is not None and i < len(d["max"]) and v > d["max"][i]:
                d["max"][i] = v
    return agg


def mem_stats(run: Path):
    p = run / "mem_probe.jsonl"
    if not p.exists():
        return {}
    avail, swapfree, rss, frac, sw = [], [], [], [], []
    for line in open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "MemAvailable_GB" in d:
            avail.append(d["MemAvailable_GB"])
        if "SwapFree_GB" in d:
            swapfree.append(d["SwapFree_GB"])
        if "trainer_rss_GB" in d:
            rss.append(d["trainer_rss_GB"])
        if isinstance(d.get("cgroup_frac"), float):
            frac.append(d["cgroup_frac"])
        if "swap_in_KBps" in d or "swap_out_KBps" in d:
            sw.append((d.get("swap_in_KBps", 0.0) or 0.0) + (d.get("swap_out_KBps", 0.0) or 0.0))
    f = lambda xs: (round(min(xs), 2), round(max(xs), 2)) if xs else None
    return {"memavailable_minmax_GB": f(avail), "swapfree_minmax_GB": f(swapfree),
            "trainer_rss_max_GB": round(max(rss), 2) if rss else None,
            "cgroup_frac_max": round(max(frac), 3) if frac else None,
            "swap_kbps_max": round(max(sw), 1) if sw else None}


def resolve_lr_limits(lrcfg, override_normal=None, override_head=None):
    """Design-target LR limits.

    Plan A: read the design targets from lr_config.json param_groups "initial_lr"
            (head = max initial_lr, normal = min initial_lr) and allow LR_TOLERANCE.
            "initial_lr" is the configured target, never the warmup-observed "lr".
    Plan B: fixed fallback constants when the config cannot be trusted.
    Returns (normal_limit, head_limit, per_group_limits_or_None, info).
    """
    info = {"source": None, "target_normal": None, "target_head": None, "tolerance": LR_TOLERANCE}
    groups = lrcfg.get("param_groups") or []
    initials = [g.get("initial_lr") for g in groups if isinstance(g.get("initial_lr"), (int, float))]
    per_group_targets = None
    if override_normal is not None or override_head is not None:
        tn = override_normal if override_normal is not None else (min(initials) if initials else FALLBACK_NORMAL_LR / LR_TOLERANCE)
        th = override_head if override_head is not None else (max(initials) if initials else FALLBACK_HEAD_LR / LR_TOLERANCE)
        info["source"] = "cli override"
    elif len(initials) >= 2:
        tn, th = min(initials), max(initials)
        info["source"] = "lr_config.json param_groups.initial_lr (normal=min, head=max)"
        if len(initials) == len(groups):
            per_group_targets = [g.get("initial_lr") for g in groups]
    elif len(initials) == 1:
        tn = th = initials[0]
        info["source"] = "lr_config.json single initial_lr (no head boost)"
        per_group_targets = [initials[0]]
    else:
        tn, th = FALLBACK_NORMAL_LR / LR_TOLERANCE, FALLBACK_HEAD_LR / LR_TOLERANCE
        info["source"] = "plan B fallback constants"
    info["target_normal"] = round(float(tn), 8)
    info["target_head"] = round(float(th), 8)
    normal_limit = float(tn) * LR_TOLERANCE
    head_limit = float(th) * LR_TOLERANCE
    per_group_limits = None
    if per_group_targets is not None:
        per_group_limits = [round(float(t) * LR_TOLERANCE, 8) for t in per_group_targets]
    info["normal_limit"] = round(normal_limit, 8)
    info["head_limit"] = round(head_limit, 8)
    return normal_limit, head_limit, per_group_limits, info


def evaluate(series, lr_per_epoch, mem, lrcfg, init_weights, nan=False,
             baseline_map50=0.871, baseline_recall=0.76012, required_epochs=3,
             override_normal_lr=None, override_head_lr=None, rss_max_GB=None,
             label="", run=""):
    """Pure verdict function (no filesystem): returns the report dict."""
    box = [float(x) for x in series.get("box_loss", []) if isinstance(x, (int, float))]
    rec = [float(x) for x in series.get("recall", []) if isinstance(x, (int, float))]
    mapa = [float(x) for x in series.get("map50", []) if isinstance(x, (int, float))]
    epochs = list(series.get("epoch", []))

    rising = 0
    max_rising = 0
    for i in range(1, len(box)):
        if box[i] > box[i - 1]:
            rising += 1
            max_rising = max(max_rising, rising)
        else:
            rising = 0

    recall_min = min(rec) if rec else None
    map50_min = min(mapa) if mapa else None
    recall_collapse = bool(recall_min is not None and recall_min < CATASTROPHIC_FRAC * baseline_recall)
    map50_catastrophic = bool(map50_min is not None and map50_min < CATASTROPHIC_FRAC * baseline_map50)

    # H1: two-level collapse logic
    catastrophic_metric_collapse = bool(recall_collapse or map50_catastrophic)
    trend_metric_degraded = bool(
        (recall_min is not None and recall_min < TREND_METRIC_FRAC * baseline_recall)
        or (map50_min is not None and map50_min < TREND_METRIC_FRAC * baseline_map50))
    trend_collapse = bool(max_rising >= TREND_RISE_EPOCHS and trend_metric_degraded)
    collapse = bool(catastrophic_metric_collapse or trend_collapse)

    # H3: LR limits from design targets
    normal_limit, head_limit, per_group_limits, lr_info = resolve_lr_limits(
        lrcfg or {}, override_normal=override_normal_lr, override_head=override_head_lr)
    lr_violations = []
    for ep in sorted(lr_per_epoch):
        for i, v in enumerate(lr_per_epoch[ep].get("max", [])):
            if v is None:
                continue
            if per_group_limits and i < len(per_group_limits):
                limit = per_group_limits[i]
            else:
                limit = head_limit if (i % 2 == 0) else normal_limit
            if v > limit:
                lr_violations.append({"epoch": ep, "group": i, "lr": round(float(v), 6), "limit": round(float(limit), 6)})

    # H2: memory participates in the verdict, same thresholds as the runtime guard
    mem_fail = False
    mem_reasons = []
    mm = None
    if mem.get("memavailable_minmax_GB"):
        mm = mem["memavailable_minmax_GB"][0]
    if mm is None and mem.get("MemAvailable_GB") is not None:
        mm = mem["MemAvailable_GB"]
    if isinstance(mm, (int, float)) and mm < MEM_AVAILABLE_MIN_GB:
        mem_fail = True
        mem_reasons.append("MemAvailable dropped below %.1f GB: %.1f GB" % (MEM_AVAILABLE_MIN_GB, mm))
    sw = mem.get("swap_kbps_max")
    if isinstance(sw, (int, float)) and sw > SWAP_IO_MAX_KBPS:
        mem_fail = True
        mem_reasons.append("swap IO exceeded %.0f MB/s: %.1f MB/s" % (SWAP_IO_MAX_KBPS / 1024.0, sw / 1024.0))
    cf = mem.get("cgroup_frac_max")
    if isinstance(cf, (int, float)) and cf > CGROUP_FRAC_MAX:
        mem_fail = True
        mem_reasons.append("cgroup memory usage exceeded %.0f%%: %.1f%%" % (CGROUP_FRAC_MAX * 100, cf * 100))
    if isinstance(rss_max_GB, (int, float)) and rss_max_GB < 0:
        mem_fail = True
        mem_reasons.append("invalid trainer RSS: %.2f GB" % rss_max_GB)
    # SwapFree == 0 alone must NOT fail: on a shared box other users can exhaust swap.

    backbone_ok = bool((lrcfg or {}).get("backbone_unfrozen"))
    init_ok = bool(init_weights and "gate6A" in init_weights and "best.pt" in init_weights)
    enough = len(box) >= required_epochs

    hard_fail = bool(nan or lr_violations or (not backbone_ok) or (not init_ok) or mem_fail)

    verdict = "UNKNOWN_NEED_MORE_EVIDENCE"
    reasons = []
    if not enough:
        reasons.append("only %d/%d epochs finished" % (len(box), required_epochs))
    elif hard_fail:
        verdict = "FAIL_STOP_STAGE_B"
        if nan:
            reasons.append("NaN/Inf in results")
        if lr_violations:
            reasons.append("LR above design target x%.2f (%s): %s" % (lr_info["tolerance"], lr_info["source"], json.dumps(lr_violations[:4])))
        if not backbone_ok:
            reasons.append("backbone not fully unfrozen")
        if not init_ok:
            reasons.append("init weights are not Stage A best.pt (%s)" % init_weights)
        reasons.extend(mem_reasons)
        if collapse:
            reasons.append("collapse also detected (catastrophic=%s, trend=%s)" % (catastrophic_metric_collapse, trend_collapse))
    elif collapse:
        verdict = "FAIL_STOP_STAGE_B"
        if catastrophic_metric_collapse:
            reasons.append("catastrophic metric collapse: recall_min=%.5f (limit %.5f) / map50_min=%.5f (limit %.5f)"
                           % (recall_min if recall_min is not None else -1, CATASTROPHIC_FRAC * baseline_recall,
                              map50_min if map50_min is not None else -1, CATASTROPHIC_FRAC * baseline_map50))
        if trend_collapse:
            reasons.append("box_loss rose %d epochs in a row AND metrics below %.1fx baseline (recall_min=%s, map50_min=%s)"
                           % (max_rising, TREND_METRIC_FRAC, recall_min, map50_min))
    else:
        verdict = "PASS_CONTINUE_STAGE_B"
        reasons.append("no NaN, LR within design target x%.2f, backbone unfrozen, init=Stage A best.pt, "
                       "no catastrophic/trend collapse, memory within runtime-guard thresholds" % lr_info["tolerance"])

    return {
        "run": str(run), "label": label, "epochs_finished": len(box), "verdict": verdict, "reasons": reasons,
        "baseline": {"map50": baseline_map50, "recall": baseline_recall},
        "series": {"epoch": epochs, "box_loss": box, "recall": rec, "map50": mapa},
        "max_consecutive_box_loss_rise": max_rising,
        "recall_min": recall_min, "map50_min": map50_min,
        "catastrophic_metric_collapse": catastrophic_metric_collapse,
        "trend_collapse": trend_collapse, "trend_metric_degraded": trend_metric_degraded,
        "collapse": collapse,
        "lr_per_epoch": {str(k): {"start": lr_per_epoch[k].get("start"), "end": lr_per_epoch[k].get("end"),
                                  "max": lr_per_epoch[k].get("max")} for k in sorted(lr_per_epoch)},
        "lr_limits": lr_info, "lr_violations": lr_violations,
        "memory": mem, "memory_fail": mem_fail, "memory_reasons": mem_reasons,
        "init_weights": init_weights, "backbone_unfrozen": backbone_ok, "nan": nan,
        "hard_fail": hard_fail, "required_epochs": required_epochs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--baseline-map50", type=float, default=0.871)
    ap.add_argument("--baseline-recall", type=float, default=0.76012)
    ap.add_argument("--required-epochs", type=int, default=3)
    ap.add_argument("--target-normal-lr", type=float, default=None,
                    help="design target for the normal param groups (default: read from lr_config.json)")
    ap.add_argument("--target-head-lr", type=float, default=None,
                    help="design target for the detection-head param groups (default: read from lr_config.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)

    res = load_results(run)
    series = series_from_results(res)
    lrs = group_lrs(run)
    mem = mem_stats(run)
    cfgp = run / "lr_config.json"
    lrcfg = json.loads(cfgp.read_text(encoding="utf-8")) if cfgp.exists() else {}
    rc = run / "resolved_config.yaml"
    init_weights = None
    if rc.exists():
        for line in rc.read_text(encoding="utf-8").splitlines():
            if line.startswith("pretrained:"):
                init_weights = line.split(":", 1)[1].strip()
    nan = any(isinstance(x, float) and (math.isnan(x) or math.isinf(x))
              for r in res for x in r.values() if isinstance(x, float))

    out = evaluate(series, lrs, mem, lrcfg, init_weights, nan=nan,
                   baseline_map50=args.baseline_map50, baseline_recall=args.baseline_recall,
                   required_epochs=args.required_epochs,
                   override_normal_lr=args.target_normal_lr, override_head_lr=args.target_head_lr,
                   label=args.label, run=str(run))
    out["lr_config"] = lrcfg
    txt = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())