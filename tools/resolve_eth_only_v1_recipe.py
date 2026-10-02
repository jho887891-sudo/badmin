#!/usr/bin/env python3
"""Resolve the official ETH training recipe and load the frozen experiment config (Task 1).

Read-only with respect to the ETH repository. Raises RecipeResolutionError instead of guessing whenever a
fact about official data membership is ambiguous, per the implementation plan's Review Focus #1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

PINNED_REQUIRED = ["model", "imgsz", "optimizer", "lr0", "freeze", "epochs", "nbs", "batch",
                   "batch_fallback", "never_touch", "classes", "train_domains", "evaluation_only_sets",
                   "selection_metric"]
EXPECTED = {"model": "yolo26s.pt", "imgsz": 1024, "batch": 8, "drop_optimizer_auto": True}


class RecipeResolutionError(RuntimeError):
    """Raised when the official recipe cannot be resolved from evidence."""


def load_experiment_config(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise RecipeResolutionError("experiment config not found: %s" % p)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    missing = [k for k in PINNED_REQUIRED if k not in cfg]
    if missing:
        raise RecipeResolutionError("config is missing pinned keys: %s" % missing)
    if cfg["model"] != EXPECTED["model"]:
        raise RecipeResolutionError("model must be %s (got %s)" % (EXPECTED["model"], cfg["model"]))
    if int(cfg["imgsz"]) != EXPECTED["imgsz"]:
        raise RecipeResolutionError("imgsz must stay 1024 (got %s)" % cfg["imgsz"])
    if str(cfg["optimizer"]).strip().lower() == "auto":
        raise RecipeResolutionError("optimizer=auto is forbidden by the plan")
    if list(cfg["batch_fallback"]) != [8, 6, 4]:
        raise RecipeResolutionError("batch fallback ladder must be [8, 6, 4]")
    for key in ("imgsz", "nbs", "optimizer", "lr0"):
        if key not in list(cfg["never_touch"]):
            raise RecipeResolutionError("never_touch must include %s" % key)
    return cfg


def _images_of(set_dir: Path) -> list:
    out = []
    for split in ("train", "val", "test"):
        d = set_dir / "images" / split
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.is_file()))
    return out


def _labels_of(set_dir: Path) -> list:
    out = []
    for split in ("train", "val", "test"):
        d = set_dir / "labels" / split
        if d.is_dir():
            out.extend(sorted(p for p in d.glob("*.txt")))
    return out


def resolve_official_recipe(eth_repo, data_root=None) -> dict:
    eth_repo = Path(eth_repo)
    cfg_path = eth_repo / "runs" / "final-model" / "config.json"
    if not cfg_path.exists():
        raise RecipeResolutionError("official config.json not found (expected %s)" % cfg_path)
    doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    data_blk = doc.get("data") or {}
    diff_blk = doc.get("diff_levels") or {}
    train_locs = [str(x) for x in (data_blk.get("train") or [])]
    train_diffs = [str(x) for x in (diff_blk.get("train") or [])]
    if not train_locs or not train_diffs:
        raise RecipeResolutionError(
            "official training membership is ambiguous: data.train=%s diff_levels.train=%s"
            % (train_locs, train_diffs))
    if data_root is None:
        data_root = eth_repo.parent / "eth_shuttle_detection"
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise RecipeResolutionError("dataset root not found: %s" % data_root)

    yaml_paths = sorted(str(p) for p in data_root.glob("*.yaml"))
    class_names = {}
    for y in yaml_paths:
        try:
            yd = yaml.safe_load(Path(y).read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for k, v in (yd.get("names") or {}).items():
            try:
                class_names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

    membership, evidence_dirs = {}, []
    counts = {"positives_train_dirs": 0, "official_negative_images": 0, "holdout_images": 0,
              "other_images": 0, "positive_empty_labels": 0, "negative_empty_labels": 0,
              "non_empty_labels_total": 0, "empty_labels_total": 0}
    for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = d.name
        loc, _, diff = name.rpartition("_")
        if not loc:
            loc, diff = name, "?"
        images = _images_of(d)
        labels = _labels_of(d)
        empty = 0
        for lb in labels:
            try:
                if not lb.read_text(encoding="utf-8", errors="replace").strip():
                    empty += 1
            except OSError:
                pass
        # Review Focus #1: membership must be explicit. coco_train IS listed in the official train locations
        # but its role is negatives/backgrounds (fraction_coco_train), so it must be classified BEFORE the
        # positives rule, otherwise COCO images are silently counted as shuttle positives.
        if loc == "coco_train":
            role, trained = "official_negatives", True
            counts["official_negative_images"] += len(images)
            counts["negative_empty_labels"] += empty
        elif loc == "coco_val":
            role, trained = "benchmark_holdout", False
            counts["holdout_images"] += len(images)
        elif loc in train_locs and diff in train_diffs:
            role, trained = "official_positives", True
            counts["positives_train_dirs"] += len(images)
            counts["positive_empty_labels"] += empty
        elif loc in train_locs:
            role, trained = "same_location_holdout", False
            counts["holdout_images"] += len(images)
        else:
            role, trained = "other", False
            counts["other_images"] += len(images)
        counts["non_empty_labels_total"] += len(labels) - empty
        counts["empty_labels_total"] += empty
        membership[name] = {"location": loc, "difficulty": diff, "role": role,
                            "in_official_training": trained, "images": len(images),
                            "labels": len(labels), "empty_labels": empty}
        if role == "official_positives":
            counts.setdefault("positives_by_location", {})
            counts["positives_by_location"][loc] = counts["positives_by_location"].get(loc, 0) + len(images)
        evidence_dirs.append({"dir": name, "images": len(images), "labels": len(labels), "empty": empty})

    override = {
        "imgsz": {"official": doc.get("imgsz"), "plan": 1024},
        "epochs": {"official": doc.get("epochs"), "plan": 50},
        "batch": {"official": doc.get("batch_size"), "plan": 8},
        "nbs": {"official": doc.get("nbs"), "plan": 32},
        "optimizer": {"official": doc.get("optim"), "plan": "AdamW"},
        "lr": {"official": doc.get("lr"), "plan": 1e-4},
    }
    rec = {
        "status": "RESOLVED",
        "eth_repo": str(eth_repo),
        "data_root": str(data_root),
        "official_model_name": doc.get("model_name"),
        "official_imgsz": doc.get("imgsz"),
        "official_epochs": doc.get("epochs"),
        "official_batch_size": doc.get("batch_size"),
        "official_nbs": doc.get("nbs"),
        "official_optimizer": doc.get("optim"),
        "official_scheduler": doc.get("scheduler"),
        "official_lr": doc.get("lr"),
        "official_momentum": doc.get("momentum"),
        "official_amsgrad": doc.get("amsgrad"),
        "official_weight_decay": doc.get("weight_decay"),
        "official_confidence": doc.get("confidence"),
        "official_dist_threshold": doc.get("dist_threshold"),
        "official_seed": doc.get("seed"),
        "official_train_locations": train_locs,
        "official_train_difficulties": train_diffs,
        "official_val_difficulties": (diff_blk.get("val") or []),
        "official_fraction_train": doc.get("fraction_train"),
        "official_fraction_coco_train": doc.get("fraction_coco_train"),
        "official_loss": doc.get("loss"),
        "official_augmentations": doc.get("augmentations") or {},
        "official_negative_sources": ["coco_train"],
        "official_negative_policy": "coco_train images at fraction_coco_train=%.3f" % (
            float(doc.get("fraction_coco_train") or 0.0)),
        "eth_iphone_decision": "EXCLUDED: the official config.json lists 12 training locations and none is an "
                               "iPhone location, so ETH iPhone frames are outside the released model's training "
                               "distribution (plan: exclude and record when uncertain).",
        "class_names": class_names,
        "location_membership": membership,
        "counts": counts,
        "plan_overrides_vs_official": override,
        "evidence": {"config_json": str(cfg_path),
                     "config_json_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                     "dataset_yamls": yaml_paths,
                     "dataset_dirs": evidence_dirs,
                     "config_raw": doc},
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve the official ETH recipe for the ETH-only V1 experiment.")
    ap.add_argument("--eth-repo", required=True, help="path to the official shuttle_detection checkout")
    ap.add_argument("--eth-data", default=None, help="path to the eth_shuttle_detection dataset root")
    ap.add_argument("--output", default=None)
    ap.add_argument("--config", default=None, help="also validate the frozen experiment config")
    a = ap.parse_args()
    try:
        rec = resolve_official_recipe(a.eth_repo, a.eth_data)
    except RecipeResolutionError as exc:
        print("RECIPE_UNRESOLVED: %s" % exc, file=sys.stderr)
        return 2
    if a.config:
        cfg = load_experiment_config(a.config)
        rec["experiment_config"] = {"path": str(a.config), "model": cfg["model"], "imgsz": cfg["imgsz"],
                                    "batch": cfg["batch"], "nbs": cfg["nbs"], "epochs": cfg["epochs"],
                                    "optimizer": cfg["optimizer"], "lr0": cfg["lr0"]}
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % out)
    print(json.dumps({k: rec[k] for k in ["status", "official_model_name", "official_imgsz",
                                          "official_train_locations", "official_train_difficulties",
                                          "official_confidence", "official_dist_threshold", "counts"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
