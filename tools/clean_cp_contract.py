#!/usr/bin/env python3
"""Frozen-contract loader for CLEAN_OUTDOOR_CP_V1 (Task 1 of the clean-outdoor copy-paste plan).

Single source of truth for three things the single training run must not drift from:

1. the frozen V2 training recipe - read from the V2 run's own Ultralytics args.yaml, which the plan calls
   the V2 resolved_config (this repository never stored a separate resolved_config.json for V2);
2. the frozen V2 dataset composition - derived from the V2 train/val manifests, never typed in by hand;
3. the official pretrained checkpoint, frozen by byte count and SHA256.

Missing or contradicting artifacts raise ContractBlocked. There is deliberately no default value
anywhere in this module: Task 1 of the plan says "missing => STOP", and the gate name is
BLOCKED_MISSING_V2_ARTIFACTS.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

EXPERIMENT = "eth_real_clean_outdoor_copypaste_v1"
RUN_ID = "CLEAN_OUTDOOR_CP_V1"
V2_EXPERIMENT = "eth_real_hardneg_yolo26s_v2"

# --- frozen V2 artifacts (paths are repo-relative; the contract map records bytes + sha256) --------
V2_ARGS_PATH = "configs/shuttle_detection/eth_real_hardneg_v2_resolved_args.yaml"
V2_TRAIN_MANIFEST_PATH = "data/eth_real_hardneg_v2_train_manifest.csv"
V2_VAL_MANIFEST_PATH = "data/eth_real_hardneg_v2_val_manifest.csv"
PRETRAINED_PATH = "assets/external/_staging/F_yolo/weights/yolo26s.pt"
V2_CHECKPOINT_REGISTRY_PATH = "configs/shuttle_detection/checkpoints_v1.yaml"
V2_CONFIG_PATH = "configs/eth_real_hardneg_yolo26s_v2.yaml"
V2_SPEC_PATH = "docs/superpowers/specs/2026-10-05-eth-real-hardneg-yolo26s-v2-design.md"
NEW_SPEC_PATH = "docs/superpowers/specs/2026-10-13-eth-coco-clean-outdoor-copypaste-hardneg-v1-design.md"
NEW_CONFIG_PATH = "configs/eth_real_clean_outdoor_copypaste_v1.yaml"
CANONICAL_EVALUATOR_PATH = "tools/eval_yolo26_v1.py"
CANONICAL_AUDIT_PATH = "tools/audit_localization_yolo26_v1.py"
TRAINING_LAUNCHER_PATH = "tools/train_eth_only_v1.py"

FROZEN_V2_ARGS_SHA256 = "41e96b029f250c12256743522af65e30dd4ac2cf5c62187f7513b4017c751c3b"
FROZEN_V2_ARGS_REMOTE = "/home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/args.yaml"
FROZEN_V2_TRAIN_SHA256 = "d2bd887357555c9262f769ba81d3bfd49350e1670e80a94ecdc1d166898a8d4d"
FROZEN_V2_VAL_SHA256 = "0ed405719af1d62763fb41fdad79b2581e0e7787da9a9c237c203a62ef735918"
FROZEN_PRETRAINED_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
FROZEN_PRETRAINED_BYTES = 20422725
FROZEN_PRETRAINED_SOURCE = "/home/T7/dgut/robot_sim/yolo26s.pt"

FROZEN_V2_TRAIN_ROWS = 15255
FROZEN_V2_POSITIVES = 13992
FROZEN_V2_ORDINARY_NEGATIVES = 551
FROZEN_V2_HARDNEG_EXPOSURES = 712
FROZEN_V2_HARDNEG_UNIQUE = 89
FROZEN_V2_HARDNEG_REPEAT = 8
FROZEN_V2_ROLE_ANOMALIES = 1
FROZEN_V2_VAL_ROWS = 2920
FROZEN_V2_VAL_LOCATIONS = ("glc_2", "uetlibergstrasse_2")

# The seven knobs the plan freezes for the new run (everything else is copied verbatim from V2).
FROZEN_KNOBS = {
    "imgsz": 1024,
    "epochs": 50,
    "optimizer": "AdamW",
    "lr0": 1.0e-4,
    "freeze": 0,
    "batch": 8,
    "nbs": 32,
    "seed": 42,
}

# V2 args.yaml keys that describe where output goes or are pure inference-IO switches; they are not part
# of the training recipe and the new config does not have to restate them.
IGNORED_V2_KEYS = frozenset({
    "task", "mode", "model", "data", "project", "name", "exist_ok", "save_dir", "time", "source",
    "tracker", "cfg", "format", "simplify", "workspace", "opset", "keras", "optimize", "dynamic",
    "save_json", "save_txt", "save_frames", "save_crop", "show", "show_labels", "show_conf",
    "show_boxes", "line_width", "visualize", "augment", "stream_buffer", "vid_stride", "embed",
    "conf", "classes", "nms", "resume", "vsplit", "val_period",
})

FORBIDDEN_TOKENS = ("a0", "p2", "second_model", "multi_model", "curriculum", "staged")
FORBIDDEN_VARIANT_PATTERN = re.compile(r"(?i)(^|[^a-z0-9])(a0|p2)([^a-z0-9]|$)")


class ContractBlocked(RuntimeError):
    """Raised when a frozen V2 artifact is missing, changed, or the new config contradicts it."""


def sha256_file(path) -> str:
    """SHA256 of a file, streamed so the 20 MB checkpoint and the 6 MB manifests are both cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path, what: str) -> Path:
    """Return the path as a Path or raise ContractBlocked naming exactly what is missing."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ContractBlocked("%s is missing: %s (BLOCKED_MISSING_V2_ARTIFACTS)" % (what, resolved))
    return resolved


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_yaml_mapping(path) -> dict:
    """Load a YAML mapping, refusing anything that is not a mapping."""
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ContractBlocked("expected a YAML mapping in %s" % path)
    return loaded


def load_v2_resolved_config(path) -> dict:
    """Load the V2 run's resolved Ultralytics args, verifying the frozen sha256 and required keys."""
    resolved = require_file(path, "V2 resolved_config (args.yaml)")
    digest = sha256_file(resolved)
    if digest != FROZEN_V2_ARGS_SHA256:
        raise ContractBlocked(
            "V2 resolved recipe changed: %s has sha256 %s, frozen value is %s"
            % (resolved, digest, FROZEN_V2_ARGS_SHA256)
        )
    args = load_yaml_mapping(resolved)
    required = set(FROZEN_KNOBS) | {"momentum", "weight_decay", "box", "cls", "dfl", "mosaic",
                                    "fliplr", "hsv_h", "deterministic", "warmup_epochs"}
    missing = sorted(required - set(args))
    if missing:
        raise ContractBlocked("V2 resolved recipe misses required keys: %s" % missing)
    return args


def load_manifest_counts(path) -> dict:
    """Derive the frozen V2 composition from a manifest instead of trusting a typed-in number."""
    resolved = require_file(path, "V2 manifest")
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ContractBlocked("V2 manifest is empty: %s" % resolved)
    # The V2 train manifest carries is_positive/is_negative/repeat_index; the val manifest only carries
    # is_negative and role. Both must resolve through this single derivation so no second convention exists.
    needed = {"is_negative", "role", "sha256", "location"}
    missing = sorted(needed - set(rows[0]))
    if missing:
        raise ContractBlocked("V2 manifest %s misses columns: %s" % (resolved, missing))

    def repeat_of(row) -> int:
        return int(row.get("repeat_index") or 0)

    def positive_of(row) -> bool:
        raw = row.get("is_positive")
        if raw is not None and str(raw).strip() != "":
            return _truthy(raw)
        return not _truthy(row["is_negative"])

    positives = [r for r in rows if positive_of(r)]
    negatives = [r for r in rows if _truthy(r["is_negative"])]
    hard = [r for r in negatives if repeat_of(r) > 0 or r["role"] == "hard_negative"]
    ordinary = [r for r in negatives if r not in hard]
    anomalies = [r for r in rows if r["role"] == "official_positives" and not positive_of(r)]

    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "rows": len(rows),
        "positives": len(positives),
        "ordinary_negatives": len(ordinary),
        "hard_negative_exposures": len(hard),
        "hard_negative_unique": len({r["sha256"] for r in hard}),
        "repeat_factor": max((repeat_of(r) for r in hard), default=-1) + 1,
        "role_column_anomalies": len(anomalies),
        "role_column_anomaly_examples": [
            {"image": r["image"], "role": r["role"], "sha256": r["sha256"], "n_boxes": r.get("n_boxes", "")}
            for r in anomalies[:5]
        ],
        "locations": sorted({r["location"] for r in rows}),
    }


def load_v2_contract(v2_resolved_config, v2_train_manifest, v2_val_manifest, pretrained) -> dict:
    """Freeze the whole V2 reference: recipe, dataset composition, val set and pretrained weights."""
    args = load_v2_resolved_config(v2_resolved_config)
    train = load_manifest_counts(v2_train_manifest)
    val = load_manifest_counts(v2_val_manifest)

    weights = require_file(pretrained, "official pretrained yolo26s.pt")
    weights_bytes = weights.stat().st_size
    weights_sha = sha256_file(weights)
    if weights_sha != FROZEN_PRETRAINED_SHA256 or weights_bytes != FROZEN_PRETRAINED_BYTES:
        raise ContractBlocked(
            "pretrained checkpoint is not the frozen official yolo26s.pt: %s has %d bytes / %s, frozen values "
            "are %d bytes / %s" % (weights, weights_bytes, weights_sha, FROZEN_PRETRAINED_BYTES, FROZEN_PRETRAINED_SHA256)
        )

    problems = []
    if train["rows"] != FROZEN_V2_TRAIN_ROWS:
        problems.append("train rows %d != %d" % (train["rows"], FROZEN_V2_TRAIN_ROWS))
    if train["positives"] != FROZEN_V2_POSITIVES:
        problems.append("positives %d != %d" % (train["positives"], FROZEN_V2_POSITIVES))
    if train["ordinary_negatives"] != FROZEN_V2_ORDINARY_NEGATIVES:
        problems.append("ordinary negatives %d != %d" % (train["ordinary_negatives"], FROZEN_V2_ORDINARY_NEGATIVES))
    if train["hard_negative_exposures"] != FROZEN_V2_HARDNEG_EXPOSURES:
        problems.append("hard-negative exposures %d != %d" % (train["hard_negative_exposures"], FROZEN_V2_HARDNEG_EXPOSURES))
    if train["hard_negative_unique"] != FROZEN_V2_HARDNEG_UNIQUE:
        problems.append("hard-negative unique %d != %d" % (train["hard_negative_unique"], FROZEN_V2_HARDNEG_UNIQUE))
    if train["repeat_factor"] != FROZEN_V2_HARDNEG_REPEAT:
        problems.append("hard-negative repeat %d != %d" % (train["repeat_factor"], FROZEN_V2_HARDNEG_REPEAT))
    if train["role_column_anomalies"] != FROZEN_V2_ROLE_ANOMALIES:
        problems.append("role-column anomalies %d != %d" % (train["role_column_anomalies"], FROZEN_V2_ROLE_ANOMALIES))
    if val["rows"] != FROZEN_V2_VAL_ROWS:
        problems.append("val rows %d != %d" % (val["rows"], FROZEN_V2_VAL_ROWS))
    if tuple(val["locations"]) != FROZEN_V2_VAL_LOCATIONS:
        problems.append("val locations %s != %s" % (val["locations"], list(FROZEN_V2_VAL_LOCATIONS)))
    if problems:
        raise ContractBlocked("frozen V2 dataset composition changed: " + "; ".join(problems))

    kwargs = {k: v for k, v in args.items() if k not in IGNORED_V2_KEYS}
    for key, value in FROZEN_KNOBS.items():
        if kwargs.get(key) != value:
            raise ContractBlocked("frozen V2 knob %s=%r, expected %r" % (key, kwargs.get(key), value))

    return {
        "experiment": V2_EXPERIMENT,
        "resolved_args_path": str(args),
        "resolved_args_remote": FROZEN_V2_ARGS_REMOTE,
        "resolved_args_sha256": FROZEN_V2_ARGS_SHA256,
        "resolved_kwargs": kwargs,
        "parity_keys": sorted(kwargs),
        "ignored_keys": sorted(IGNORED_V2_KEYS),
        "frozen_knobs": dict(FROZEN_KNOBS),
        "counts": {
            "train_rows": train["rows"],
            "positives": train["positives"],
            "ordinary_negatives": train["ordinary_negatives"],
            "hard_negative_exposures": train["hard_negative_exposures"],
            "hard_negative_unique": train["hard_negative_unique"],
            "repeat_factor": train["repeat_factor"],
            "role_column_anomalies": train["role_column_anomalies"],
            "role_column_anomaly_examples": train["role_column_anomaly_examples"],
            "train_sha256": train["sha256"],
            "val_rows": val["rows"],
            "val_negative_rows": val["ordinary_negatives"] + val["hard_negative_exposures"],
            "val_locations": list(val["locations"]),
            "val_sha256": val["sha256"],
        },
        "pretrained": {
            "path": str(weights),
            "bytes": weights_bytes,
            "sha256": weights_sha,
            "remote_source": FROZEN_PRETRAINED_SOURCE,
        },
    }


def assert_frozen_recipe(new_config: dict, v2_contract: dict) -> None:
    """The new run must reproduce the V2 recipe field by field, with the frozen knobs pinned."""
    if not isinstance(new_config, dict):
        raise ContractBlocked("new config must be a mapping of resolved training kwargs")

    optimizer = new_config.get("optimizer")
    if isinstance(optimizer, str) and optimizer.strip().lower() == "auto":
        raise ContractBlocked("optimizer 'auto' is forbidden: the V2 recipe pins optimizer=%r" % FROZEN_KNOBS["optimizer"])

    for key, value in FROZEN_KNOBS.items():
        if key not in new_config:
            raise ContractBlocked("frozen knob %s is missing from the new config" % key)
        if new_config[key] != value:
            raise ContractBlocked("frozen knob %s=%r, expected %r" % (key, new_config[key], value))

    v2_kwargs = v2_contract["resolved_kwargs"]
    parity_keys = v2_contract["parity_keys"]
    missing = [k for k in parity_keys if k not in new_config]
    if missing:
        raise ContractBlocked("new config must restate every frozen V2 recipe field; missing: %s" % sorted(missing))
    drift = {k: (new_config[k], v2_kwargs[k]) for k in parity_keys if new_config[k] != v2_kwargs[k]}
    if drift:
        raise ContractBlocked("new config drifts from the frozen V2 recipe: %s" % drift)
    unknown = sorted(set(new_config) - set(parity_keys) - set(v2_contract["ignored_keys"]))
    if unknown:
        raise ContractBlocked("new config introduces recipe fields that V2 never resolved: %s" % unknown)


def assert_single_model_config(cfg: dict) -> None:
    """Exactly one plain YOLO26s run: no A0 arm, no P2 head, no second model, no resuming V2."""
    if not isinstance(cfg, dict):
        raise ContractBlocked("config must be a mapping")

    if cfg.get("experiment") != EXPERIMENT:
        raise ContractBlocked("experiment must be %r, found %r" % (EXPERIMENT, cfg.get("experiment")))
    if cfg.get("run_id") != RUN_ID:
        raise ContractBlocked("run_id must be %r, found %r" % (RUN_ID, cfg.get("run_id")))

    for key in ("model", "model_variant", "model_source", "experiment", "run_id"):
        value = cfg.get(key)
        if isinstance(value, str) and FORBIDDEN_VARIANT_PATTERN.search(value):
            raise ContractBlocked("%s=%r names a forbidden A0/P2 variant" % (key, value))

    if cfg.get("model") not in (None, "yolo26s.pt"):
        raise ContractBlocked("the single run must initialise from yolo26s.pt, found %r" % cfg.get("model"))
    if cfg.get("model_variant") not in (None, "yolo26s"):
        raise ContractBlocked("model_variant must be plain yolo26s, found %r" % cfg.get("model_variant"))
    if cfg.get("resume") not in (None, False):
        raise ContractBlocked("resume must be false: the single run starts from the official yolo26s.pt")

    for key in ("models", "runs", "arms", "variants"):
        value = cfg.get(key)
        if isinstance(value, (list, tuple)) and len(value) > 1:
            raise ContractBlocked("exactly one model/run is allowed, found %d under %r" % (len(value), key))
    for key in ("a0", "baseline_arm", "no_augmentation_arm"):
        if cfg.get(key):
            raise ContractBlocked("an A0 control arm (%s) is explicitly out of scope for this plan" % key)

    model_block = cfg.get("model_block")
    if isinstance(model_block, dict) and model_block.get("sha256") not in (None, FROZEN_PRETRAINED_SHA256):
        raise ContractBlocked("frozen pretrained sha256 mismatch in the config")


def create_contract_map(repo_root) -> dict:
    """Read-only mapping of every frozen artifact: verified entries carry bytes + sha256."""
    root = Path(repo_root)
    targets = {
        "v2_config": V2_CONFIG_PATH,
        "v2_spec": V2_SPEC_PATH,
        "new_spec": NEW_SPEC_PATH,
        "new_config": NEW_CONFIG_PATH,
        "v2_resolved_args": V2_ARGS_PATH,
        "v2_train_manifest": V2_TRAIN_MANIFEST_PATH,
        "v2_val_manifest": V2_VAL_MANIFEST_PATH,
        "pretrained": PRETRAINED_PATH,
        "v2_checkpoint_registry": V2_CHECKPOINT_REGISTRY_PATH,
        "canonical_evaluator": CANONICAL_EVALUATOR_PATH,
        "canonical_localization_audit": CANONICAL_AUDIT_PATH,
        "training_launcher": TRAINING_LAUNCHER_PATH,
    }
    verified: dict = {}
    missing: list = []
    for name, rel in targets.items():
        path = root / rel
        if path.is_file():
            verified[name] = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            missing.append({"name": name, "path": rel})

    counts: dict = {}
    train_path = root / V2_TRAIN_MANIFEST_PATH
    if train_path.is_file():
        train_counts = load_manifest_counts(train_path)
        counts.update(train_counts)
        counts["train_rows"] = train_counts["rows"]
    val_path = root / V2_VAL_MANIFEST_PATH
    if val_path.is_file():
        val_counts = load_manifest_counts(val_path)
        counts["val_rows"] = val_counts["rows"]
        counts["val_negative_rows"] = val_counts["ordinary_negatives"] + val_counts["hard_negative_exposures"]
        counts["val_locations"] = val_counts["locations"]
        counts["val_sha256"] = val_counts["sha256"]

    # The new authoritative spec is a precondition of the plan ("Review this plan together with the amended
    # Spec first"), not a V2 artifact. It is reported separately so a missing spec never masquerades as a
    # verified V2 contract, and a verified V2 contract never hides a missing spec.
    spec_missing = [entry for entry in missing if entry["name"] == "new_spec"]
    v2_missing = [entry for entry in missing if entry["name"] != "new_spec"]

    return {
        "generated_by": "tools/clean_cp_contract.py",
        "plan": "ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md",
        "task": "Task 1: read-only repository mapping and contract freeze",
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "repo_root": str(root),
        "verified": verified,
        "missing_inputs": missing,
        "counts": counts,
        "frozen_recipe": dict(FROZEN_KNOBS),
        "v2_reference": {
            "experiment": V2_EXPERIMENT,
            "resolved_args_remote": FROZEN_V2_ARGS_REMOTE,
            "checkpoint_registry_name": "eth_real_hardneg_v2_best",
            "checkpoint_epoch": 21,
            "checkpoint_sha256": "3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7",
        },
        "gate": {
            "name": "Source contract",
            "status": "PASS" if not v2_missing else "BLOCKED_MISSING_V2_ARTIFACTS",
            "rule": "V2 weights/config/manifest/evaluator verified before Tasks 2-10 may start",
            "missing": v2_missing,
        },
        "spec_gate": {
            "name": "Authoritative spec",
            "status": "BLOCKED_MISSING_SPEC" if spec_missing else "PASS",
            "path": NEW_SPEC_PATH,
            "rule": "the plan designates this spec section as authoritative and asks for it to be reviewed "
                    "and checked in before execution; it was not supplied with the plan",
            "missing": spec_missing,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen-contract loader for CLEAN_OUTDOOR_CP_V1 (Task 1)")
    parser.add_argument("--write-map", type=Path, default=None,
                        help="write the read-only contract map JSON to this path")
    parser.add_argument("--contract", action="store_true", help="load and print the frozen V2 contract")
    parser.add_argument("--check-new-config", action="store_true",
                        help="assert the new frozen YAML reproduces the V2 recipe")
    args = parser.parse_args(argv)

    if args.contract:
        contract = load_v2_contract(
            REPO / V2_ARGS_PATH, REPO / V2_TRAIN_MANIFEST_PATH, REPO / V2_VAL_MANIFEST_PATH, REPO / PRETRAINED_PATH
        )
        print(json.dumps({"counts": contract["counts"], "pretrained": contract["pretrained"],
                          "frozen_knobs": contract["frozen_knobs"]}, indent=2, sort_keys=True))
    if args.check_new_config:
        contract = load_v2_contract(
            REPO / V2_ARGS_PATH, REPO / V2_TRAIN_MANIFEST_PATH, REPO / V2_VAL_MANIFEST_PATH, REPO / PRETRAINED_PATH
        )
        cfg = load_yaml_mapping(REPO / NEW_CONFIG_PATH)
        assert_single_model_config(cfg)
        assert_frozen_recipe(cfg["resolved_kwargs"], contract)
        print("new config PASSES the frozen V2 recipe parity check")
    if args.write_map is not None:
        mapping = create_contract_map(REPO)
        args.write_map.parent.mkdir(parents=True, exist_ok=True)
        args.write_map.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote %s (gate=%s, missing=%d)" % (args.write_map, mapping["gate"]["status"], len(mapping["missing_inputs"])))
    if not any((args.contract, args.check_new_config, args.write_map is not None)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
