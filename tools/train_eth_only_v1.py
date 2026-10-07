#!/usr/bin/env python3
"""24 GB-safe launcher for the frozen ETH-only YOLO26s 1024 baseline V1 (Task 3).

Contract (configs/eth_only_yolo26s_1024_v1.yaml + docs/superpowers/specs/2026-10-02-eth-only-yolo26s-1024-baseline-v1-design.md):
  * imgsz / nbs / optimizer / lr0 are frozen - any attempt to change them raises RecipeViolationError;
  * only the physical batch may fall back (8 -> 6 -> 4) and only on CUDA OOM; any other error propagates;
  * weights must match the pinned sha256 and byte size before training starts;
  * checkpoint selection uses the internal location-disjoint validation mAP50-95 only;
  * a smoke run is diagnostic and can never be reported as a final result.
ultralytics is imported lazily so this module (and its tests) run without torch/GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FROZEN = ("imgsz", "nbs", "optimizer", "lr0")
SELECTION_METRIC = "metrics/mAP50-95(B)"
EVALUATION_ONLY_SPLITS = ("val|eth_unseen", "external_real_only", "controlled_capability", "challenge_test")
DEFAULT_METRIC_NAME = "shuttlecock"
_OOM_PATTERN = re.compile(r"out of memory|cuda oom|outofmemoryerror|cublas_status_alloc_failed", re.I)
_OFFICIAL_MAP = {
    "imgsz": "official_imgsz", "epochs": "official_epochs", "optimizer": "official_optimizer",
    "lr0": "official_lr", "momentum": "official_momentum", "weight_decay": "official_weight_decay",
    "seed": "official_seed",
}
_LOSS_KEYS = ("box", "cls", "dfl")
_AUG_KEYS = ("mosaic", "degrees", "translate", "scale", "shear", "perspective", "copy_paste",
             "copy_paste_mode", "fliplr", "flipud", "mixup", "cutmix", "hsv_h", "hsv_s", "hsv_v",
             "deterministic")


class RecipeViolationError(RuntimeError):
    """Raised when a caller tries to change a frozen recipe knob."""


class WeightsVerificationError(RuntimeError):
    """Raised when the pretrained weights do not match the pinned sha256/byte size."""


class ContractError(RuntimeError):
    """Raised when the frozen contract or the dataset yaml is malformed."""


class TrainingFailed(RuntimeError):
    """Raised when every candidate physical batch ran out of memory."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contract(path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - remote miniconda has pyyaml
        raise ContractError("pyyaml is required to read %s: %s" % (path, exc))
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ContractError("contract %s is not a mapping" % path)
    missing = [k for k in ("experiment", "model", "model_sha256", "model_bytes", *FROZEN, "batch",
                           "batch_fallback", "selection_metric", "evaluation_only_sets") if k not in data]
    if missing:
        raise ContractError("contract %s misses required keys: %s" % (path, ", ".join(missing)))
    if list(data["never_touch"]) != list(FROZEN):
        raise ContractError("contract never_touch must be exactly %s" % (list(FROZEN),))
    return data


def resolve_recipe(contract: dict, recipe_json: dict | None = None, epochs: int | None = None,
                   deviating: dict | None = None) -> dict:
    """Build the exact ultralytics kwargs. Frozen knobs are taken from the contract and can only be
    removed by deviating from them, which is an error."""
    kwargs: dict = {}
    if recipe_json:
        for key, off_key in _OFFICIAL_MAP.items():
            if off_key in recipe_json and recipe_json[off_key] is not None:
                kwargs[key] = recipe_json[off_key]
        if kwargs.get("optimizer") == "adamw":
            kwargs["optimizer"] = "AdamW"
        for key in _LOSS_KEYS:
            if key in (recipe_json.get("official_loss") or {}):
                kwargs[key] = recipe_json["official_loss"][key]
        for key in _AUG_KEYS:
            if key in (recipe_json.get("official_augmentations") or {}):
                kwargs[key] = recipe_json["official_augmentations"][key]
    kwargs["imgsz"] = contract["imgsz"]
    kwargs["epochs"] = contract["epochs"]
    kwargs["optimizer"] = contract["optimizer"]
    kwargs["lr0"] = float(contract["lr0"])
    kwargs["nbs"] = contract["nbs"]
    kwargs["freeze"] = contract["freeze"]
    kwargs["seed"] = contract["seed"]
    if contract.get("save_period") is not None:
        kwargs["save_period"] = int(contract["save_period"])
    if epochs is not None:
        kwargs["epochs"] = int(epochs)
    for key, value in (deviating or {}).items():
        if key in FROZEN:
            raise RecipeViolationError(
                "frozen knob %r cannot be changed (contract pins %r=%r, requested %r)"
                % (key, key, contract.get(key), value))
        kwargs[key] = value
    if float(kwargs["lr0"]) != float(contract["lr0"]):
        raise RecipeViolationError("lr0 drifted to %r (contract: %r)" % (kwargs["lr0"], contract["lr0"]))
    if kwargs["optimizer"].lower() != str(contract["optimizer"]).lower():
        raise RecipeViolationError("optimizer drifted to %r" % (kwargs["optimizer"],))
    return kwargs


def plan_batch_attempts(primary: int, fallback) -> list:
    """Candidate physical batches: the primary batch, then every strictly smaller fallback, descending.

    A fallback larger than the primary is never tried (it would raise memory pressure, not relieve it)."""
    primary = int(primary)
    if primary <= 0:
        raise RecipeViolationError("batch must be positive, got %r" % primary)
    smaller = sorted({int(b) for b in fallback if 0 < int(b) < primary}, reverse=True)
    return [primary] + smaller


def is_oom(exc: BaseException) -> bool:
    return bool(_OOM_PATTERN.search("%s: %s" % (type(exc).__name__, exc)))


def run_with_fallback(attempts, train_fn):
    """Try each physical batch in order; advance only on OOM, re-raise everything else."""
    tried = []
    for batch in attempts:
        try:
            result = train_fn(batch)
        except BaseException as exc:  # noqa: BLE001 - deliberate: only OOM is swallowed
            tried.append({"batch": int(batch), "outcome": "oom" if is_oom(exc) else "error",
                          "detail": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
            if not is_oom(exc):
                raise
            continue
        tried.append({"batch": int(batch), "outcome": "ok", "detail": ""})
        return int(batch), result, tried
    raise TrainingFailed("every candidate batch ran out of memory: %s"
                         % [a["batch"] for a in tried])


def verify_weights(path, sha256: str, size: int) -> dict:
    p = Path(path)
    if not p.is_file():
        raise WeightsVerificationError("weights not found: %s" % p)
    actual_size = p.stat().st_size
    if int(size) and actual_size != int(size):
        raise WeightsVerificationError("weights size mismatch: %d != %d (%s)" % (actual_size, int(size), p))
    actual = sha256_file(p)
    if actual.lower() != str(sha256).lower():
        raise WeightsVerificationError("weights sha256 mismatch: %s != %s (%s)" % (actual, sha256, p))
    return {"path": str(p), "bytes": actual_size, "sha256": actual}


def check_dataset_yaml(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ContractError("dataset yaml not found: %s" % p)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("pyyaml is required to read %s: %s" % (path, exc))
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError("dataset yaml %s is not a mapping" % p)
    if int(data.get("nc", -1)) != 1:
        raise ContractError("dataset yaml must declare nc: 1 (got %r)" % data.get("nc"))
    names = data.get("names")
    first = names[0] if isinstance(names, list) and names else (names.get(0) if isinstance(names, dict) else None)
    if first != DEFAULT_METRIC_NAME:
        raise ContractError("dataset yaml class 0 must be %r (got %r)" % (DEFAULT_METRIC_NAME, first))
    root = Path(data.get("path") or p.parent)
    if not root.is_absolute():
        root = p.parent / root
    lists = {}
    for split in ("train", "val"):
        entry = data.get(split)
        if not entry:
            raise ContractError("dataset yaml %s has no %r list" % (p, split))
        txt = Path(entry)
        if not txt.is_absolute():
            candidate = root / txt
            txt = candidate if candidate.exists() else (p.parent / txt)
        if not txt.is_file():
            raise ContractError("dataset list %s not found for split %r" % (txt, split))
        lines = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ContractError("dataset list %s is empty" % txt)
        lists[split] = {"file": str(txt), "images": len(lines), "sha256": sha256_file(txt)}
    return {"yaml": str(p), "sha256": sha256_file(p), "nc": 1, "names": [DEFAULT_METRIC_NAME],
            "root": str(root), "lists": lists}


def metric_aliases(metric: str) -> set:
    """The contract names the metric 'mAP50-95'; ultralytics writes 'metrics/mAP50-95(B)'."""
    m = str(metric).strip()
    core = m[len("metrics/"):] if m.startswith("metrics/") else m
    if core.endswith("(B)"):
        core = core[:-3]
    return {core, core + "(B)", "metrics/" + core, "metrics/%s(B)" % core}


def select_best_epoch(results_csv, metric: str = SELECTION_METRIC) -> dict:
    """Internal-validation-only selection: argmax of the internal val metric column."""
    p = Path(results_csv)
    if not p.is_file():
        raise ContractError("results csv not found: %s" % p)
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    if not rows:
        raise ContractError("results csv %s has no epoch rows" % p)
    aliases = metric_aliases(metric)
    key = None
    for field in rows[0]:
        if field and field.strip() in aliases:
            key = field
            break
    if key is None:
        raise ContractError("results csv %s has no %r column (aliases %s; columns: %s)"
                            % (p, metric, sorted(aliases), ", ".join(k.strip() for k in rows[0] if k)))
    def value(row):
        return float(str(row[key]).strip() or "-1")
    best = max(rows, key=value)
    epoch = best.get("epoch") or best.get("                  epoch") or ""
    recall_key = next((f for f in best if f and f.strip() == "metrics/recall(B)"), None)
    return {"metric": metric, "column": key.strip(), "value": value(best),
            "epoch": int(float(str(epoch).strip() or 0)),
            "epochs_recorded": len(rows),
            "recall": (float(str(best[recall_key]).strip()) if recall_key else None),
            "selection_scope": "internal_validation_only",
            "results_csv": str(p)}


def gpu_cap_fraction(cap_gib: float, device_total_bytes: int) -> float:
    """Fraction to hand to torch.cuda.set_per_process_memory_fraction for a per-process cap."""
    if cap_gib is None or cap_gib <= 0:
        return 1.0
    if device_total_bytes <= 0:
        raise RecipeViolationError("device total memory must be positive, got %r" % device_total_bytes)
    return min(1.0, float(cap_gib) * (1024 ** 3) / float(device_total_bytes))


def cap_gpu_memory(cap_gib, device_index: int, torch_module=None) -> dict:
    """Apply the 24 GB per-process cap (plan non-negotiable) and return what was applied."""
    if cap_gib is None or float(cap_gib) <= 0:
        return {"cap_gib": None, "fraction": 1.0, "applied": False}
    torch = torch_module if torch_module is not None else __import__("torch")
    props = torch.cuda.get_device_properties(int(device_index))
    total = int(props.total_memory)
    fraction = gpu_cap_fraction(cap_gib, total)
    torch.cuda.set_per_process_memory_fraction(fraction, int(device_index))
    return {"cap_gib": float(cap_gib), "device_total_gib": round(total / 1024 ** 3, 2),
            "fraction": fraction, "applied": True}


def disable_third_party_loggers() -> dict:
    """Keep the run hermetic. ultralytics 8.4 has no wandb= argument: it auto-registers integration
    callbacks whenever the package is importable (the remote box had W&B uploading run artifacts), so the
    only reliable switch is the environment."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("COMET_MODE", "DISABLED")
    return {"WANDB_MODE": os.environ["WANDB_MODE"], "WANDB_DISABLED": os.environ["WANDB_DISABLED"],
            "COMET_MODE": os.environ["COMET_MODE"]}


def build_train_args(args, batch: int, recipe_kwargs: dict, weights: str) -> dict:
    """The exact ultralytics train kwargs (every key is validated against default.yaml by the tests)."""
    return dict(data=args.data_yaml, project=args.project, name=args.name, exist_ok=False,
                device=args.device, workers=args.workers, batch=batch, cache=args.cache,
                pretrained=True, val=True, plots=True, verbose=True,
                **recipe_kwargs)


def resolve_save_dir(project, name, model=None) -> Path:
    """The trainer's own save_dir is authoritative: ultralytics may nest the run under runs/detect/<project>.

    Reading results.csv from the wrong directory would silently lose the internal-val checkpoint selection."""
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return Path(project) / name


def diagnostic_reasons(run: dict) -> list:
    """Why a run cannot be reported as the final baseline (empty list = eligible)."""
    reasons = []
    if run.get("dry_run"):
        reasons.append("dry_run")
    if run.get("smoke"):
        reasons.append("smoke")
    epochs = (run.get("resolved_kwargs") or {}).get("epochs")
    expected = run.get("contract_epochs")
    if epochs is not None and expected is not None and int(epochs) != int(expected):
        reasons.append("epochs %s != contract %s" % (epochs, expected))
    return reasons


def build_manifest(run: dict) -> dict:
    """Assemble the run manifest; anything shortened or diagnostic is explicitly marked non-final."""
    reasons = diagnostic_reasons(run)
    diagnostic = bool(reasons)
    manifest = dict(run)
    manifest["diagnostic_reasons"] = reasons
    manifest["diagnostic_only"] = diagnostic
    manifest["eligible_for_final_report"] = not diagnostic
    manifest["checkpoint_selection"] = {"scope": "internal_validation_only",
                                        "metric": run.get("selection_metric", SELECTION_METRIC)}
    endpoint = run.get("endpoint") or {}
    if endpoint.get("mode") == "fixed_epoch":
        manifest["checkpoint_selection"]["used_for_primary_comparison"] = False
        manifest["checkpoint_selection"]["primary_endpoint"] = endpoint
    manifest["never_used_for_selection"] = list(EVALUATION_ONLY_SPLITS)
    return manifest


def write_manifest(path, manifest: dict) -> str:
    p = Path(path)
    if p.exists():
        raise ContractError("refusing to overwrite existing manifest: %s" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_endpoint(contract: dict, epochs: int, smoke: bool = False) -> dict:
    """The primary checkpoint endpoint, fixed by the contract before any metric is observed.

    This continuation reports a fixed epoch (20) rather than the best post-hoc epoch: the internal validation is an
    overlapping slice of the same training pool, so it must not be able to choose the reported checkpoint."""
    endpoint = contract.get("endpoint") or {}
    mode = endpoint.get("mode", "best_epoch_internal_val")
    if mode != "fixed_epoch":
        return {"mode": mode, "epoch": None, "diagnostic_only": bool(smoke)}
    want = int(endpoint["epoch"])
    if not smoke and int(epochs) != want:
        raise RecipeViolationError("endpoint epoch %d does not match the resolved epochs %d" % (want, epochs))
    return {"mode": "fixed_epoch", "epoch": want, "diagnostic_only": bool(smoke),
            "use_best_pt_for_primary_comparison": bool(endpoint.get("use_best_pt_for_primary_comparison", False)),
            "internal_val_selection_is_diagnostic_only":
                bool(endpoint.get("internal_val_selection_is_diagnostic_only", True))}


def optimizer_steps(train_images: int, batch: int, epochs: int, nbs: int) -> dict:
    """Gradient accumulation and the nominal optimizer-step count (provenance, spec section 27)."""
    batch = max(1, int(batch))
    nbs = max(1, int(nbs))
    accum = max(1, int(round(nbs / batch)))
    per_epoch = max(0, int(train_images) // batch)
    return {"gradient_accumulation": accum, "nominal_batch": nbs, "steps_per_epoch": per_epoch,
            "optimizer_steps": per_epoch * int(epochs)}


def run_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ETH-only YOLO26s 1024 baseline V1 trainer (24 GB safe)")
    ap.add_argument("--contract", default="configs/eth_only_yolo26s_1024_v1.yaml")
    ap.add_argument("--recipe-json", default="outputs/shuttle_capability/metrics/eth_only_v1_official_recipe.json")
    ap.add_argument("--data-yaml", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None, help="default: contract epochs (50)")
    ap.add_argument("--batch", type=int, default=None, help="default: contract batch (8)")
    ap.add_argument("--smoke", action="store_true", help="diagnostic run; never a final result")
    ap.add_argument("--smoke-epochs", type=int, default=3, help="epochs for --smoke (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="verify inputs and write the manifest, no training")
    ap.add_argument("--cache", default=False, help="ultralytics image cache (False recommended on NTFS)")
    ap.add_argument("--mem-cap-gib", type=float, default=24.0,
                    help="per-process GPU memory cap in GiB (contract: 24 on the shared A6000)")
    args = ap.parse_args(argv)

    contract = load_contract(args.contract)
    recipe_json = {}
    if args.recipe_json and Path(args.recipe_json).is_file():
        recipe_json = json.loads(Path(args.recipe_json).read_text(encoding="utf-8"))
    smoke_epochs = int(args.smoke_epochs if args.smoke_epochs is not None else 3)
    epochs = smoke_epochs if args.smoke else (args.epochs if args.epochs is not None else contract["epochs"])
    if args.smoke and args.epochs not in (None, smoke_epochs):
        raise RecipeViolationError("--smoke implies %d epochs, got %r" % (smoke_epochs, args.epochs))
    primary = int(args.batch if args.batch is not None else contract["batch"])
    attempts = plan_batch_attempts(primary, contract["batch_fallback"])
    kwargs = resolve_recipe(contract, recipe_json, epochs=epochs)
    weights = verify_weights(args.weights, contract["model_sha256"], contract["model_bytes"])
    data = check_dataset_yaml(args.data_yaml)

    run = {
        "experiment": contract["experiment"],
        "spec": contract.get("spec", ""),
        "contract": {"path": str(args.contract), "sha256": sha256_file(args.contract)},
        "weights": weights,
        "data": data,
        "resolved_kwargs": kwargs,
        "batch_attempts": attempts,
        "smoke": bool(args.smoke),
        "dry_run": bool(args.dry_run),
        "selection_metric": contract["selection_metric"],
        "contract_epochs": contract["epochs"],
        "endpoint": resolve_endpoint(contract, epochs, bool(args.smoke)),
        "contract_lr0": float(contract["lr0"]),
        "contract_nbs": int(contract["nbs"]),
        "contract_optimizer": contract["optimizer"],
        "device": args.device,
        "workers": args.workers,
        "mem_cap_gib": args.mem_cap_gib,
        "started": _now(),
        "third_party_loggers": disable_third_party_loggers(),
        "notes": ["physical batch may fall back only on CUDA OOM",
                  "imgsz/nbs/optimizer/lr0 are frozen by the contract",
                  "augmentation + loss values mirror the ETH official recipe (only batch/nbs deviate)"],
    }
    if args.dry_run:
        run["finished"] = _now()
        print(json.dumps({k: run[k] for k in ("experiment", "resolved_kwargs", "batch_attempts",
                                              "weights", "smoke", "dry_run")}, indent=1))
        print("[train] dry-run manifest -> %s" % write_manifest(args.out_manifest, build_manifest(run)))
        return 0

    from ultralytics import YOLO  # lazy: keeps this module importable without torch
    import torch

    device_index = int(str(args.device).split(",")[0])
    run["gpu_cap"] = cap_gpu_memory(args.mem_cap_gib, device_index, torch)

    def train_fn(batch):
        model = YOLO(args.weights)
        model.train(**build_train_args(args, batch, kwargs, args.weights))
        return model

    batch, model, tried = run_with_fallback(attempts, train_fn)
    run["batch_attempts_tried"] = tried
    run["batch"] = batch
    run["resolved_kwargs"]["batch"] = batch
    save_dir = resolve_save_dir(args.project, args.name, model)
    run["save_dir"] = str(save_dir)
    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        selection = select_best_epoch(results_csv, contract["selection_metric"])
        run["selection"] = selection
        best = save_dir / "weights" / "best.pt"
        run["best_checkpoint"] = {"path": str(best), "bytes": (best.stat().st_size if best.is_file() else None),
                                  "sha256": (sha256_file(best) if best.is_file() else None)}
    else:
        run["selection"] = {"error": "results.csv missing after training", "results_csv": str(results_csv)}
    run["steps"] = optimizer_steps((run.get("data", {}).get("lists", {}).get("train", {}) or {}).get("images", 0),
                                   batch, epochs, contract["nbs"])
    endpoint = run.get("endpoint") or {}
    if endpoint.get("mode") == "fixed_epoch" and endpoint.get("epoch"):
        want = int(endpoint["epoch"])
        cand = save_dir / "weights" / ("epoch%d.pt" % want)
        if not cand.is_file():
            cand = save_dir / "weights" / "last.pt"
            run["endpoint_checkpoint_note"] = ("epoch%d.pt not found (save_period=%s); last.pt recorded instead"
                                               % (want, contract.get("save_period")))
        run["primary_endpoint_checkpoint"] = {
            "epoch": want, "path": str(cand), "bytes": (cand.stat().st_size if cand.is_file() else None),
            "sha256": (sha256_file(cand) if cand.is_file() else None),
            "used_for_primary_comparison": True}
    run["finished"] = _now()
    print("[train] manifest -> %s" % write_manifest(args.out_manifest, build_manifest(run)))
    return 0


def main() -> int:  # pragma: no cover - thin wrapper
    try:
        return run_cli()
    except (RecipeViolationError, WeightsVerificationError, ContractError, TrainingFailed) as exc:
        print("[train] FAILED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
