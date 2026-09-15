"""Immutable run metadata for a reproducible shuttlecock baseline training run.

Why this module exists: spec section 7 lists what every training run must record --
model structure, initialisation weights, data version, synthetic/real sampling ratio,
input resolution, batch size, optimizer, learning rate, epochs, augmentation, random
seed, code commit, GPU, wall time, peak VRAM, best weight and last weight. A run that
omits one of them cannot be reproduced or compared with the next run, and the omission
is usually discovered long after the process is gone. Collecting the record in one
place, before and after training, makes that list a contract instead of a habit.

This module deliberately imports neither Ultralytics nor a GPU library at module level:
a machine without CUDA (this workstation) must still be able to build and test the
record. Only the CLI training path imports Ultralytics.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_run_metadata.py -v
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_audit import ManifestError, canonical_source_family, read_manifest

METADATA_FILENAME = "run_metadata.json"
METADATA_SCHEMA_VERSION = 1

# Every record spec section 7 requires. Kept as a tuple so a test can assert that a
# produced record really carries all of them, rather than trusting the writer.
SPEC_REQUIRED_FIELDS: tuple[str, ...] = (
    "model_structure",
    "init_weights",
    "data_version",
    "sampling_ratio",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "epochs",
    "augmentation",
    "seed",
    "code_commit",
    "gpu",
    "wall_time_sec",
    "peak_vram_mb",
    "best_weight",
    "last_weight",
)

# Ultralytics YOLO training augment defaults. The baseline changes none of them, but a
# run must still say so explicitly: otherwise a changed augment could never be spotted
# when two runs are compared.
DEFAULT_AUGMENTATION: dict[str, Any] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "close_mosaic": 10,
}

# Sampling-ratio buckets. "negative" is tracked separately because spec section 6
# requires negative samples as their own content class, and "other" exists so an
# unrecognised source family cannot silently vanish from the totals.
RATIO_BUCKETS: tuple[str, ...] = ("synthetic", "real", "negative", "other")
_BUCKET_BY_SOURCE_FAMILY: dict[str, str] = {
    "SYNTHETIC_3D": "synthetic",
    "REAL_IMAGE": "real",
    "REAL_VIDEO": "real",
    "NEGATIVE": "negative",
}

HASH_CHUNK = 1024 * 1024

# src/perception/shuttle_detection/run_metadata.py -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_git(arguments: Sequence[str], repository_root: Path | str | None = None) -> str | None:
    """Run one read-only git command, returning None when git cannot answer."""
    root = Path(repository_root) if repository_root is not None else REPOSITORY_ROOT
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def detect_code_commit(repository_root: Path | str | None = None) -> str | None:
    """The commit a run was launched from, or None outside a git checkout."""
    return _run_git(["rev-parse", "HEAD"], repository_root)


def detect_code_dirty(repository_root: Path | str | None = None) -> bool | None:
    """Whether the checkout had uncommitted changes when the run started."""
    status = _run_git(["status", "--porcelain"], repository_root)
    if status is None:
        # An empty status is a clean tree, so distinguish it from "git failed".
        return False if _run_git(["rev-parse", "HEAD"], repository_root) else None
    return bool(status.strip())


def detect_gpu() -> dict[str, Any]:
    """Describe the training GPU without requiring one to be present.

    A missing GPU is recorded as unavailable rather than raised: the metadata layer
    has to run on the workstation that prepares and reviews runs, not only on the
    machine that trains them.
    """
    info: dict[str, Any] = {
        "available": False,
        "name": None,
        "device_count": 0,
        "source": "none",
    }
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except Exception:
        return info
    info["source"] = "torch"
    try:
        if not torch.cuda.is_available():
            return info
        info["available"] = True
        info["device_count"] = int(torch.cuda.device_count())
        info["name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        return info
    return info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(path: Path | str) -> dict[str, Any]:
    """Identify one manifest by content, size and row count.

    A missing manifest is recorded as missing instead of raising: the metadata is
    written for forensics, and refusing to write it because one input is gone would
    destroy the very record that explains what happened.
    """
    manifest_path = Path(path)
    entry: dict[str, Any] = {
        "name": manifest_path.name,
        "path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "sha256": None,
        "bytes": None,
        "rows": None,
    }
    if not entry["exists"]:
        return entry
    entry["sha256"] = _sha256_file(manifest_path)
    entry["bytes"] = manifest_path.stat().st_size
    try:
        rows, _columns = read_manifest(manifest_path)
    except ManifestError:
        return entry
    entry["rows"] = len(rows)
    return entry


def data_version_from_manifests(manifests: Iterable[Path | str]) -> str:
    """Identity of the exact manifest set a run trained on.

    Order-independent, like the audit's own dataset version: the same two manifests
    describe the same data no matter which one is named as train or as val.
    """
    payload = "\n".join(
        sorted(
            str(entry["name"]) + ":" + str(entry["sha256"])
            for entry in (manifest_fingerprint(path) for path in manifests)
        )
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sampling_ratio_from_counts(counts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Turn source-type counts into the explicit synthetic / real sampling ratio.

    Returns None fractions when nothing was counted: "the mix was not recorded" and
    "this run used no real data" are different claims and must not look alike.
    """
    buckets = {key: 0 for key in RATIO_BUCKETS}
    for source_type, count in (counts or {}).items():
        family = canonical_source_family(str(source_type)) or ""
        buckets[_BUCKET_BY_SOURCE_FAMILY.get(family, "other")] += int(count)
    total = sum(buckets.values())
    ratio: dict[str, Any] = {key: None for key in RATIO_BUCKETS}
    if total > 0:
        for key, value in buckets.items():
            ratio[key] = round(value / total, 6)
    ratio["total"] = total
    ratio["counts"] = {str(key): int(value) for key, value in (counts or {}).items()}
    return ratio


def _normalise_sampling_ratio(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return sampling_ratio_from_counts(None)
    if any(key in value for key in RATIO_BUCKETS):
        ratio = {
            key: (None if value.get(key) is None else float(value[key]))
            for key in RATIO_BUCKETS
        }
        ratio["total"] = value.get("total")
        ratio["counts"] = {str(k): int(v) for k, v in dict(value.get("counts", {})).items()}
        return ratio
    return sampling_ratio_from_counts(value)


def augmentation_from(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The augment settings a run used: the Ultralytics defaults plus any override."""
    settings = dict(DEFAULT_AUGMENTATION)
    for key, value in (overrides or {}).items():
        settings[str(key)] = value
    return settings


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _config_sha256(config_path: Path | str | None) -> str | None:
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return _sha256_file(path)


def make_metadata(
    *,
    seed: int,
    data_version: str,
    model: str,
    run_name: str | None = None,
    init_weights: str | Path | None = None,
    model_structure: str | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    epochs: int | None = None,
    augmentation: Mapping[str, Any] | None = None,
    sampling_ratio: Mapping[str, Any] | None = None,
    source_counts: Mapping[str, Any] | None = None,
    manifests: Sequence[Path | str] | None = None,
    config_path: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    data_root: Path | str | None = None,
    gpu: Mapping[str, Any] | str | None = None,
    code_commit: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    wall_time_sec: float | None = None,
    peak_vram_mb: float | None = None,
    best_weight: Path | str | None = None,
    last_weight: Path | str | None = None,
    status: str = "prepared",
    created_at: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata record of one baseline run.

    Every field spec section 7 requires is always present, even when the run has not
    produced it yet: an absent wall time is recorded as null rather than omitted, so a
    reviewer can tell "not measured yet" from "the writer forgot".

    The record is JSON-serialisable and deterministic for the same inputs and the same
    environment, which is what makes two runs comparable at all.
    """
    commit = code_commit if code_commit is not None else detect_code_commit()
    if gpu is None:
        gpu_info = detect_gpu()
    elif isinstance(gpu, str):
        gpu_info = {"available": True, "name": gpu, "device_count": 1, "source": "given"}
    else:
        gpu_info = dict(gpu)
        gpu_info.setdefault("available", gpu_info.get("name") is not None)
        gpu_info.setdefault("device_count", 0)
        gpu_info.setdefault("source", "given")

    if sampling_ratio is None and source_counts is not None:
        ratio = sampling_ratio_from_counts(source_counts)
    else:
        ratio = _normalise_sampling_ratio(sampling_ratio)

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "status": status,
        "run_name": run_name,
        "created_at": created_at if created_at is not None else _utc_now(),
        "started_at": started_at,
        "finished_at": finished_at,
        "code_commit": commit,
        "code_dirty": detect_code_dirty() if code_commit is None else None,
        "model": model,
        "model_structure": model_structure if model_structure is not None else model,
        "init_weights": _text_or_none(init_weights) if init_weights is not None else model,
        "data_version": str(data_version),
        "data_root": _text_or_none(data_root),
        "manifests": [manifest_fingerprint(path) for path in (manifests or [])],
        "sampling_ratio": ratio,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "lr0": lr0,
        "epochs": epochs,
        "augmentation": augmentation_from(augmentation),
        "seed": int(seed),
        "gpu": gpu_info,
        "wall_time_sec": wall_time_sec,
        "peak_vram_mb": peak_vram_mb,
        "best_weight": _text_or_none(best_weight),
        "last_weight": _text_or_none(last_weight),
        "config_path": _text_or_none(config_path),
        "config_sha256": _config_sha256(config_path),
        "config": None if config is None else dict(config),
        "extra": dict(extras or {}),
    }


def write_metadata(path: Path | str, metadata: Mapping[str, Any]) -> Path:
    """Write the record as UTF-8 JSON and return the path that was written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return destination
