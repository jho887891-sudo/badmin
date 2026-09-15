#!/usr/bin/env python3
"""Train the nc=1 shuttlecock baseline reproducibly (plan Task 3).

Usage:
    python scripts/shuttle_detection/train_baseline.py \
        --config configs/shuttle_detection/baseline.yaml \
        --train-manifest manifest_train_synthetic.csv \
        --val-manifest manifest_val_synthetic.csv \
        --data-root /home/T7/dgut/robot_sim \
        --run-name baseline_synthetic_only

Why this script exists: Ultralytics will happily train from any paths it is handed.
This wrapper is the only place that decides what a baseline run is allowed to be --
it loads the validated baseline config, converts the audited manifests into a
single-class dataset configuration (which refuses any fixed_core_test or
challenge_test sample), and writes the run identity before and after training so the
record survives a crash.

The run directory is <data-root>/outputs/shuttle_detection/training/<run-name> and
holds resolved_config.json, dataset/, run_metadata.json and, after training, the
Ultralytics artifacts including weights/best.pt and weights/last.pt.

Exit codes:
    0  training finished and the metadata was written
    1  training started but failed (the failure is recorded in run_metadata.json)
    2  the run could not be prepared (bad config, missing manifest, held-out sample)

Data root: --data-root is required and must exist. It is the base for relative inputs
and the parent of outputs/shuttle_detection/training. It is never inferred from git:
the remote training host holds the project at /home/T7/dgut/robot_sim, which is
deliberately not a git checkout, so the plan's
--data-root "$(git rev-parse --show-toplevel)" cannot be substituted there.

Generation host: the dataset configuration contains absolute image paths of the
machine that generated it, so this script belongs on the training host. The metadata
records the generating host, and the run summary prints it, for that reason.

Ultralytics is imported inside train() only, so preparing and inspecting a run works
on a machine that has neither Ultralytics nor a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.dataset_audit import (  # noqa: E402
    ManifestError,
    read_manifest,
)
from src.perception.shuttle_detection.run_metadata import (  # noqa: E402
    METADATA_FILENAME,
    data_version_from_manifests,
    detect_gpu,
    make_metadata,
    write_metadata,
)
from src.perception.shuttle_detection.training_config import (  # noqa: E402
    BaselineConfig,
    ConfigError,
)
from src.perception.shuttle_detection.ultralytics_dataset import (  # noqa: E402
    DatasetConfigError,
    write_dataset_yaml,
)

EXIT_OK = 0
EXIT_TRAINING_FAILED = 1
EXIT_UNUSABLE_INPUT = 2

RESOLVED_CONFIG_NAME = "resolved_config.json"
DATASET_DIR_NAME = "dataset"
WEIGHTS_DIR_NAME = "weights"
OUTPUT_SUBPATH: tuple[str, ...] = ("outputs", "shuttle_detection", "training")


class RunPreparationError(RuntimeError):
    """The run cannot be prepared from the arguments it was given."""


@dataclass(frozen=True)
class PreparedRun:
    """Everything a run needs before the first weight is touched."""

    run_name: str
    data_root: Path
    run_dir: Path
    dataset_dir: Path
    dataset_yaml: Path
    train_manifest: Path
    val_manifest: Path
    config: BaselineConfig
    resolved_config_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    host: dict[str, Any]
    metadata_arguments: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the nc=1 shuttlecock baseline from audited manifests."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="baseline configuration (configs/shuttle_detection/baseline.yaml)",
    )
    parser.add_argument(
        "--train-manifest",
        required=True,
        metavar="CSV",
        help="audited manifest of the training pool",
    )
    parser.add_argument(
        "--val-manifest",
        required=True,
        metavar="CSV",
        help="audited manifest of the validation pool",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="run directory name under the training output root",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        metavar="DIR",
        help=(
            "project data root; required and must exist, and never inferred from git "
            "because the remote training host is deliberately not a git checkout"
        ),
    )
    parser.add_argument(
        "--out-root",
        default=None,
        metavar="DIR",
        help="override the training output root (default <data-root>/outputs/shuttle_detection/training)",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_out_root(data_root: Path | str) -> Path:
    return Path(data_root).joinpath(*OUTPUT_SUBPATH)


def resolve_input(path: Path | str, data_root: Path | str) -> Path:
    """Resolve a CLI path against the working directory, then against the data root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.is_file():
        return candidate.resolve()
    return Path(data_root) / candidate


def _check_run_name(run_name: str) -> str:
    name = str(run_name).strip()
    if not name:
        raise RunPreparationError("run name must not be empty")
    if any(separator in name for separator in ("/", chr(92))) or name in (".", ".."):
        raise RunPreparationError(
            "run name must be a plain directory name, got " + repr(run_name)
        )
    return name


def _check_data_root(data_root: Path | str) -> Path:
    """The data root is required, must exist, and is never inferred from git.

    The remote training host holds the project at /home/T7/dgut/robot_sim, which is
    deliberately not a git checkout, so a git-derived default could not work there.
    A wrong root would otherwise surface only after training, when the absolute image
    paths in dataset.yaml turn out to name files that do not exist.
    """
    if data_root is None or not str(data_root).strip():
        raise RunPreparationError(
            "--data-root is required: it is the base for relative inputs and the parent "
            "of the training output root"
        )
    root = Path(data_root)
    if not root.exists():
        raise RunPreparationError("data root does not exist: " + str(root))
    if not root.is_dir():
        raise RunPreparationError("data root is not a directory: " + str(root))
    return root.resolve()


def source_counts(manifests: Sequence[Path | str]) -> dict[str, int]:
    """Count manifest rows per source_type, for the recorded sampling ratio.

    A plain image list has no source column; it then contributes nothing and the
    ratio is recorded as unknown rather than guessed.
    """
    counts: dict[str, int] = {}
    for path in manifests:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            continue
        try:
            rows, columns = read_manifest(manifest_path)
        except ManifestError:
            continue
        if "source_type" not in {str(column).strip() for column in columns}:
            continue
        for row in rows:
            key = str(row.get("source_type", "") or "").strip() or "UNKNOWN"
            counts[key] = counts.get(key, 0) + 1
    return counts


def prepare_run(
    *,
    config_path: Path | str,
    train_manifest: Path | str,
    val_manifest: Path | str,
    run_name: str,
    data_root: Path | str,
    out_root: Path | str | None = None,
) -> PreparedRun:
    """Validate the run, write its dataset configuration and its opening record.

    Raises ConfigError, DatasetConfigError or ManifestError when the run must not
    start; nothing is written in that case beyond the directories already created.
    """
    name = _check_run_name(run_name)
    root = _check_data_root(data_root)
    config_file = resolve_input(config_path, root)
    train_file = resolve_input(train_manifest, root)
    val_file = resolve_input(val_manifest, root)

    config = BaselineConfig.load(config_file)

    run_dir = (Path(out_root).resolve() if out_root is not None else default_out_root(root)) / name
    dataset_dir = run_dir / DATASET_DIR_NAME
    dataset_yaml = write_dataset_yaml(
        dataset_dir, train_file, val_file, data_root=root
    )

    resolved = {
        "run_name": name,
        "data_root": str(root),
        "config_path": str(config_file),
        "dataset_yaml": str(dataset_yaml),
        "train_manifest": str(train_file),
        "val_manifest": str(val_file),
        **config.as_dict(),
    }
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    resolved_path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifests = [train_file, val_file]
    arguments: dict[str, Any] = {
        "run_name": name,
        "seed": config.seed,
        "data_version": data_version_from_manifests(manifests),
        "model": config.model,
        "model_structure": config.model,
        "init_weights": config.model,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "epochs": config.epochs,
        "source_counts": source_counts(manifests),
        "manifests": manifests,
        "config_path": config_file,
        "config": config.as_dict(),
        "data_root": root,
    }
    metadata = make_metadata(**arguments, status="prepared")
    metadata_path = write_metadata(run_dir / METADATA_FILENAME, metadata)

    return PreparedRun(
        run_name=name,
        data_root=root,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        dataset_yaml=dataset_yaml,
        train_manifest=train_file,
        val_manifest=val_file,
        config=config,
        resolved_config_path=resolved_path,
        metadata_path=metadata_path,
        metadata=metadata,
        host=metadata["host"],
        metadata_arguments=arguments,
    )


def generation_host_notice(prepared: PreparedRun) -> str:
    """One line naming the host whose absolute paths this run configuration holds.

    dataset.yaml and the image lists are written with absolute paths, so a run is only
    valid on the machine that generated it; whoever inspects the run later has to be
    told which machine that was.
    """
    return (
        "dataset configuration generated on host "
        + str(prepared.host.get("hostname"))
        + " at "
        + str(prepared.dataset_dir)
        + "; its absolute paths are only valid on that host"
    )


def _reset_peak_vram() -> None:
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_vram_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    except Exception:
        return None


def train(prepared: PreparedRun) -> dict[str, Any]:
    """Run the plan's Ultralytics training call and return the closing record fields.

    Ultralytics is imported here, not at module import, so a machine without it can
    still prepare, inspect and test a run.

    exist_ok=True is passed because the run directory already exists: it holds the
    dataset configuration and the opening metadata, and Ultralytics would otherwise
    train into a second directory named <run-name>2 and split the run in two.
    """
    from ultralytics import YOLO  # noqa: PLC0415 - training-only dependency

    config = prepared.config
    started_at = _utc_now()
    started = time.time()
    _reset_peak_vram()

    model = YOLO(config.model)
    model.train(
        data=str(prepared.dataset_yaml),
        imgsz=config.imgsz,
        epochs=config.epochs,
        batch=config.batch,
        seed=config.seed,
        optimizer=config.optimizer,
        lr0=config.lr0,
        project=str(prepared.run_dir.parent),
        name=prepared.run_name,
        exist_ok=True,
    )

    weights_dir = prepared.run_dir / WEIGHTS_DIR_NAME
    return {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_time_sec": round(time.time() - started, 3),
        "peak_vram_mb": _peak_vram_mb(),
        "gpu": detect_gpu(),
        "best_weight": weights_dir / "best.pt",
        "last_weight": weights_dir / "last.pt",
        "extras": {
            "weights_dir": str(weights_dir),
            "results_csv": str(prepared.run_dir / "results.csv"),
            "args_yaml": str(prepared.run_dir / "args.yaml"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = prepare_run(
            config_path=args.config,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            run_name=args.run_name,
            data_root=args.data_root,
            out_root=args.out_root,
        )
    except (
        RunPreparationError,
        ConfigError,
        DatasetConfigError,
        ManifestError,
        FileNotFoundError,
        OSError,
    ) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    print("train_baseline: prepared " + prepared.run_name + " in " + str(prepared.run_dir))
    print("  data " + str(prepared.dataset_yaml))
    print("  " + generation_host_notice(prepared))

    try:
        finished = train(prepared)
    except Exception as error:  # training failures must leave a record behind
        print("ERROR: training failed: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        write_metadata(
            prepared.metadata_path,
            make_metadata(
                **prepared.metadata_arguments,
                status="failed",
                extras={"error": type(error).__name__ + ": " + str(error)},
            ),
        )
        return EXIT_TRAINING_FAILED

    metadata = make_metadata(**prepared.metadata_arguments, **finished, status="completed")
    write_metadata(prepared.metadata_path, metadata)

    print("  best " + str(metadata["best_weight"]))
    print("  last " + str(metadata["last_weight"]))
    print("  metadata " + str(prepared.metadata_path))
    print(
        "train_baseline: completed run "
        + prepared.run_name
        + " in "
        + str(metadata["wall_time_sec"])
        + "s"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())