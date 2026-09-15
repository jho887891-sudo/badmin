#!/usr/bin/env python3
"""Run metadata for a reproducible shuttlecock baseline training run.

Why this module exists: spec section 7 lists seventeen things every training run
must record -- model structure, initialisation weights, data version, sampling
ratio, resolution, batch, optimizer, learning rate, epochs, augmentation, seed,
code commit, GPU, wall time, peak VRAM, best weight, last weight. A run whose
metadata misses one of them cannot be reproduced or compared with the next run, and
the omission is usually discovered months later. These tests pin all seventeen, and
pin the two environment facts that make the module usable on this workstation: it
must work without Ultralytics and without a GPU.

Run:
    python tests/perception/shuttle_detection/test_run_metadata.py -v
    pytest tests/perception/shuttle_detection/test_run_metadata.py -v
"""
from __future__ import annotations

import csv
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.run_metadata import (  # noqa: E402
    METADATA_FILENAME,
    SPEC_REQUIRED_FIELDS,
    make_metadata,
    sampling_ratio_from_counts,
    write_metadata,
)
from src.perception.shuttle_detection.training_config import (  # noqa: E402
    BASELINE_CONFIG_PATH,
)

BASELINE_CONFIG = "configs/shuttle_detection/baseline.yaml"
TRAIN_SCRIPT = "scripts/shuttle_detection/train_baseline.py"
MANIFEST_COLUMNS = ("file", "split", "source_type", "background", "is_negative")


def make_image(root: Path, split: str, name: str) -> Path:
    images = root / split / "images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / name
    image.write_bytes(b"not-a-real-jpeg")
    labels = root / split / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / (Path(name).stem + ".txt")).write_text("0 0.5 0.5 0.02 0.02", encoding="utf-8")
    return image


def write_manifest(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), restval="")
        writer.writeheader()
        for record in rows:
            writer.writerow(record)
    return path


def manifest_row(name: str, split: str, source_type: str = "SYNTHETIC_3D") -> dict:
    return {
        "file": name,
        "split": split,
        "source_type": source_type,
        "background": "tbg_001.jpg",
        "is_negative": "False",
    }


def build_dataset(root: Path, train_split: str = "train"):
    """A tiny train/val pair used by the trainer tests."""
    make_image(root, "train", "train_000.jpg")
    make_image(root, "val", "val_000.jpg")
    train_manifest = write_manifest(
        root / "manifest_train.csv", [manifest_row("train_000.jpg", train_split)]
    )
    val_manifest = write_manifest(root / "manifest_val.csv", [manifest_row("val_000.jpg", "val")])
    return train_manifest, val_manifest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, TRAIN_SCRIPT, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_metadata_contains_commit_and_seed():
    """Plan step 1: the three facts a run identity cannot do without."""
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert metadata["seed"] == 17
    assert "code_commit" in metadata
    assert metadata["model"] == "yolo26s.pt"


def test_metadata_records_every_field_spec_7_requires():
    """Spec section 7: seventeen required records, none of them optional."""
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    missing = [field for field in SPEC_REQUIRED_FIELDS if field not in metadata]
    assert missing == []


def test_metadata_records_the_resolved_training_parameters():
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        imgsz=640,
        batch=16,
        optimizer="auto",
        lr0=0.01,
        epochs=100,
    )
    assert (metadata["imgsz"], metadata["batch"]) == (640, 16)
    assert (metadata["optimizer"], metadata["lr0"]) == ("auto", 0.01)
    assert metadata["epochs"] == 100
    assert metadata["init_weights"] == "yolo26s.pt"
    assert metadata["model_structure"] == "yolo26s.pt"


def test_metadata_is_json_serialisable_and_writable(tmp_path):
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    path = write_metadata(tmp_path / "run" / METADATA_FILENAME, metadata)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(metadata))


def test_make_metadata_needs_neither_ultralytics_nor_a_gpu():
    """The metadata layer must work on this workstation, which has neither."""
    metadata = make_metadata(seed=3, data_version="v0", model="yolo26n.pt")
    assert "ultralytics" not in sys.modules
    assert isinstance(metadata["gpu"]["available"], bool)
    assert metadata["gpu"]["name"] is None or isinstance(metadata["gpu"]["name"], str)


def test_augmentation_defaults_are_recorded():
    """An empty augmentation record would hide a changed training recipe."""
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert metadata["augmentation"]["mosaic"] == 1.0
    assert metadata["augmentation"]["scale"] == 0.5
    assert metadata["augmentation"]["close_mosaic"] == 10


def test_augmentation_override_keeps_the_other_defaults():
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        augmentation={"mosaic": 0.0},
    )
    assert metadata["augmentation"]["mosaic"] == 0.0
    assert metadata["augmentation"]["fliplr"] == 0.5


def test_sampling_ratio_from_source_counts():
    """Spec section 5.2: the synthetic / real sampling ratio is explicit."""
    ratio = sampling_ratio_from_counts({"SYNTHETIC_HIFI_3D": 300, "REAL_IMAGE": 100})
    assert ratio["synthetic"] == 0.75
    assert ratio["real"] == 0.25
    assert ratio["negative"] == 0.0
    assert ratio["total"] == 400


def test_sampling_ratio_is_unknown_without_counts():
    """An unknown ratio must read as unknown, not as "zero real data".

    The two are different claims: one says the mix was not recorded, the other says
    the run was synthetic-only.
    """
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert metadata["sampling_ratio"]["synthetic"] is None
    assert metadata["sampling_ratio"]["real"] is None


def test_sampling_ratio_accepts_a_ready_made_mapping():
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        sampling_ratio={"synthetic": 1.0, "real": 0.0},
    )
    assert metadata["sampling_ratio"]["synthetic"] == 1.0
    assert metadata["sampling_ratio"]["real"] == 0.0


def test_source_counts_are_converted_to_a_sampling_ratio():
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        source_counts={"SYNTHETIC_3D": 3, "REAL_VIDEO": 1},
    )
    assert metadata["sampling_ratio"]["synthetic"] == 0.75
    assert metadata["sampling_ratio"]["real"] == 0.25


def test_manifest_fingerprints_are_recorded(tmp_path):
    """A data version is only meaningful with the manifest hashes beside it."""
    train_manifest, val_manifest = build_dataset(tmp_path)
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        manifests=[train_manifest, val_manifest],
    )
    recorded = {entry["name"]: entry for entry in metadata["manifests"]}
    assert set(recorded) == {"manifest_train.csv", "manifest_val.csv"}
    assert all(len(entry["sha256"]) == 64 for entry in recorded.values())
    assert all(entry["rows"] == 1 for entry in recorded.values())


def test_a_missing_manifest_is_recorded_as_missing(tmp_path):
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        manifests=[tmp_path / "absent.csv"],
    )
    assert metadata["manifests"][0]["sha256"] is None
    assert metadata["manifests"][0]["exists"] is False


def test_best_and_last_weights_are_recorded(tmp_path):
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        best_weight=tmp_path / "weights" / "best.pt",
        last_weight=tmp_path / "weights" / "last.pt",
        wall_time_sec=1234.5,
        peak_vram_mb=8123.0,
        status="completed",
    )
    assert metadata["best_weight"].endswith("best.pt")
    assert metadata["last_weight"].endswith("last.pt")
    assert metadata["wall_time_sec"] == 1234.5
    assert metadata["peak_vram_mb"] == 8123.0
    assert metadata["status"] == "completed"


def test_code_commit_is_the_current_git_head():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert metadata["code_commit"] == head


def test_metadata_is_deterministic_for_the_same_inputs():
    """Two runs with the same inputs must produce the same record, field for field."""
    arguments = dict(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        created_at="2026-09-15T00:00:00Z",
        code_commit="0" * 40,
        gpu={"available": False, "name": None, "device_count": 0},
    )
    assert make_metadata(**arguments) == make_metadata(**arguments)


def test_extra_notes_are_kept_next_to_the_schema():
    metadata = make_metadata(
        seed=17,
        data_version="v1",
        model="yolo26s.pt",
        extras={"freeze": "level-0"},
    )
    assert metadata["extra"]["freeze"] == "level-0"


def test_cli_module_can_be_imported_without_ultralytics():
    """Only the training call may import Ultralytics, never the module import."""
    from scripts.shuttle_detection import train_baseline

    assert "ultralytics" not in sys.modules
    assert train_baseline.parse_args([
        "--config", BASELINE_CONFIG,
        "--train-manifest", "a.csv",
        "--val-manifest", "b.csv",
        "--run-name", "r",
        "--data-root", ".",
    ]).run_name == "r"


def test_cli_refuses_a_held_out_manifest_before_training(tmp_path):
    """A held-out row must stop the run with a non-zero exit, not a warning."""
    train_manifest, val_manifest = build_dataset(tmp_path, train_split="challenge_test")
    completed = run_cli(
        "--config", BASELINE_CONFIG,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--run-name", "refused",
        "--data-root", str(tmp_path),
        "--out-root", str(tmp_path / "out"),
    )
    assert completed.returncode != 0
    assert "challenge_test" in (completed.stdout + completed.stderr)


def test_cli_prepares_the_run_without_training(tmp_path):
    """Everything before the training call is testable on a machine without Ultralytics."""
    from scripts.shuttle_detection import train_baseline

    train_manifest, val_manifest = build_dataset(tmp_path)
    prepared = train_baseline.prepare_run(
        config_path=BASELINE_CONFIG_PATH,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        run_name="baseline_synthetic_only",
        data_root=tmp_path,
        out_root=tmp_path / "out",
    )
    assert "ultralytics" not in sys.modules
    assert prepared.dataset_yaml.is_file()
    resolved = json.loads(prepared.resolved_config_path.read_text(encoding="utf-8"))
    assert resolved["nc"] == 1
    assert resolved["model"] == "yolo26s.pt"
    assert resolved["lr0"] == 0.01
    metadata = json.loads(prepared.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "prepared"
    assert metadata["seed"] == 17
    assert metadata["data_version"].startswith("sha256:")
    assert len(metadata["config_sha256"]) == 64
    assert [entry["name"] for entry in metadata["manifests"]]
    assert metadata["run_name"] == "baseline_synthetic_only"


def test_cli_run_directory_follows_the_plan_layout(tmp_path):
    from scripts.shuttle_detection import train_baseline

    train_manifest, val_manifest = build_dataset(tmp_path)
    prepared = train_baseline.prepare_run(
        config_path=BASELINE_CONFIG_PATH,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        run_name="baseline_synthetic_only",
        data_root=tmp_path,
    )
    expected = tmp_path / "outputs" / "shuttle_detection" / "training" / "baseline_synthetic_only"
    assert prepared.run_dir == expected
    assert prepared.metadata_path == expected / METADATA_FILENAME

def test_metadata_records_the_generating_host():
    """Absolute paths inside a run are only valid on the machine that wrote them."""
    metadata = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert metadata["host"]["hostname"] == socket.gethostname()
    assert metadata["host"]["platform"]
    assert metadata["host"]["python"]


def test_cli_states_the_host_the_run_was_generated_on(tmp_path):
    """The operator has to be told which machine the absolute paths belong to."""
    from scripts.shuttle_detection import train_baseline

    train_manifest, val_manifest = build_dataset(tmp_path)
    prepared = train_baseline.prepare_run(
        config_path=BASELINE_CONFIG_PATH,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        run_name="host_notice",
        data_root=tmp_path,
        out_root=tmp_path / "out",
    )
    notice = train_baseline.generation_host_notice(prepared)
    assert socket.gethostname() in notice
    assert str(prepared.dataset_dir) in notice
    assert "only valid" in notice


def test_cli_requires_an_explicit_data_root(tmp_path):
    """The remote project directory is not a git checkout, so the plan's

    --data-root "$(git rev-parse --show-toplevel)" cannot be substituted there. The
    data root is therefore required and is never inferred.
    """
    train_manifest, val_manifest = build_dataset(tmp_path)
    completed = run_cli(
        "--config", BASELINE_CONFIG,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--run-name", "no_data_root",
        "--out-root", str(tmp_path / "out"),
    )
    assert completed.returncode == 2
    assert "--data-root" in completed.stderr


def test_prepare_run_rejects_a_data_root_that_does_not_exist(tmp_path):
    """A wrong data root would silently produce unusable absolute image paths."""
    from scripts.shuttle_detection.train_baseline import RunPreparationError, prepare_run

    train_manifest, val_manifest = build_dataset(tmp_path)
    with pytest.raises(RunPreparationError) as error:
        prepare_run(
            config_path=BASELINE_CONFIG_PATH,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            run_name="absent_root",
            data_root=tmp_path / "absent",
            out_root=tmp_path / "out",
        )
    assert "data root" in str(error.value)
