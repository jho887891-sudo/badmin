#!/usr/bin/env python3
"""Runtime Ultralytics dataset configuration built from audited manifests.

Why this module exists: Ultralytics trains from an image list plus a dataset YAML,
but what may be trained is decided by manifests whose rows carry a split. Those two
views are easy to disagree, and a single fixed_core_test or challenge_test row that
leaks into a train list silently destroys the independence of the frozen test set --
Ultralytics would never notice. These tests pin the single-class YAML shape and,
above all, the refusal to build a dataset configuration from a held-out pool.

Run:
    python tests/perception/shuttle_detection/test_ultralytics_dataset.py -v
    pytest tests/perception/shuttle_detection/test_ultralytics_dataset.py -v
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.dataset_audit import (  # noqa: E402
    ManifestError,
    ManifestMissingError,
)
from src.perception.shuttle_detection.ultralytics_dataset import (  # noqa: E402
    CLASS_NAMES,
    DATASET_YAML_NAME,
    DatasetConfigError,
    ForbiddenSplitError,
    SplitRoleMismatchError,
    UnknownSplitError,
    write_dataset_yaml,
)

MANIFEST_COLUMNS = ("file", "split", "source_type", "background", "is_negative")


def make_image(root: Path, split: str, name: str) -> Path:
    """Create a placeholder image and its YOLO label in the frozen layout."""
    images = root / split / "images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / name
    image.write_bytes(b"not-a-real-jpeg")
    labels = root / split / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / (Path(name).stem + ".txt")).write_text("0 0.5 0.5 0.02 0.02", encoding="utf-8")
    return image


def write_manifest(path: Path, rows, columns=MANIFEST_COLUMNS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        for record in rows:
            writer.writerow(record)
    return path


def row(name: str, split: str, **extra: str) -> dict:
    base = {
        "file": name,
        "split": split,
        "source_type": "SYNTHETIC_3D",
        "background": "tbg_001.jpg",
        "is_negative": "False",
    }
    base.update(extra)
    return base


def build_dataset(root: Path):
    """One train row, one val row, both with an image on disk."""
    train_image = make_image(root, "train", "train_000.jpg")
    val_image = make_image(root, "val", "val_000.jpg")
    train_manifest = write_manifest(root / "manifest_train.csv", [row("train_000.jpg", "train")])
    val_manifest = write_manifest(root / "manifest_val.csv", [row("val_000.jpg", "val")])
    return train_manifest, val_manifest, train_image, val_image


def list_lines(path: Path) -> list:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dataset_yaml_is_single_class(tmp_path):
    """Plan step 1: the generated dataset configuration is the nc=1 task."""
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    dataset_yaml = write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)
    assert dataset_yaml.name == DATASET_YAML_NAME
    text = dataset_yaml.read_text(encoding="utf-8")
    assert "nc: 1" in text
    assert "shuttlecock" in text
    document = yaml.safe_load(text)
    assert document["nc"] == 1
    assert document["names"] == CLASS_NAMES


def test_image_lists_are_written_next_to_the_yaml(tmp_path):
    train_manifest, val_manifest, train_image, val_image = build_dataset(tmp_path)
    out = tmp_path / "out"
    write_dataset_yaml(out, train_manifest, val_manifest)
    assert list_lines(out / "train.txt") == [str(train_image.resolve())]
    assert list_lines(out / "val.txt") == [str(val_image.resolve())]


def _write_manifest_with_repeat(tmp_path, repeat):
    """A trainable manifest whose single positive row declares an explicit repeat count."""
    image = tmp_path / "images" / "train_00000.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    manifest = tmp_path / "train_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "split", "source_type", "repeat"])
        w.writeheader()
        w.writerow({"file": image.name, "split": "train", "source_type": "SYNTHETIC_3D",
                    "repeat": "" if repeat is None else str(repeat)})
    val = tmp_path / "val_manifest.csv"
    with val.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "split", "source_type"])
        w.writeheader()
        w.writerow({"file": image.name, "split": "val", "source_type": "SYNTHETIC_3D"})
    return manifest, val


class TestExplicitOversampling:
    """Spec 07 section 3.4 requires a difficult-example oversampling comparison.

    The first attempt repeated manifest ROWS, and nothing happened: _dedupe removed the repeats before the
    trainer saw them, because _dedupe exists to stop an image being silently trained on twice. That guard is
    correct and stays. What this adds is an EXPLICIT, declared repeat count, so oversampling is something a
    manifest says on purpose rather than something a duplicated row does by accident.
    """

    def test_a_declared_repeat_appears_that_many_times_in_the_list(self, tmp_path):
        manifest, val = _write_manifest_with_repeat(tmp_path, 3)
        out = tmp_path / "out"
        write_dataset_yaml(out, manifest, val)
        lines = list_lines(out / "train.txt")
        assert len(lines) == 3, f"expected 3 entries for repeat=3, got {len(lines)}"
        assert len(set(lines)) == 1

    def test_no_repeat_column_keeps_one_entry(self, tmp_path):
        manifest, val = _write_manifest_with_repeat(tmp_path, None)
        out = tmp_path / "out"
        write_dataset_yaml(out, manifest, val)
        assert len(list_lines(out / "train.txt")) == 1

    def test_accidental_duplicate_rows_are_still_collapsed(self, tmp_path):
        """The guard this feature must NOT remove: three identical rows are one image, not three."""
        manifest, val = _write_manifest_with_repeat(tmp_path, None)
        with manifest.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["train_00000.jpg", "train", "SYNTHETIC_3D", ""])
            w.writerow(["train_00000.jpg", "train", "SYNTHETIC_3D", ""])
        out = tmp_path / "out"
        write_dataset_yaml(out, manifest, val)
        assert len(list_lines(out / "train.txt")) == 1

    def test_duplicate_rows_disagreeing_on_repeat_are_refused(self, tmp_path):
        """Silently picking one of two different repeat counts is exactly the kind of quiet wrong answer
        this module exists to prevent."""
        manifest, val = _write_manifest_with_repeat(tmp_path, 2)
        with manifest.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["train_00000.jpg", "train", "SYNTHETIC_3D", "5"])
        with pytest.raises(DatasetConfigError):
            write_dataset_yaml(tmp_path / "out", manifest, val)


def test_dataset_configuration_never_calls_path_resolve(tmp_path, monkeypatch):
    """Path.resolve() calls lstat/readlink on every path component.

    On the training host the project tree is a FUSE mount (/home/T7 is fuseblk) where that call
    HANGS rather than returning slowly. A 673-row manifest stalled the trainer indefinitely with
    the process blocked in posixpath._joinrealpath and consuming ZERO CPU, which is
    indistinguishable from a dead GPU job until faulthandler is enabled and SIGABRT dumps the
    stack. The paths only need to be absolute and normalised, which abspath does without touching
    symlinks. This test fails the moment a resolve() call returns to this module.
    """
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    out = tmp_path / "out"

    def forbidden(self, *args, **kwargs):
        raise AssertionError(
            "Path.resolve() must not be used here: it hangs on the FUSE-mounted training tree"
        )

    monkeypatch.setattr(Path, "resolve", forbidden)
    write_dataset_yaml(out, train_manifest, val_manifest)
    assert (out / "train.txt").is_file()


def test_yaml_points_at_the_written_lists_and_the_data_root(tmp_path):
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    out = tmp_path / "out"
    document = yaml.safe_load(
        write_dataset_yaml(out, train_manifest, val_manifest, data_root=tmp_path).read_text(
            encoding="utf-8"
        )
    )
    assert document["train"] == str((out / "train.txt").resolve())
    assert document["val"] == str((out / "val.txt").resolve())
    assert document["path"] == str(tmp_path.resolve())


def test_output_directory_is_created(tmp_path):
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    out = tmp_path / "run" / "dataset"
    write_dataset_yaml(out, train_manifest, val_manifest)
    assert (out / "train.txt").is_file()
    assert (out / "val.txt").is_file()
    assert (out / DATASET_YAML_NAME).is_file()


def test_hard_negatives_stay_in_the_training_pool(tmp_path):
    """Spec section 6: negative samples are training data, not noise to drop."""
    make_image(tmp_path, "train", "train_000.jpg")
    make_image(tmp_path, "train", "train_001.jpg")
    _, val_manifest, _, _ = build_dataset(tmp_path)
    train_manifest = write_manifest(
        tmp_path / "manifest_train.csv",
        [
            row("train_000.jpg", "train"),
            row("train_001.jpg", "train", source_type="NEGATIVE", is_negative="True"),
        ],
    )
    out = tmp_path / "out"
    write_dataset_yaml(out, train_manifest, val_manifest)
    assert len(list_lines(out / "train.txt")) == 2


def test_existing_image_list_is_accepted(tmp_path):
    """A pre-built list is a legal source; it is validated like a manifest."""
    train_image = make_image(tmp_path, "train", "train_000.jpg")
    val_image = make_image(tmp_path, "val", "val_000.jpg")
    train_list = tmp_path / "source_train.txt"
    train_list.write_text(str(train_image.resolve()), encoding="utf-8")
    val_list = tmp_path / "source_val.txt"
    val_list.write_text(str(val_image.resolve()), encoding="utf-8")
    out = tmp_path / "out"
    write_dataset_yaml(out, train_list, val_list)
    assert list_lines(out / "train.txt") == [str(train_image.resolve())]
    assert list_lines(out / "val.txt") == [str(val_image.resolve())]


def test_rebuilding_from_the_written_lists_is_idempotent(tmp_path):
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    out = tmp_path / "out"
    write_dataset_yaml(out, train_manifest, val_manifest)
    first = list_lines(out / "train.txt")
    write_dataset_yaml(out, out / "train.txt", out / "val.txt")
    assert list_lines(out / "train.txt") == first


@pytest.mark.parametrize("split", ["fixed_core_test", "challenge_test"])
def test_refuses_a_held_out_split_row(tmp_path, split):
    """Spec sections 6 and 3.10: the frozen and challenge pools never train."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    train_manifest = write_manifest(
        tmp_path / "manifest_leak.csv", [row("train_000.jpg", split)]
    )
    with pytest.raises(ForbiddenSplitError) as error:
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)
    assert split in str(error.value)
    assert not (tmp_path / "out" / DATASET_YAML_NAME).exists()


@pytest.mark.parametrize(
    "value",
    [
        "FIXED_CORE_TEST",
        "Fixed_Core_Test",
        "  fixed_core_test  ",
        "fixed-core-test",
        "FixedCoreTest",
        "Fixed Core Test",
        "CHALLENGE_TEST",
        "challenge test",
        "ChallengeTest",
        "test",
        "core_test",
        "holdout_test",
    ],
)
def test_refuses_disguised_held_out_split_values(tmp_path, value):
    """Case, whitespace and separator tricks must not defeat the guard."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    train_manifest = write_manifest(
        tmp_path / "manifest_leak.csv", [row("train_000.jpg", value)]
    )
    with pytest.raises(ForbiddenSplitError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


@pytest.mark.parametrize("value", ["holdout", "extra", "val2", "production", ""])
def test_refuses_an_unknown_split_value(tmp_path, value):
    """An unreadable split cannot be assumed trainable, so it is refused too."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    train_manifest = write_manifest(
        tmp_path / "manifest_odd.csv", [row("train_000.jpg", value)]
    )
    with pytest.raises(UnknownSplitError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_refuses_a_val_row_in_the_train_pool(tmp_path):
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    write_manifest(
        train_manifest, [row("train_000.jpg", "train"), row("val_000.jpg", "val")]
    )
    with pytest.raises(SplitRoleMismatchError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_refuses_a_train_row_in_the_val_pool(tmp_path):
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    write_manifest(val_manifest, [row("val_000.jpg", "train")])
    with pytest.raises(SplitRoleMismatchError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_refuses_an_image_that_lives_in_a_held_out_directory(tmp_path):
    """A row labelled train whose file sits in challenge_test is still leakage."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    leaked = make_image(tmp_path, "challenge_test", "leaked.jpg")
    relative = leaked.relative_to(tmp_path).as_posix()
    train_manifest = write_manifest(
        tmp_path / "manifest_train.csv", [row(relative, "train")]
    )
    with pytest.raises(ForbiddenSplitError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_refuses_a_held_out_path_in_a_prebuilt_image_list(tmp_path):
    val_image = make_image(tmp_path, "val", "val_000.jpg")
    held_out = make_image(tmp_path, "fixed_core_test", "core_000.jpg")
    train_list = tmp_path / "source_train.txt"
    train_list.write_text(str(held_out.resolve()), encoding="utf-8")
    val_list = tmp_path / "source_val.txt"
    val_list.write_text(str(val_image.resolve()), encoding="utf-8")
    with pytest.raises(ForbiddenSplitError):
        write_dataset_yaml(tmp_path / "out", train_list, val_list)


def test_refuses_an_empty_pool(tmp_path):
    """Training on nothing is a mistake, not a valid configuration."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    train_manifest = write_manifest(tmp_path / "manifest_empty.csv", [])
    with pytest.raises(DatasetConfigError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_missing_manifest_is_refused(tmp_path):
    _, val_manifest, _, _ = build_dataset(tmp_path)
    with pytest.raises(ManifestMissingError):
        write_dataset_yaml(tmp_path / "out", tmp_path / "absent.csv", val_manifest)


def test_manifest_without_a_split_column_is_refused(tmp_path):
    """Without a split column the guard cannot run, so the input is rejected."""
    _, val_manifest, _, _ = build_dataset(tmp_path)
    make_image(tmp_path, "train", "train_000.jpg")
    train_manifest = write_manifest(
        tmp_path / "manifest_nosplit.csv",
        [{"file": "train_000.jpg", "source_type": "SYNTHETIC_3D"}],
        columns=("file", "source_type"),
    )
    with pytest.raises(ManifestError):
        write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)


def test_building_a_dataset_configuration_does_not_import_ultralytics(tmp_path):
    """Only the CLI training path may import Ultralytics; this step never does."""
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    write_dataset_yaml(tmp_path / "out", train_manifest, val_manifest)
    assert "ultralytics" not in sys.modules

def test_written_files_use_lf_line_endings(tmp_path):
    """Training runs on a Linux host: a trailing CR would break every image path.

    Windows text mode turns every written newline into CRLF, and Ultralytics on the
    remote host would then look for "train_00000.jpg\r" and find nothing. The list
    files must therefore be written with an explicit LF.
    """
    train_manifest, val_manifest, _, _ = build_dataset(tmp_path)
    out = tmp_path / "out"
    write_dataset_yaml(out, train_manifest, val_manifest)
    for name in ("train.txt", "val.txt", DATASET_YAML_NAME):
        raw = (out / name).read_bytes()
        assert b"\r" not in raw
        assert raw.endswith(b"\n")
    assert (out / "train.txt").read_bytes().count(b"\n") == 1


def test_the_module_states_that_it_must_run_on_the_training_host():
    """The written configuration carries absolute paths of the generating machine.

    That is only correct if the configuration was generated on the host that trains,
    so the requirement belongs in the module documentation, not in a reviewer's head.
    """
    from src.perception.shuttle_detection import ultralytics_dataset

    doc = ultralytics_dataset.__doc__ or ""
    assert "training host" in doc
    assert "absolute" in doc
