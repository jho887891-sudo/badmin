#!/usr/bin/env python3
"""Schema for the shuttlecock baseline training configuration.

Why this module exists: every baseline run in this sub-project has to be
comparable with every other one. A YAML file that is only read by the training
script cannot promise that -- a typo (`nc: 2`, `lr0: 0.1`, a renamed key) would
silently change the experiment instead of failing. This module turns the baseline
YAML into one frozen, validated object with an explicit field list, so a run
either uses exactly the declared baseline or refuses to start.

Spec: docs/superpowers/specs/02_BASELINE_TRAINING_SPEC.md (sections 2 and 3: the
task is nc=1, class 0 "shuttlecock", and the baseline is a standard single-frame
detector).

Run:
    python tests/perception/shuttle_detection/test_training_config.py -v
    pytest tests/perception/shuttle_detection/test_training_config.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.training_config import (  # noqa: E402
    BASELINE_CONFIG_PATH,
    BaselineConfig,
)

MINIMAL_YAML = """
model: yolo26s.pt
nc: 1
name: shuttlecock
imgsz: 640
seed: 17
"""


def write_yaml(tmp_path: Path, text: str, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_baseline_config_requires_single_class(tmp_path):
    """Plan step 1: the minimal baseline file loads and reports nc=1."""
    path = write_yaml(tmp_path, MINIMAL_YAML)
    config = BaselineConfig.load(path)
    assert config.nc == 1
    assert config.name == "shuttlecock"


def test_missing_optional_keys_fall_back_to_the_declared_baseline(tmp_path):
    """A terse file must not mean an experiment with unset hyper-parameters."""
    config = BaselineConfig.load(write_yaml(tmp_path, MINIMAL_YAML))
    assert (config.model, config.imgsz, config.seed) == ("yolo26s.pt", 640, 17)
    assert (config.epochs, config.batch) == (100, 16)
    assert (config.optimizer, config.lr0) == ("auto", 0.01)


def test_multi_class_configuration_is_rejected(tmp_path):
    """nc=2 would silently turn the task into a different detection problem."""
    path = write_yaml(tmp_path, MINIMAL_YAML.replace("nc: 1", "nc: 2"))
    with pytest.raises(ValueError) as error:
        BaselineConfig.load(path)
    assert "nc" in str(error.value)


def test_zero_class_configuration_is_rejected(tmp_path):
    path = write_yaml(tmp_path, MINIMAL_YAML.replace("nc: 1", "nc: 0"))
    with pytest.raises(ValueError):
        BaselineConfig.load(path)


def test_config_is_frozen(tmp_path):
    """A loaded config is the record of a run: nothing may rewrite it in place."""
    config = BaselineConfig.load(write_yaml(tmp_path, MINIMAL_YAML))
    with pytest.raises(Exception):
        config.lr0 = 0.5  # type: ignore[misc]


def test_unknown_key_is_rejected(tmp_path):
    """A misspelled key is a changed experiment, not a harmless extra field."""
    path = write_yaml(tmp_path, MINIMAL_YAML + "learnign_rate: 0.01")
    with pytest.raises(ValueError) as error:
        BaselineConfig.load(path)
    assert "learnign_rate" in str(error.value)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        BaselineConfig.load(tmp_path / "nope.yaml")


def test_non_mapping_document_is_rejected(tmp_path):
    path = write_yaml(tmp_path, "- yolo26s.pt\n- shuttlecock\n")
    with pytest.raises(ValueError):
        BaselineConfig.load(path)


def test_non_numeric_imgsz_is_rejected(tmp_path):
    path = write_yaml(tmp_path, MINIMAL_YAML.replace("imgsz: 640", "imgsz: big"))
    with pytest.raises(ValueError):
        BaselineConfig.load(path)


def test_checked_in_baseline_config_has_the_declared_settings():
    """configs/shuttle_detection/baseline.yaml is the baseline of record."""
    config = BaselineConfig.load(BASELINE_CONFIG_PATH)
    assert config.model == "yolo26s.pt"
    assert config.nc == 1
    assert config.name == "shuttlecock"
    assert config.imgsz == 640
    assert config.seed == 17
    assert config.epochs == 100
    assert config.batch == 16
    assert config.optimizer == "auto"
    assert config.lr0 == 0.01


def test_as_dict_round_trips_every_field():
    """The metadata layer records the resolved config, so the export must be complete."""
    config = BaselineConfig()
    exported = config.as_dict()
    assert set(exported) == set(BaselineConfig.FIELDS)
    assert exported["nc"] == 1
    assert BaselineConfig.from_mapping(exported) == config
