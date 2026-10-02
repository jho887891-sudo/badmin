#!/usr/bin/env python3
"""Tests for the ETH-only YOLO26s 1024 baseline experiment contract (Task 1).

TDD order: these tests are written before the implementation exists.
Run:  python -m pytest tests/test_eth_only_v1_recipe.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

CONFIG_PATH = ROOT / "configs" / "eth_only_yolo26s_1024_v1.yaml"
RECIPE_JSON = ROOT / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_official_recipe.json"
ETH_REPO = Path(r"/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection")


def test_v1_config_pins_model_and_optimizer():
    from resolve_eth_only_v1_recipe import load_experiment_config
    cfg = load_experiment_config(CONFIG_PATH)
    assert cfg["model"] == "yolo26s.pt"
    assert cfg["imgsz"] == 1024
    assert cfg["optimizer"] == "AdamW"
    assert cfg["lr0"] == 1e-4
    assert cfg["freeze"] == 0
    assert cfg["epochs"] == 50
    assert cfg["nbs"] == 32
    assert cfg["batch"] == 8
    assert cfg["classes"] == ["shuttlecock"]
    assert cfg["train_domains"] == "eth_only"
    assert cfg["evaluation_only_sets"] == ["val|eth_unseen", "external_real_only",
                                          "controlled_capability", "challenge_test"]


def test_v1_config_batch_fallback_is_declared():
    from resolve_eth_only_v1_recipe import load_experiment_config
    cfg = load_experiment_config(CONFIG_PATH)
    assert cfg["batch_fallback"] == [8, 6, 4]
    assert cfg["never_touch"] == ["imgsz", "nbs", "optimizer", "lr0"]


def test_recipe_rejects_empty_repo(tmp_path):
    from resolve_eth_only_v1_recipe import RecipeResolutionError, resolve_official_recipe
    with pytest.raises(RecipeResolutionError):
        resolve_official_recipe(tmp_path)


def test_recipe_requires_official_config_json(tmp_path):
    from resolve_eth_only_v1_recipe import RecipeResolutionError, resolve_official_recipe
    (tmp_path / "runs" / "final-model").mkdir(parents=True)
    with pytest.raises(RecipeResolutionError):
        resolve_official_recipe(tmp_path)


def test_recipe_parses_official_config_and_dataset_yamls(tmp_path):
    from resolve_eth_only_v1_recipe import resolve_official_recipe
    fm = tmp_path / "runs" / "final-model"
    fm.mkdir(parents=True)
    (fm / "config.json").write_text(json.dumps({
        "seed": 42,
        "data": {"train": ["cab_1", "ml_3"], "val": [], "test": []},
        "diff_levels": {"train": ["easy", "medium"], "val": ["easy", "medium", "hard"],
                        "test": ["easy", "medium", "hard"]},
        "fraction_train": 1.0, "fraction_coco_train": 0.1, "dist_threshold": 25.0, "confidence": 0.5,
        "model_name": "yolov8s", "epochs": 50, "batch_size": 32, "workers": 8, "optim": "adamw",
        "scheduler": "linear", "lr": 0.0001, "momentum": 0.9, "amsgrad": False, "weight_decay": 0.0005,
        "imgsz": 1024, "nbs": 64, "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "augmentations": {"mosaic": 1.0, "mixup": 0.7, "scale": 0.5, "fliplr": 0.5, "hsv_h": 0.015},
    }), encoding="utf-8")
    data = tmp_path / "data"
    (data / "cab_1_easy").mkdir(parents=True)
    (data / "cab_1_easy.yaml").write_text("names:\n  0: shuttle\npath: cab_1_easy\ntrain: images/train\nval: images/val\n",
                                          encoding="utf-8")
    rec = resolve_official_recipe(tmp_path, data_root=data)
    assert rec["official_model_name"] == "yolov8s"
    assert rec["official_imgsz"] == 1024
    assert rec["official_train_locations"] == ["cab_1", "ml_3"]
    assert rec["official_train_difficulties"] == ["easy", "medium"]
    assert rec["official_confidence"] == 0.5
    assert rec["official_dist_threshold"] == 25.0
    assert rec["official_optimizer"] == "adamw"
    assert rec["official_lr"] == 1e-4
    assert rec["official_epochs"] == 50
    assert rec["official_nbs"] == 64
    assert rec["official_batch_size"] == 32
    assert rec["official_fraction_coco_train"] == 0.1
    assert rec["official_augmentations"]["mosaic"] == 1.0
    assert rec["official_negative_sources"] == ["coco_train"]
    assert rec["class_names"] == {0: "shuttle"}
    assert rec["status"] == "RESOLVED"
    assert rec["evidence"]["config_json"].endswith("config.json")
    assert any("cab_1_easy.yaml" in e for e in rec["evidence"]["dataset_yamls"])


def test_recipe_raises_when_train_locations_are_ambiguous(tmp_path):
    from resolve_eth_only_v1_recipe import RecipeResolutionError, resolve_official_recipe
    fm = tmp_path / "runs" / "final-model"
    fm.mkdir(parents=True)
    (fm / "config.json").write_text(json.dumps({"data": {"train": []}, "diff_levels": {"train": []},
                                                "model_name": "yolov8s", "imgsz": 1024}), encoding="utf-8")
    with pytest.raises(RecipeResolutionError):
        resolve_official_recipe(tmp_path, data_root=tmp_path)


def test_recipe_reports_membership_for_every_location(tmp_path):
    from resolve_eth_only_v1_recipe import resolve_official_recipe
    fm = tmp_path / "runs" / "final-model"
    fm.mkdir(parents=True)
    (fm / "config.json").write_text(json.dumps({
        "data": {"train": ["cab_1", "coco_train"], "val": [], "test": []},
        "diff_levels": {"train": ["easy", "medium"], "val": ["hard"], "test": ["hard"]},
        "model_name": "yolov8s", "imgsz": 1024, "confidence": 0.5, "dist_threshold": 25.0,
    }), encoding="utf-8")
    data = tmp_path / "data"
    for name in ["cab_1_easy", "cab_1_medium", "cab_1_hard", "ml_3_easy", "coco_train_easy", "coco_val_easy"]:
        d = data / name / "images" / "train"
        d.mkdir(parents=True)
        for i in range(3):
            (d / ("f%d.jpg" % i)).write_bytes(b"x")
    rec = resolve_official_recipe(tmp_path, data_root=data)
    m = rec["location_membership"]
    assert m["cab_1_easy"]["in_official_training"] is True
    assert m["cab_1_medium"]["in_official_training"] is True
    assert m["cab_1_hard"]["in_official_training"] is False
    assert m["ml_3_easy"]["in_official_training"] is False
    assert m["coco_train_easy"]["role"] == "official_negatives"
    assert m["coco_train_easy"]["in_official_training"] is True
    assert m["coco_val_easy"]["role"] == "benchmark_holdout"
    # coco_train is listed as a training location but must NOT be counted as shuttle positives
    assert rec["counts"]["official_negative_images"] == 3
    assert rec["counts"]["positives_train_dirs"] == 6
    assert rec["counts"]["positives_by_location"] == {"cab_1": 6}
    assert rec["status"] == "RESOLVED"


def test_real_recipe_json_when_present():
    if not RECIPE_JSON.exists():
        pytest.skip("recipe json not generated yet (Step 5 of Task 1)")
    rec = json.loads(RECIPE_JSON.read_text(encoding="utf-8"))
    assert rec["status"] == "RESOLVED"
    assert rec["official_imgsz"] == 1024
    assert "easy" in rec["official_train_difficulties"] and "medium" in rec["official_train_difficulties"]
    assert rec["official_train_locations"], "must list the official training locations"
    assert rec["evidence"]["config_json"]
    assert rec["counts"]["positives_train_dirs"] > 0
    assert rec["counts"]["official_negative_images"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
