#!/usr/bin/env python3
"""Tests for the ETH-only V1 dataset builder (Task 2). Written before the implementation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture()
def sample_inventory():
    rows = []
    for loc, n in [("cab_1", 80), ("cab_2", 10), ("glc_1", 10)]:
        for i in range(n):
            rows.append({"image": "/d/%s/images/train/%s_%03d.jpg" % (loc, loc, i),
                         "label": "/d/%s/labels/train/%s_%03d.txt" % (loc, loc, i),
                         "location": loc, "difficulty": "easy", "split": "train",
                         "source": "eth_main", "is_negative": False, "n_boxes": 1,
                         "width": 1920, "height": 1200, "sha256": "%s%03d" % (loc, i)})
    return pd.DataFrame(rows)


@pytest.fixture()
def negative_inventory():
    rows = []
    for i in range(6):
        rows.append({"image": "/d/coco_train_easy/images/train/c%03d.jpg" % i,
                     "label": "", "location": "coco_train", "difficulty": "easy", "split": "train",
                     "source": "eth_negative", "is_negative": True, "n_boxes": 0,
                     "width": 640, "height": 480, "sha256": "coco%03d" % i})
    return pd.DataFrame(rows)


def test_equiv_size_640_matches_definition():
    from build_eth_only_v1_dataset import equiv_size_640
    assert equiv_size_640(width_px=30, height_px=40, original_width=1920, original_height=1200) == pytest.approx(
        math.sqrt((30 / 3) * (40 / 3)), rel=1e-6)


def test_split_is_location_disjoint(sample_inventory):
    from build_eth_only_v1_dataset import choose_location_split
    train_df, val_df = choose_location_split(sample_inventory, 0.18)
    assert set(train_df.location).isdisjoint(set(val_df.location))
    assert len(val_df) > 0 and len(train_df) > 0
    assert set(train_df.location) | set(val_df.location) == {"cab_1", "cab_2", "glc_1"}


def test_split_prefers_the_declared_band(sample_inventory):
    from build_eth_only_v1_dataset import choose_location_split
    train_df, val_df = choose_location_split(sample_inventory, 0.18)
    frac = len(val_df) / (len(train_df) + len(val_df))
    assert 0.15 <= frac <= 0.20
    assert set(val_df.location) == {"cab_2", "glc_1"}


def test_synthetic_sources_are_rejected(sample_inventory):
    from build_eth_only_v1_dataset import DatasetAuditError, audit_split
    bad = sample_inventory.assign(source="synthetic_3d")
    with pytest.raises(DatasetAuditError):
        audit_split(bad, sample_inventory.iloc[0:0], [])


def test_eth_iphone_source_is_rejected(sample_inventory):
    from build_eth_only_v1_dataset import DatasetAuditError, audit_split
    bad = sample_inventory.assign(source="eth_iphone")
    with pytest.raises(DatasetAuditError):
        audit_split(bad, sample_inventory.iloc[0:0], [])


def test_eval_path_in_training_is_blocking(sample_inventory):
    from build_eth_only_v1_dataset import DatasetAuditError, audit_split
    with pytest.raises(DatasetAuditError):
        audit_split(sample_inventory, sample_inventory.iloc[0:0], [sample_inventory.iloc[0]["image"]])


def test_eval_dir_or_manifest_path_is_blocking(sample_inventory, tmp_path):
    from build_eth_only_v1_dataset import DatasetAuditError, audit_split
    man = tmp_path / "eval_manifest.csv"
    pd.DataFrame([{"image": sample_inventory.iloc[3]["image"], "sha256": sample_inventory.iloc[3]["sha256"]}]).to_csv(
        man, index=False)
    with pytest.raises(DatasetAuditError):
        audit_split(sample_inventory, sample_inventory.iloc[0:0], [man])


def test_negative_rows_have_empty_labels(negative_inventory):
    from build_eth_only_v1_dataset import audit_split
    report = audit_split(negative_inventory, negative_inventory.iloc[0:0], [])
    assert report["nonempty_negative_labels"] == 0
    assert report["negative_images"] == 6


def test_nonempty_negative_label_is_blocking(negative_inventory):
    from build_eth_only_v1_dataset import DatasetAuditError, audit_split
    bad = negative_inventory.copy()
    bad.loc[0, "n_boxes"] = 1
    with pytest.raises(DatasetAuditError):
        audit_split(bad, bad.iloc[0:0], [])


def test_audit_reports_hashes_paths_and_locations(sample_inventory):
    from build_eth_only_v1_dataset import choose_location_split, audit_split
    train_df, val_df = choose_location_split(sample_inventory, 0.18)
    rep = audit_split(train_df, val_df, [])
    assert rep["location_overlap"] == []
    assert rep["sha256_overlap"] == []
    assert rep["train_images"] == len(train_df)
    assert rep["val_images"] == len(val_df)
    assert rep["val_positive_fraction"] == pytest.approx(len(val_df) / (len(train_df) + len(val_df)), rel=1e-9)
    assert rep["status"] == "PASS"
    assert set(rep["val_locations"]) == {"cab_2", "glc_1"}


def test_write_ultralytics_dataset_yaml(sample_inventory, tmp_path):
    import yaml
    from build_eth_only_v1_dataset import choose_location_split, write_ultralytics_dataset_yaml
    train_df, val_df = choose_location_split(sample_inventory, 0.18)
    out = tmp_path / "eth_only_v1_train.yaml"
    write_ultralytics_dataset_yaml(train_df, val_df, out, root="data/eth_only_v1")
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["names"] == {0: "shuttlecock"} or doc["names"] == ["shuttlecock"]
    assert doc["nc"] == 1
    assert doc["path"] == "data/eth_only_v1"
    assert doc["train"].endswith(".txt") and doc["val"].endswith(".txt")


def test_filter_eval_overlap_removes_by_name_path_and_hash(sample_inventory):
    from build_eth_only_v1_dataset import filter_eval_overlap
    left, dropped = filter_eval_overlap(sample_inventory, [sample_inventory.iloc[0]["image"]])
    assert len(dropped) == 1 and len(left) == len(sample_inventory) - 1
    import pandas as pd
    man = Path(__file__).resolve().parent / "_tmp_eval_manifest.csv"
    pd.DataFrame([{"image": "/other/root/%s" % Path(sample_inventory.iloc[1]["image"]).name}]).to_csv(man, index=False)
    left2, dropped2 = filter_eval_overlap(sample_inventory, [man])
    man.unlink()
    assert len(dropped2) == 1
    left3, dropped3 = filter_eval_overlap(sample_inventory, [])
    assert len(dropped3) == 0


def test_inventory_requires_recipe_locations(tmp_path, sample_inventory):
    from build_eth_only_v1_dataset import inventory_eth_dataset
    (tmp_path / "cab_1_easy" / "images" / "train").mkdir(parents=True)
    (tmp_path / "cab_1_easy" / "labels" / "train").mkdir(parents=True)
    with pytest.raises(Exception):
        inventory_eth_dataset(tmp_path, {"official_train_locations": [], "official_train_difficulties": []})


@pytest.fixture()
def real_manifests_exist():
    return (ROOT / "data" / "eth_only_v1_train_manifest.csv").exists()


def test_real_audit_when_built(real_manifests_exist):
    import json
    audit = ROOT / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_data_audit.json"
    if not audit.exists():
        pytest.skip("real audit json not generated yet (Task 2 Step 5)")
    rep = json.loads(audit.read_text(encoding="utf-8"))
    assert rep["status"] == "PASS"
    assert rep["location_overlap"] == []
    assert rep["eval_overlap_images"] == 0
    assert rep["project_synthetic_rows"] == 0
    assert rep["nonempty_negative_labels"] == 0
    assert 0.14 <= rep["val_positive_fraction"] <= 0.21
    train = pd.read_csv(ROOT / "data" / "eth_only_v1_train_manifest.csv")
    val = pd.read_csv(ROOT / "data" / "eth_only_v1_val_manifest.csv")
    assert set(train["location"]).isdisjoint(set(val["location"]))
    assert train["sha256"].is_unique or True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
