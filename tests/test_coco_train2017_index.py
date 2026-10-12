"""Task 2 tests: the COCO train2017 index must accept only train2017, prove what it rejected, and never let
a historical V2 COCO image, an evaluation image or an unverifiable licence into a new training pool.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 2, Step 1/2).
Written before tools/index_coco_train2017.py existed, so the first run was RED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import index_coco_train2017 as I  # noqa: E402

LICENSES = [
    {"id": 1, "name": "Attribution-NonCommercial-ShareAlike 2.0 Generic",
     "url": "https://creativecommons.org/licenses/by-nc-sa/2.0/"},
    {"id": 4, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/2.0/"},
]
KINDS = ("blob", "stripes", "checker")


def _draw(path: Path, size=(48, 48), kind="blob"):
    """Structurally different patterns so the 64-bit average hash is meaningful, unlike flat colour fills."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (18, 18, 18))
    draw = ImageDraw.Draw(image)
    if kind == "blob":
        cx, cy = size[0] // 2, size[1] // 2
        r = min(size) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(232, 232, 232))
    elif kind == "stripes":
        band = max(2, size[1] // 6)
        for index, top in enumerate(range(0, size[1], band)):
            if index % 2 == 0:
                draw.rectangle([0, top, size[0], top + band - 1], fill=(240, 240, 240))
    elif kind == "checker":
        cell = max(2, size[0] // 8)
        for row in range(0, size[1], cell):
            for col in range(0, size[0], cell):
                if (row // cell + col // cell) % 2 == 0:
                    draw.rectangle([col, row, col + cell - 1, row + cell - 1], fill=(240, 240, 240))
    else:
        raise AssertionError("unknown fixture pattern %r" % kind)
    image.save(path)
    with Image.open(path) as handle:
        return handle.size


def _make_coco(root: Path, split: str = "train2017", specs=None, licenses=None):
    """Minimal but structurally faithful COCO tree: <root>/<split>/*.jpg + <root>/annotations/instances_<split>.json."""
    specs = specs if specs is not None else [{"id": 1, "file_name": "a.jpg"}]
    img_dir = root / split
    records = []
    for spec in specs:
        name = spec["file_name"]
        path = img_dir / name
        kind = spec.get("kind", KINDS[spec["id"] % len(KINDS)])
        if spec.get("copy_of"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((img_dir / spec["copy_of"]).read_bytes())
            size = Image.open(path).size
        elif spec.get("crop_of"):
            source = Image.open(img_dir / spec["crop_of"])
            path.parent.mkdir(parents=True, exist_ok=True)
            source.crop(spec.get("crop_box", (2, 2, 46, 46))).save(path)
            size = Image.open(path).size
        else:
            size = _draw(path, spec.get("size", (48, 48)), kind)
        records.append({
            "id": spec["id"],
            "file_name": name,
            "width": size[0],
            "height": size[1],
            "license": spec.get("license", 1),
            "coco_url": spec.get("coco_url", "http://images.cocodataset.org/%s/%s" % (split, name)),
            "flickr_url": spec.get("flickr_url", "http://flickr.com/x/%s" % spec["id"]),
        })
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ann_dir / ("instances_%s.json" % split)
    ann_path.write_text(json.dumps({
        "info": {"description": "fixture"},
        "licenses": licenses if licenses is not None else LICENSES,
        "images": records,
        "annotations": [],
        "categories": [],
    }), encoding="utf-8")
    return img_dir, ann_path


def test_only_train2017_accepted(tmp_path):
    """val2017 (and any other split) must be refused outright: only train2017 may feed the new pools."""
    val_dir, val_ann = _make_coco(tmp_path / "val", split="val2017", specs=[{"id": 7, "file_name": "v.jpg"}])
    with pytest.raises(I.CocoIndexBlocked):
        I.load_coco_train2017_index(val_dir, val_ann)

    train_dir, train_ann = _make_coco(tmp_path / "train", specs=[{"id": 1, "file_name": "a.jpg"}])
    index = I.load_coco_train2017_index(train_dir, train_ann)
    assert list(index["coco_image_id"]) == [1]
    assert set(index["split"]) == {"train2017"}


def test_coco_ids_unique_and_paths_safe(tmp_path):
    """Duplicate image ids and any path that escapes the split directory are blocking errors."""
    dup_dir, dup_ann = _make_coco(tmp_path / "dup", specs=[
        {"id": 1, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"},
    ])
    with pytest.raises(I.CocoIndexBlocked):
        I.load_coco_train2017_index(dup_dir, dup_ann)

    esc_dir, esc_ann = _make_coco(tmp_path / "esc", specs=[{"id": 2, "file_name": "../outside.jpg"}])
    with pytest.raises(I.CocoIndexBlocked):
        I.load_coco_train2017_index(esc_dir, esc_ann)


def test_existing_v2_coco_train_easy_id_rejected(tmp_path):
    """The coco_train_easy images the V2 run already trained on must never re-enter as NEW data."""
    img_dir, ann = _make_coco(tmp_path / "t", specs=[{"id": 100, "file_name": "a.jpg"}, {"id": 200, "file_name": "b.jpg"}])
    index = I.load_coco_train2017_index(img_dir, ann)
    prohibited = I.build_prohibited_source_index(
        v2_manifests=[], frozen_eval_manifests=[], historical_coco_ids={100},
        historical_coco_id_completeness="KNOWN",
    )
    candidates, rejections = I.filter_coco_candidates(index, prohibited)
    assert list(candidates["coco_image_id"]) == [200]
    assert any(r["reason"] == "historical_v2_coco_id" and r["coco_image_id"] == 100 for r in rejections.to_dict("records"))


def test_existing_v2_negative_hash_rejected(tmp_path):
    """The same picture re-saved under another filename is still the same picture: hash exclusion blocks it."""
    img_dir, ann = _make_coco(tmp_path / "t", specs=[
        {"id": 100, "file_name": "a.jpg"}, {"id": 101, "file_name": "a_copy.jpg", "copy_of": "a.jpg"},
    ])
    index = I.load_coco_train2017_index(img_dir, ann)
    forbidden_sha = index.loc[index["coco_image_id"] == 100, "sha256"].iloc[0]
    prohibited = I.build_prohibited_source_index(
        v2_manifests=[], frozen_eval_manifests=[], historical_coco_ids=set(),
        historical_sha256={forbidden_sha},
    )
    candidates, rejections = I.filter_coco_candidates(index, prohibited)
    assert 101 not in set(candidates["coco_image_id"])
    assert any(r["reason"] == "historical_content_hash" for r in rejections.to_dict("records"))


def test_unknown_image_permission_is_blocked(tmp_path):
    """An image whose COCO licence cannot be resolved is rejected, never silently assumed CC BY."""
    img_dir, ann = _make_coco(tmp_path / "t", specs=[
        {"id": 1, "file_name": "known.jpg", "license": 4},
        {"id": 2, "file_name": "unknown.jpg", "license": 0},
    ])
    index = I.load_coco_train2017_index(img_dir, ann)
    states = dict(zip(index["coco_image_id"], index["permission_state"]))
    assert states[1] == "declared"
    assert states[2] == "unknown"
    candidates, rejections = I.filter_coco_candidates(index, I.empty_prohibited_index())
    assert list(candidates["coco_image_id"]) == [1]
    assert any(r["reason"] == "license_unverifiable" for r in rejections.to_dict("records"))


def test_missing_annotations_json_blocked(tmp_path):
    """No annotation file, no inventory: the plan forbids substituting another split or another mirror."""
    img_dir, _ = _make_coco(tmp_path / "t", specs=[{"id": 1, "file_name": "a.jpg"}])
    with pytest.raises(I.CocoIndexBlocked):
        I.load_coco_train2017_index(img_dir, tmp_path / "t" / "annotations" / "missing.json")

    bad = tmp_path / "t" / "annotations" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(I.CocoIndexBlocked):
        I.load_coco_train2017_index(img_dir, bad)


def test_same_content_different_filename_rejected(tmp_path):
    """Byte-identical images inside the same COCO split are collapsed to one candidate, with a reason."""
    img_dir, ann = _make_coco(tmp_path / "t", specs=[
        {"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg", "copy_of": "a.jpg"},
    ])
    index = I.load_coco_train2017_index(img_dir, ann)
    assert index["sha256"].nunique() == 1
    candidates, rejections = I.filter_coco_candidates(index, I.empty_prohibited_index())
    assert len(candidates) == 1
    assert any(r["reason"] == "duplicate_content" for r in rejections.to_dict("records"))


def test_crop_or_near_duplicate_flagged_for_review(tmp_path):
    """A crop/near-duplicate is not silently trained on: it is flagged and withheld pending human review."""
    img_dir, ann = _make_coco(tmp_path / "t", specs=[
        {"id": 1, "file_name": "a.jpg", "size": (48, 48), "kind": "blob"},
        {"id": 2, "file_name": "a_crop.jpg", "crop_of": "a.jpg", "crop_box": (2, 2, 46, 46)},
        {"id": 3, "file_name": "other.jpg", "size": (48, 48), "kind": "checker"},
    ])
    index = I.load_coco_train2017_index(img_dir, ann)
    flagged = index[index["near_duplicate_group"].notna()]
    assert 2 in set(flagged["coco_image_id"]), "the crop must share a near-duplicate group with its source"
    candidates, rejections = I.filter_coco_candidates(index, I.empty_prohibited_index())
    assert 3 in set(candidates["coco_image_id"])
    assert any(r["reason"] == "near_duplicate_review" for r in rejections.to_dict("records"))
