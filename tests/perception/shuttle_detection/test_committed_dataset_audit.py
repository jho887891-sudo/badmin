#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-01: tests for the committed-manifest-only dataset audit.

Two layers:
  * fixture tests plant a leak and prove the audit FINDS it (a detector of leaks
    that never fires proves nothing);
  * committed-manifest tests run on the real committed manifests and FAIL on any
    group leakage that is not in KNOWN_SESSION_CROSSINGS, while reporting
    DANGLING rows as a count instead of crashing.

Run:
    python -m pytest tests/perception/shuttle_detection/test_committed_dataset_audit.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.committed_dataset_audit import (  # noqa: E402
    AUDIT_SCOPE,
    EVAL_SIZE_BUCKETS,
    KNOWN_SESSION_CROSSINGS,
    UNBUCKETED_SMALL,
    UNKNOWN_SIZE_BUCKET,
    audit_manifest,
    audit_manifests,
    bucket_size_eval,
    derive_groups,
    dumps,
    find_cross_split_overlaps,
    find_group_leakage,
    frame_identity,
    read_manifest_rows,
    summarise_manifest,
)

V1_MANIFEST = ROOT / "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv"
V1_DIGEST = ROOT / "outputs/shuttle_capability/v1_dataset/v1_dataset_digest.json"
TRAIN_MANIFEST = ROOT / "outputs/shuttle_capability/train_data/manifest_train.csv"
VAL_MANIFEST = ROOT / "outputs/shuttle_capability/train_data/manifest_val.csv"

V1_HEADER = "image,label,source,split,location,difficulty,width,height,n_boxes,eq640_min,eq640_max,bucket\n"


def _write(path: Path, header: str, rows) -> Path:
    path.write_text(header + "".join(row + "\n" for row in rows), encoding="utf-8")
    return path


class FrameIdentityTests(unittest.TestCase):
    def test_volume_prefix_is_dropped_so_one_frame_has_one_identity(self):
        a = frame_identity(r"D:\_eth_data\ds\cab_1_easy\images\train\cab_1_00000.jpg")
        b = frame_identity(r"_eth_data\ds\cab_1_easy\images\train\cab_1_00000.jpg")
        self.assertEqual(a.relpath, b.relpath)
        self.assertEqual(a.session, "_eth_data/ds/cab_1_easy")
        self.assertEqual(a.session, b.session)
        self.assertEqual(a.frame_index, 0)
        self.assertEqual(a.clip, "_eth_data/ds/cab_1_easy/cab_1")

    def test_native_and_posix_separators_give_the_same_identity(self):
        a = frame_identity("outputs/shuttle_capability/train_data/train/images/train_00042.jpg")
        b = frame_identity("outputs\\shuttle_capability\\train_data\\train\\images\\train_00042.jpg")
        self.assertEqual(a.relpath, b.relpath)
        self.assertEqual(a.frame_index, 42)

    def test_trailing_index_is_optional(self):
        ident = frame_identity("ds/pool/raw/bg_008.jpg")
        self.assertIsNone(ident.frame_index)
        self.assertEqual(ident.session, "ds/pool")
        self.assertEqual(ident.clip, "ds/pool/bg_008")

    def test_declared_group_becomes_a_grouping_key(self):
        groups = derive_groups("ds/x/images/train/a_00001.jpg", "tbg_008.jpg")
        self.assertIn("declared_group", groups)
        self.assertEqual(groups["declared_group"], "declared:tbg_008.jpg")


class SizeBucketTests(unittest.TestCase):
    def test_edges_are_half_open(self):
        self.assertEqual(bucket_size_eval(3.999), UNBUCKETED_SMALL)
        self.assertEqual(bucket_size_eval(4.0), EVAL_SIZE_BUCKETS[0][0])
        self.assertEqual(bucket_size_eval(5.999), "4-6px")
        self.assertEqual(bucket_size_eval(6.0), "6-10px")
        self.assertEqual(bucket_size_eval(10.0), "10-20px")
        self.assertEqual(bucket_size_eval(20.0), ">20px")
        self.assertEqual(bucket_size_eval(1000.0), ">20px")

    def test_unknown_sizes_are_not_silently_bucketed(self):
        for value in (None, "", "nan", "inf", 0.0, -3.0):
            self.assertEqual(bucket_size_eval(value), UNKNOWN_SIZE_BUCKET)


class FixtureLeakageTests(unittest.TestCase):
    """Planted defects: every one of them must be DETECTED."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "repo"
        self.images = self.root / "ds" / "session_a" / "images"
        self.images.mkdir(parents=True)
        for name in ("frame_00001.jpg", "frame_00002.jpg", "frame_00003.jpg"):
            (self.images / name).write_bytes(b"not-an-image")

    def tearDown(self):
        self._tmp.cleanup()

    def _manifest(self, rows, name="manifest.csv"):
        return _write(self.tmp / name, V1_HEADER, rows)

    def test_session_leakage_train_and_val_across_one_directory(self):
        manifest = self._manifest([
            "ds/session_a/images/frame_00001.jpg,,real,train,session_a,,1280,720,1,5.0,5.0,4-6",
            "ds/session_a/images/frame_00002.jpg,,real,train,session_a,,1280,720,1,6.5,6.5,6-8",
            "ds/session_a/images/frame_00003.jpg,,real,val,session_a,,1280,720,1,22.0,22.0,16-24",
        ])
        findings = find_group_leakage(read_manifest_rows(manifest, "v1_dataset"))
        self.assertEqual(len(findings["session"]), 1)
        self.assertEqual(findings["session"][0]["group"], "session:ds/session_a")
        self.assertEqual(findings["session"][0]["splits"], {"train": 2, "val": 1})
        self.assertGreaterEqual(len(findings["clip"]), 1)

    def test_clean_manifest_reports_no_leakage(self):
        manifest = self._manifest([
            "ds/session_a/images/frame_00001.jpg,,real,train,session_a,,1280,720,1,5.0,5.0,4-6",
            "ds/session_b/images/frame_00001.jpg,,real,val,session_b,,1280,720,1,6.5,6.5,6-8",
        ])
        findings = find_group_leakage(read_manifest_rows(manifest, "v1_dataset"))
        self.assertEqual(findings["session"], [])
        self.assertEqual(findings["clip"], [])

    def test_duplicate_frame_identifier_across_splits(self):
        manifest = self._manifest([
            "ds/a/images/frame_00001.jpg,,real,train,a,,1280,720,1,5.0,5.0,4-6",
            "ds/b/images/frame_00001.jpg,,real,val,b,,1280,720,1,5.0,5.0,4-6",
            "ds/a/images/frame_00002.jpg,,real,train,a,,1280,720,1,5.0,5.0,4-6",
            "ds/a/images/frame_00002.jpg,,real,val,a,,1280,720,1,5.0,5.0,4-6",
        ])
        overlaps = find_cross_split_overlaps(read_manifest_rows(manifest, "v1_dataset"))
        self.assertIn("frame_00001.jpg", overlaps["same_basename_cross_split"])
        self.assertEqual(overlaps["same_path_cross_split"], ["ds/a/images/frame_00002.jpg"])
        self.assertEqual(overlaps["same_basename_repeated_rows"], 2)

    def test_dangling_rows_are_counted_not_raised(self):
        manifest = self._manifest([
            "ds/session_a/images/frame_00001.jpg,,real,train,session_a,,1280,720,1,5.0,5.0,4-6",
            "ds/session_a/images/frame_00009.jpg,,real,train,session_a,,1280,720,1,5.0,5.0,4-6",
            "ds/session_a/images/frame_00010.jpg,,real,train,session_a,,1280,720,0,,,negative",
        ])
        summary = summarise_manifest(read_manifest_rows(manifest, "v1_dataset"), self.root)
        self.assertEqual(summary["dangling"]["rows"], 2)
        self.assertEqual(summary["dangling"]["present"], 1)
        self.assertEqual(len(summary["dangling"]["sample"]), 2)
        self.assertEqual(summary["negatives_total"], 1)
        self.assertEqual(summary["splits"], {"train": 3})

    def test_negative_flag_disagreement_is_counted(self):
        manifest = self._manifest([
            # n_boxes says positive but the bucket column says negative
            "ds/a/images/f_00001.jpg,,real,train,a,,1280,720,1,5.0,5.0,negative",
            "ds/a/images/f_00002.jpg,,real,train,a,,1280,720,0,,,negative",
        ])
        summary = summarise_manifest(read_manifest_rows(manifest, "v1_dataset"), self.root)
        self.assertEqual(summary["negative_bucket_disagreements"], 1)

    def test_size_distribution_sums_to_rows(self):
        manifest = self._manifest([
            "ds/a/images/f_00001.jpg,,real,train,a,,1280,720,1,5.0,5.0,4-6",
            "ds/a/images/f_00002.jpg,,real,train,a,,1280,720,1,7.0,7.0,6-8",
            "ds/a/images/f_00003.jpg,,real,train,a,,1280,720,0,,,negative",
        ])
        summary = summarise_manifest(read_manifest_rows(manifest, "v1_dataset"), self.root)
        self.assertEqual(sum(summary["size_distribution"].values()), summary["rows"])
        self.assertEqual(summary["size_distribution"]["unknown"], 1)
        self.assertEqual(summary["size_distribution"]["4-6px"], 1)
        self.assertEqual(summary["size_distribution"]["6-10px"], 1)

    def test_missing_manifest_raises_a_typed_error_not_a_traceback(self):
        from src.perception.shuttle_detection.committed_dataset_audit import ManifestAuditError
        with self.assertRaises(ManifestAuditError):
            read_manifest_rows(self.tmp / "does_not_exist.csv", "v1_dataset")

    def test_report_is_json_serialisable_and_scope_is_manifest_only(self):
        manifest = self._manifest([
            "ds/a/images/f_00001.jpg,,real,train,a,,1280,720,1,5.0,5.0,4-6",
        ])
        report = audit_manifests([(str(manifest), "v1_dataset")], repo_root=self.root)
        text = dumps(report)
        reloaded = json.loads(text)
        self.assertEqual(reloaded["scope"], AUDIT_SCOPE)
        self.assertFalse(reloaded["manifests"][0]["scope"]["image_content_checked"])
        self.assertIn("image-content duplicates", " ".join(reloaded["checks_not_performed"]))


@unittest.skipUnless(V1_MANIFEST.is_file(), "committed v1 dataset manifest is not present")
class CommittedManifestTests(unittest.TestCase):
    """The real audit: runs on git-committed manifests only."""

    @classmethod
    def setUpClass(cls):
        cls.report = audit_manifests(
            [(str(V1_MANIFEST), "v1_dataset"),
             (str(TRAIN_MANIFEST), "synthetic_render"),
             (str(VAL_MANIFEST), "synthetic_render")],
            repo_root=ROOT,
        )
        cls.v1 = cls.report["manifests"][0]
        cls.synthetic = cls.report["manifests"][1:]

    def test_no_frame_straddles_a_split(self):
        self.assertEqual(self.v1["overlaps"]["same_path_cross_split"], [])
        self.assertEqual(self.v1["overlaps"]["same_basename_cross_split"], [])
        self.assertEqual(self.v1["overlaps"]["same_clip_frame_index_cross_split"], [])

    def test_group_leakage_is_only_the_reviewed_and_open_list(self):
        """FAILS on any session/clip crossing that is not documented as known."""
        leakage = self.report["leakage_summary"]
        self.assertEqual(
            leakage["unexplained_session_crossings"], [],
            "a session now straddles train/val and is not in KNOWN_SESSION_CROSSINGS: "
            + repr(leakage["unexplained_session_crossings"]))
        self.assertEqual(leakage["unexplained_clip_crossings"], [])

    def test_known_crossings_cannot_rot(self):
        """If a documented crossing disappears the audit must be updated, not silently pass."""
        self.assertEqual(self.report["leakage_summary"]["known_crossings_absent_now"], [])
        for group, meta in KNOWN_SESSION_CROSSINGS.items():
            self.assertIn(group, self.report["leakage_summary"]["session_crossings"])
            self.assertEqual(meta["status"], "OPEN_LEAK_UNRESOLVED")

    def test_split_counts_cover_every_row_and_no_test_split_exists(self):
        self.assertEqual(sum(self.v1["splits"].values()), self.v1["rows"])
        self.assertEqual(sorted(self.v1["splits"]), ["train", "val"])
        self.assertFalse(self.v1["test_split_present"])

    def test_negatives_and_sizes_are_counted(self):
        self.assertGreater(self.v1["negatives_total"], 0)
        self.assertEqual(sum(self.v1["size_distribution"].values()), self.v1["rows"])
        self.assertGreater(self.v1["size_distribution"]["4-6px"], 0)
        self.assertGreater(self.v1["size_distribution"]["unknown"], 0)  # negatives have no size

    def test_dangling_is_a_count_and_present_plus_dangling_is_every_row(self):
        for block in self.report["manifests"]:
            self.assertEqual(block["dangling"]["rows"] + block["dangling"]["present"],
                             block["rows"], block["manifest"])

    def test_manifest_sha256_is_recorded_for_self_evidence(self):
        for block in self.report["manifests"]:
            self.assertIsNotNone(block["manifest_sha256"])
            self.assertEqual(len(block["manifest_sha256"]), 64)

    def test_manifest_rows_agree_with_the_committed_digest(self):
        if not V1_DIGEST.is_file():
            self.skipTest("v1_dataset_digest.json not committed")
        digest = json.loads(V1_DIGEST.read_text(encoding="utf-8"))
        self.assertEqual(digest["rows"], self.v1["rows"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
