"""Tests for tools/analyze_eth_only_v1_fn.py (read-only V1 false-negative error analysis)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_eth_only_v1_fn as A  # noqa: E402

FN_JSON = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_fn_analysis.json"
FN_CSV = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_fn_analysis.csv"
ETH_VS_V1 = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_vs_ours_eth_only_v1.csv"

MANIFEST_COLUMNS = ["image", "label", "source", "split", "location", "difficulty", "width", "height"]


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    return path


def _gt(image, gt_index=0, bucket="8-12", matched=True):
    return {"image": image, "gt_index": gt_index, "bucket": bucket, "matched": matched}


@pytest.fixture()
def location_manifest(tmp_path):
    return _manifest(tmp_path, [
        {"image": "/imgs/a.jpg", "location": "ml_3", "source": "eth_main"},
        {"image": "/imgs/b.jpg", "location": "ml_3", "source": "eth_main"},
        {"image": "/imgs/c.jpg", "location": "synthetic", "source": "synthetic"},
        {"image": "/imgs/d.jpg", "location": "ml_3", "source": "eth_main"},
    ])


# ------------------------------------------------------------------------------------------------
# fn_by_location
# ------------------------------------------------------------------------------------------------
def test_fn_by_location_aggregates_per_manifest_location(location_manifest):
    rows = [_gt("/imgs/a.jpg", 0, matched=True), _gt("/imgs/a.jpg", 1, matched=False),
            _gt("/imgs/b.jpg", 0, matched=False), _gt("/imgs/c.jpg", 0, matched=True)]
    out = A.fn_by_location(rows, location_manifest)
    assert [r["key"] for r in out] == ["ml_3", "synthetic"], "worst (most FN) location first"
    assert (out[0]["gt"], out[0]["fn"]) == (3, 2)
    assert out[0]["fn_rate"] == pytest.approx(2 / 3)
    assert out[0]["extra"] == {"images": 2, "field": "location"}
    assert (out[1]["gt"], out[1]["fn"], out[1]["fn_rate"]) == (1, 0, 0.0)


def test_fn_by_location_falls_back_to_unknown_without_a_manifest_row(location_manifest):
    rows = [_gt("/imgs/nope.jpg", 0, matched=False)]
    out = A.fn_by_location(rows, location_manifest)
    assert len(out) == 1 and out[0]["key"] == "?" and out[0]["fn_rate"] == 1.0


def test_fn_by_location_normalises_windows_separators_and_case(tmp_path):
    man = _manifest(tmp_path, [{"image": "C:" + chr(92) + "data" + chr(92) + "A.JPG", "location": "ml_6"}])
    out = A.fn_by_location([_gt("c:/DATA/a.jpg", 0, matched=False)], man)
    assert out[0]["key"] == "ml_6" and out[0]["gt"] == 1 and out[0]["fn"] == 1


# ------------------------------------------------------------------------------------------------
# fn_by_size
# ------------------------------------------------------------------------------------------------
def test_fn_by_size_honours_the_canonical_bucket_boundaries():
    """[lo, hi) buckets: a value exactly on a boundary joins the upper bucket."""
    assert A.err.ev.bucket_of(3.999) == "<4"
    assert A.err.ev.bucket_of(4.0) == "4-6"
    assert A.err.ev.bucket_of(7.999) == "6-8"
    assert A.err.ev.bucket_of(8.0) == "8-12", "8.0 must join 8-12, not 6-8"
    rows = [{"image": "/i.jpg", "gt_index": i, "bucket": A.err.ev.bucket_of(eq), "matched": i != 0}
            for i, eq in enumerate([3.999, 4.0, 7.999, 8.0])]
    out = A.fn_by_size(rows)
    assert [r["key"] for r in out] == ["<4", "4-6", "6-8", "8-12", "<8"]
    by_key = {r["key"]: r for r in out}
    assert (by_key["<4"]["gt"], by_key["<4"]["fn"]) == (1, 1)
    assert (by_key["8-12"]["gt"], by_key["8-12"]["fn"]) == (1, 0)
    assert by_key["8-12"]["fn_rate"] == 0.0


def test_fn_by_size_adds_the_lt8_aggregate():
    rows = [_gt("/i.jpg", 0, "<4", False), _gt("/i.jpg", 1, "4-6", False), _gt("/i.jpg", 2, "6-8", True),
            _gt("/i.jpg", 3, "16-24", False)]
    out = A.fn_by_size(rows)
    agg = [r for r in out if r["key"] == "<8"][0]
    assert (agg["gt"], agg["fn"]) == (3, 2)
    assert agg["fn_rate"] == pytest.approx(2 / 3)
    assert agg["extra"]["aggregate"] is True and agg["extra"]["buckets"] == ["<4", "4-6", "6-8"]
    assert agg is out[-1], "the aggregate is appended after the canonical buckets"


def test_fn_by_size_on_empty_input_is_none_safe():
    out = A.fn_by_size([])
    assert len(out) == 1 and out[0]["key"] == "<8"
    assert (out[0]["gt"], out[0]["fn"]) == (0, 0) and out[0]["fn_rate"] is None


# ------------------------------------------------------------------------------------------------
# fn_by_source
# ------------------------------------------------------------------------------------------------
def test_fn_by_source_groups_by_frame_domain(tmp_path):
    man = _manifest(tmp_path, [
        {"image": "/imgs/a.jpg", "location": "ml_3", "source": "eth_main"},
        {"image": "/imgs/b.jpg", "location": "ml_3", "source": "eth_main"},
        {"image": "/imgs/c.jpg", "location": "synthetic", "source": "synthetic"}])
    rows = [_gt("/imgs/a.jpg", 0, matched=False), _gt("/imgs/b.jpg", 0, matched=True),
            _gt("/imgs/c.jpg", 0, matched=False)]
    out = A.fn_by_source(rows, man)
    assert {r["key"]: (r["gt"], r["fn"]) for r in out} == {"eth_main": (2, 1), "synthetic": (1, 1)}
    assert out[0]["key"] == "eth_main" and out[0]["fn_rate"] == 0.5
    assert out[0]["extra"]["field"] == "source"


# ------------------------------------------------------------------------------------------------
# recoverable_fn_profile
# ------------------------------------------------------------------------------------------------
def test_recoverable_fn_profile_counts_both_directions_and_profiles(tmp_path):
    man = _manifest(tmp_path, [
        {"image": "/imgs/a.jpg", "location": "ml_3", "source": "eth_main"},
        {"image": "/imgs/b.jpg", "location": "iphone", "source": "eth_iphone"},
        {"image": "/imgs/c.jpg", "location": "synthetic", "source": "synthetic"},
        {"image": "/imgs/d.jpg", "location": "ml_3", "source": "eth_main"}])
    roots = ["/imgs/a.jpg", "/imgs/b.jpg", "/imgs/c.jpg", "/imgs/d.jpg"]
    eth = [_gt(roots[0], 0, "8-12", True), _gt(roots[0], 1, "8-12", True), _gt(roots[0], 2, "8-12", False),
           _gt(roots[1], 0, "16-24", False), _gt(roots[2], 0, "8-12", False), _gt(roots[3], 0, "8-12", True)]
    v1 = [_gt(roots[0], 0, "8-12", False), _gt(roots[0], 1, "8-12", False), _gt(roots[0], 2, "8-12", False),
          _gt(roots[1], 0, "16-24", True), _gt(roots[2], 0, "8-12", False), _gt(roots[3], 0, "8-12", True)]
    out = A.recoverable_fn_profile(eth, v1, man)
    assert (out["common_gt"], out["eth_tp_total"], out["v1_tp_total"]) == (6, 3, 2)
    assert (out["v1_fn_total"], out["eth_fn_total"]) == (4, 3)
    rec = out["recoverable"]
    assert rec["count"] == 2, "ETH hits /imgs/a#0,#1 which V1 misses"
    assert rec["fn_rate_of_eth_tp"] == pytest.approx(2 / 3)
    assert rec["share_of_v1_fn"] == pytest.approx(2 / 4)
    assert rec["size_profile"] == {"8-12": 2}
    assert rec["size_denominator"] == {"8-12": 3}, "denominator is the ETH-hit set, not the recoverable subset"
    assert rec["size_fn_rate"]["8-12"] == pytest.approx(2 / 3)
    assert rec["location_profile"] == {"ml_3": 2}
    assert rec["location_denominator"] == {"ml_3": 3}, "denominator is the ETH-hit set, not the V1 FN set"
    assert rec["location_fn_rate"]["ml_3"] == pytest.approx(2 / 3)
    sym = out["symmetric"]
    assert sym["count"] == 1, "V1 hits /imgs/b#0 which ETH misses"
    assert sym["fn_rate_of_v1_tp"] == pytest.approx(1 / 2)
    assert sym["share_of_eth_fn"] == pytest.approx(1 / 3)
    assert sym["size_profile"] == {"8-12": 0, "16-24": 1}, "canonical order, zero-filled from the hit set"
    assert sym["size_fn_rate"]["16-24"] == pytest.approx(1.0)
    assert sym["size_fn_rate"]["8-12"] == 0.0
    assert sym["location_profile"] == {"iphone": 1, "ml_3": 0}, "zero-filled from the V1-hit set"
    assert sym["location_denominator"] == {"iphone": 1, "ml_3": 1}


def test_recoverable_fn_profile_location_denominator_is_the_hit_set(location_manifest):
    """/imgs/d.jpg is hit by both models, so it counts as a location denominator but never as a recoverable FN."""
    roots = ["/imgs/a.jpg", "/imgs/d.jpg"]
    eth = [_gt(roots[0], 0, "8-12", True), _gt(roots[1], 0, "8-12", True)]
    v1 = [_gt(roots[0], 0, "8-12", False), _gt(roots[1], 0, "8-12", True)]
    rec = A.recoverable_fn_profile(eth, v1, location_manifest)["recoverable"]
    assert rec["count"] == 1 and rec["location_profile"] == {"ml_3": 1}
    assert rec["location_denominator"] == {"ml_3": 2}
    assert rec["location_fn_rate"] == {"ml_3": 0.5}


def test_recoverable_fn_profile_is_none_safe_without_common_gt():
    out = A.recoverable_fn_profile([], [], None)
    assert out["common_gt"] == 0 and out["recoverable"]["count"] == 0
    assert out["recoverable"]["fn_rate_of_eth_tp"] is None
    assert out["recoverable"]["size_profile"] == {} and out["recoverable"]["location_profile"] == {}
    assert out["symmetric"]["fn_rate_of_v1_tp"] is None


# ------------------------------------------------------------------------------------------------
# rates, report assembly and CSV shape
# ------------------------------------------------------------------------------------------------
def test_fn_rate_is_none_safe_when_gt_is_zero():
    assert A._rate(0, 0) is None
    assert A._rate(0, 5) == 0.0 and A._rate(4, 8) == 0.5
    assert A._row("empty", 0, 0)["fn_rate"] is None
    report = A.build_report([], [], None, {"conf_thr": 0.25})
    assert report["gt_total"] == 0 and report["fn_total"] == 0 and report["fn_rate"] is None
    assert "no GT rows" in report["headline"]
    assert all(r["fn_rate"] is None for r in A.flatten_report(report))


def test_fn_rate_reuses_the_imported_statistics_helpers():
    """Rates carry the shared CI helpers: statistics are imported from the errors module, never re-implemented."""
    rows = [_gt("/i.jpg", 0, "8-12", False), _gt("/i.jpg", 1, "8-12", True)]
    report = A.build_report(rows, [], None)
    assert report["fn_total_ci95"] == A.err.poisson_ci95(1)
    assert report["fn_rate_ci95"] == A.err.wilson_ci95(1, 2)
    assert report["fn_rate_ci95"][0] < 0.5 < report["fn_rate_ci95"][1]
    strata = A.fn_by_location(rows, None)
    assert strata[0]["fn_rate_ci95"] == A.err.wilson_ci95(1, 2)
    assert A.build_report([], [], None)["fn_rate_ci95"] == [None, None]


def test_build_report_exposes_the_four_blocks_and_echoes_the_inputs(location_manifest):
    rows = [_gt("/imgs/a.jpg", 0, "8-12", False), _gt("/imgs/b.jpg", 0, "6-8", True)]
    eth = [_gt("/imgs/a.jpg", 0, "8-12", True), _gt("/imgs/b.jpg", 0, "6-8", True)]
    inputs = {"v1_dump": "/dumps/v1.json", "iou_thr": 0.5, "conf_thr": 0.25}
    report = A.build_report(rows, eth, location_manifest, inputs)
    assert set(report) >= {"headline", "inputs", "by_location", "by_size", "by_source", "recoverable_fn"}
    assert (report["gt_total"], report["fn_total"], report["tp_total"]) == (2, 1, 1)
    assert report["fn_rate"] == pytest.approx(0.5)
    assert report["inputs"] == inputs
    assert "misses 1 of 2 GT" in report["headline"] and "ml_3" in report["headline"]
    assert report["recoverable_fn"]["recoverable"]["count"] == 1


def test_flatten_report_and_write_fn_csv_emit_the_long_format(location_manifest, tmp_path):
    rows = [_gt("/imgs/a.jpg", 0, "8-12", False), _gt("/imgs/b.jpg", 0, "6-8", True)]
    eth = [_gt("/imgs/a.jpg", 0, "8-12", True), _gt("/imgs/b.jpg", 0, "6-8", True)]
    report = A.build_report(rows, eth, location_manifest, {"conf_thr": 0.25})
    flat = A.flatten_report(report)
    assert [list(r) for r in flat] == [list(A.CSV_COLUMNS)] * len(flat)
    assert {"summary", "by_location", "by_size", "by_source", "recoverable", "recoverable_size",
            "recoverable_location"} <= {r["scope"] for r in flat}
    summary = [r for r in flat if r["scope"] == "summary"][0]
    assert (summary["key"], summary["gt"], summary["fn"]) == ("val|eth_unseen", 2, 1)
    json.loads(summary["extra"])
    path = A.write_fn_csv(tmp_path / "out.csv", flat)
    got = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(got[0].keys()) == list(A.CSV_COLUMNS)
    assert len(got) == len(flat)
    assert [r for r in got if r["scope"] == "summary"][0]["fn_rate"] == "0.5"


def test_main_writes_the_two_artifacts_end_to_end(tmp_path):
    """Full CLI wiring on a 2-image fixture: 2 GT, one exact hit and one miss."""
    from PIL import Image
    imgs = []
    for name in ("a.jpg", "b.jpg"):
        p = tmp_path / name
        Image.new("RGB", (100, 100), (10, 20, 30)).save(p)
        lab = tmp_path / (name + ".txt")
        lab.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")     # GT box 40,40,60,60
        imgs.append((str(p), str(lab)))
    man = _manifest(tmp_path, [
        {"image": imgs[0][0], "label": imgs[0][1], "split": "val", "location": "loc_a", "source": "eth_main"},
        {"image": imgs[1][0], "label": imgs[1][1], "split": "val", "location": "loc_b", "source": "eth_main"}])
    leak = tmp_path / "leak.csv"
    with leak.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "image", "leakage_class"])
        w.writeheader()
        for img, _ in imgs:
            w.writerow({"set": "val|eth_unseen", "image": img,
                        "leakage_class": "B_same_location_not_trained"})
    v1 = {"dets": {imgs[0][0]: [{"box": [40, 40, 60, 60], "conf": 0.9}], imgs[1][0]: []}}
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(v1), encoding="utf-8")
    out_json, out_csv = tmp_path / "fn.json", tmp_path / "fn.csv"

    rc = A.main(["--v1-dump", str(v1_path), "--eth-dump", str(v1_path), "--dataset-manifest", str(man),
                 "--leakage-inventory", str(leak), "--out-json", str(out_json), "--out-csv", str(out_csv)])
    assert rc == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert (report["gt_total"], report["fn_total"], report["tp_total"]) == (2, 1, 1)
    assert report["inputs"]["conf_thr"] == 0.25 and report["inputs"]["iou_thr"] == 0.5
    assert report["inputs"]["v1_dump"] == str(v1_path.resolve())
    got = {r["key"]: (r["gt"], r["fn"]) for r in report["by_location"]}
    assert got == {"loc_b": (1, 1), "loc_a": (1, 0)}
    assert list(csv.DictReader(open(out_csv, encoding="utf-8")))[0]["scope"] == "summary"


# ------------------------------------------------------------------------------------------------
# real-artifact consistency (skips until the analysis has been generated)
# ------------------------------------------------------------------------------------------------
def _canonical_row(model="eth_only_v1_best"):
    rows = list(csv.DictReader(ETH_VS_V1.open(encoding="utf-8")))
    return [r for r in rows if r["model"] == model and r["set"] == "val|eth_unseen"][0]


def test_real_artifact_reproduces_the_canonical_gt_fn_counts():
    if not FN_JSON.is_file() or not ETH_VS_V1.is_file():
        pytest.skip("FN analysis artifact not built yet")
    report = json.loads(FN_JSON.read_text(encoding="utf-8"))
    assert report["gt_total"] == 495 and report["fn_total"] == 414
    assert report["tp_total"] == 81
    assert report["fn_rate"] == pytest.approx(414 / 495)
    canonical = _canonical_row()
    assert int(canonical["GT"]) == report["gt_total"]
    assert int(canonical["GT"]) - int(canonical["TP"]) == report["fn_total"], "FN must equal GT - TP"
    assert int(canonical["FN"]) == report["fn_total"]
    assert report["inputs"]["conf_thr"] == 0.25 and report["inputs"]["iou_thr"] == 0.5


def test_real_artifact_size_buckets_match_the_canonical_table():
    if not FN_JSON.is_file() or not ETH_VS_V1.is_file():
        pytest.skip("FN analysis artifact not built yet")
    report = json.loads(FN_JSON.read_text(encoding="utf-8"))
    canonical = _canonical_row()
    by_key = {r["key"]: r for r in report["by_size"]}
    for mine, col in (("<4", "<4"), ("4-6", "4_6"), ("6-8", "6_8"), ("8-12", "8_12"),
                      ("12-16", "12_16"), ("32-64", "32_64"), (">64", ">64")):
        assert by_key[mine]["gt"] == int(canonical["GT_" + col]), "GT mismatch for bucket " + mine
        assert by_key[mine]["fn"] == int(canonical["GT_" + col]) - int(canonical["TP_" + col])
    merged = by_key["16-24"]["gt"] + by_key["24-32"]["gt"]
    assert merged == int(canonical["GT_16_32"]) == 77
    assert by_key["<8"]["gt"] == int(canonical["GT_<8"]) == 105
    assert by_key["<8"]["fn"] == 105 - int(canonical["TP_<8"]) == 97


def test_real_artifact_recoverable_block_matches_the_canonical_eth_row():
    if not FN_JSON.is_file() or not ETH_VS_V1.is_file():
        pytest.skip("FN analysis artifact not built yet")
    rec = json.loads(FN_JSON.read_text(encoding="utf-8"))["recoverable_fn"]
    eth = _canonical_row("eth_official")
    assert rec["eth_tp_total"] == int(eth["TP"]) == 269
    assert rec["v1_tp_total"] == 81 and rec["v1_fn_total"] == 414 and rec["eth_fn_total"] == 226
    assert rec["recoverable"]["count"] == 194
    assert rec["recoverable"]["count"] <= rec["eth_tp_total"]
    assert rec["recoverable"]["count"] <= rec["v1_fn_total"]
    assert rec["symmetric"]["count"] == 6
    assert rec["recoverable"]["fn_rate_of_eth_tp"] == pytest.approx(194 / 269)
    assert sum(rec["recoverable"]["size_profile"].values()) == 194
    assert sum(rec["recoverable"]["location_profile"].values()) == 194
    assert sum(rec["symmetric"]["size_profile"].values()) == 6


def test_real_artifact_csv_is_long_format_and_consistent_with_the_json():
    if not FN_CSV.is_file():
        pytest.skip("FN analysis artifact not built yet")
    rows = list(csv.DictReader(FN_CSV.open(encoding="utf-8")))
    assert list(rows[0].keys()) == list(A.CSV_COLUMNS)
    assert all(json.loads(r["extra"]) is not None for r in rows)
    summary = [r for r in rows if r["scope"] == "summary" and r["key"] == "val|eth_unseen"][0]
    assert (summary["gt"], summary["fn"]) == ("495", "414")
    assert float(summary["fn_rate"]) == pytest.approx(414 / 495)
    by_key = {r["key"]: r for r in rows if r["scope"] == "by_size"}
    assert by_key["<8"]["fn"] == "97" and by_key["<4"]["fn"] == "24"
