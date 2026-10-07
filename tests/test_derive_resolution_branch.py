"""Tests for tools/derive_resolution_branch.py (applies the frozen branch rule to a probe run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import derive_resolution_branch as D  # noqa: E402

DECISION = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_v1_decision.json"
LOCAL_PROBE = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_local_v1_probe.json"


def _gt(image, bucket, eq640, best_iou, best_conf, pairs):
    return {"image": image, "gt_index": 0, "eq640": eq640, "bucket": bucket, "tiny": True,
            "matched_op": best_iou >= 0.5 and best_conf >= 0.25, "best_iou_weak": best_iou,
            "best_conf_at_best_iou": best_conf, "n_dets_iou30": 0, "n_dets": len(pairs),
            "n_dets_op": sum(1 for c, _ in pairs if c >= 0.25)}


def _probe(resolutions):
    return {"weights_sha256": "x", "env": {}, "resolutions": resolutions}


def _write(tmp_path, probe):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(probe), encoding="utf-8")
    return p


def _block(gts):
    pic = {}
    for g in gts:
        pic.setdefault(g["image"], []).append({"gt_index": g["gt_index"],
                                               "pairs": [[g["best_conf_at_best_iou"], g["best_iou_weak"]]]})
    return {"gt": gts, "per_image_iou_conf": pic}


def _run(tmp_path, resolutions):
    out = tmp_path / "decision.json"
    rc = D.main(["--json", str(_write(tmp_path, _probe(resolutions))), "--tag", "t", "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_rows_from_probe_maps_fields_and_scales_net_px():
    probe = _probe({"1024": _block([_gt("a.jpg", "<4", 3.0, 0.0, 0.0, [])]),
                    "1536": _block([_gt("a.jpg", "<4", 3.0, 0.7, 0.6, [])])})
    rows = D.rows_from_probe(probe)
    assert len(rows) == 2
    r1 = [r for r in rows if r["imgsz"] == 1024][0]
    r2 = [r for r in rows if r["imgsz"] == 1536][0]
    assert r1["net_px"] == pytest.approx(1.6 * 3.0)
    assert r2["net_px"] == pytest.approx(1.6 * 3.0 * 1536 / 1024.0)
    assert r1["matched_op"] is False and r2["matched_op"] is True
    assert r1["n_candidates"] == 1 and r2["n_candidates"] == 1


def test_branch_is_p2_stride_4_when_sub_stride_stays_silent(tmp_path):
    # 6-8 responds at both sizes, the <4/4-6 population never does -> grid is the binding constraint
    gts = lambda R: ([_gt("m%d.jpg" % i, "6-8", 7.0, 0.8, 0.7, [[0.7, 0.8]]) for i in range(9)]
                     + [_gt("s%d.jpg" % i, "4-6", 5.0, 0.0, 0.0, []) for i in range(37)])
    res = {str(R): _block(gts(R)) for R in (1024, 1536)}
    payload = _run(tmp_path, res)
    assert payload["decision"]["branch"] == "P2_STRIDE_4"
    assert payload["decision"]["sub_stride_weak_ge_0.1"] == {"1024": 0, "1536": 0}
    assert payload["decision"]["mid_bucket_weak_ge_0.1"] == {"1024": 9, "1536": 9}


def test_branch_is_higher_resolution_when_sub_stride_wakes_up(tmp_path):
    res = {"1024": _block([_gt("s%d.jpg" % i, "4-6", 5.0, 0.2 if i > 2 else 0.0, 0.1, []) for i in range(5)]),
           "1536": _block([_gt("s%d.jpg" % i, "4-6", 5.0, 0.6, 0.5, [[0.5, 0.6]]) for i in range(5)])}
    payload = _run(tmp_path, res)
    assert payload["decision"]["branch"] == "HIGHER_RESOLUTION_TRAINING"
    assert payload["decision"]["sub_stride_weak_ge_0.1"]["1536"] - payload["decision"]["sub_stride_weak_ge_0.1"]["1024"] >= 3


def test_branch_is_mid_bucket_weighting_when_only_6_8_gains(tmp_path):
    def gts(R):
        # at 1024 the mid bucket is below the 0.1 weak mark, at 1536 it clears it by a wide margin
        return ([_gt("m%d.jpg" % i, "6-8", 7.0, 0.05 if R == 1024 else 0.8, 0.04 if R == 1024 else 0.7, [])
                 for i in range(6)]
                + [_gt("s%d.jpg" % i, "<4", 3.0, 0.0, 0.0, []) for i in range(10)])
    res = {str(R): _block(gts(R)) for R in (1024, 1536)}
    payload = _run(tmp_path, res)
    assert payload["decision"]["branch"] == "P2_PLUS_MID_BUCKET_WEIGHTING"
    assert payload["decision"]["mid_bucket_weak_ge_0.1"]["1536"] - payload["decision"]["mid_bucket_weak_ge_0.1"]["1024"] >= 3
    assert payload["decision"]["sub_stride_weak_ge_0.1"]["1536"] <= 1


def test_committed_decision_reproduces_from_the_committed_probe(tmp_path):
    if not DECISION.is_file() or not LOCAL_PROBE.is_file():
        pytest.skip("resolution-response artifacts not built yet")
    committed = json.loads(DECISION.read_text(encoding="utf-8"))
    assert committed["tag"] == "local_v1" and committed["n_gt"] == 105
    assert committed["decision"]["branch"] == "P2_STRIDE_4"
    assert "compute_branch" in committed["rule_source"]
    out = tmp_path / "recomputed.json"
    assert D.main(["--json", str(LOCAL_PROBE), "--tag", "local_v1", "--out", str(out)]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["decision"] == committed["decision"]
    assert again["summaries"] == committed["summaries"]
    assert again["probe_sha256"] == committed["probe_sha256"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
