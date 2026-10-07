"""Tests for tools/analyze_remote_tiny_resolution.py (post-processing of the A6000 resolution probe)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_remote_tiny_resolution as D  # noqa: E402


def _pairs(*rows):
    return [{"gt_index": i, "pairs": p} for i, p in enumerate(rows)]


def _gt(image, gt_index, eq640, bucket, n_dets, n_dets_op, n_dets_iou30, best, best_conf):
    return {"image": image, "gt_index": gt_index, "eq640": eq640, "bucket": bucket, "tiny": eq640 < 8.0,
            "matched_op": best >= 0.5 and best_conf >= 0.25, "best_iou_weak": best,
            "best_conf_at_best_iou": best_conf, "n_dets_iou30": n_dets_iou30,
            "n_dets": n_dets, "n_dets_op": n_dets_op}


def _block(gt, iou_conf, *, cand, cand_op, e2e=39.69, inf=10.59, pre=4.68, post=1.57, vram=203.0,
           net_input=1024):
    return {"per_image": [], "gt": gt, "peak_vram_mib": vram,
            "latency_ms": {"median": e2e, "mean": e2e + 1.0, "min": e2e - 2, "max": e2e + 5,
                           "reps": 30, "batch": 1},
            "latency_speed_median_ms": {"preprocess": pre, "inference": inf, "postprocess": post},
            "pass_speed_median_ms": {"preprocess": pre, "inference": inf, "postprocess": post},
            "per_image_ms_from_pass": {"mean": e2e, "median": e2e}, "pass_total_s": 1.0,
            "img_per_s": 23.67, "n_candidates": cand, "n_candidates_op": cand_op,
            "net_input_px": [net_input, net_input], "per_image_iou_conf": iou_conf}


def _report(tmp_path, resolutions):
    rep = {"weights": "/w/best.pt", "weights_sha256": "deadbeef", "weights_bytes": 20344133,
           "root": "/d", "device": "0", "conf_floor": 0.01, "conf_op": 0.25, "nms_iou": 0.7,
           "max_det": 300, "matcher_thr": 0.5, "n_part_rows": 2, "n_images": 2, "missing": [],
           "picked": [], "resolutions": resolutions, "env": {"python": "3.13.2"},
           "crosscheck_vs_local_v2_1024": {"n_checked": 0, "mismatches": []}}
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    return p


def _part(tmp_path, rows):
    p = tmp_path / "part.csv"
    keys = ["basename", "rel_path", "bucket", "equiv_size_640", "net_px_1024",
            "v2_matched_op_1024", "v2_best_iou_weak_1024", "v2_best_conf_weak_1024"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _wire(monkeypatch, tmp_path, rep, part):
    monkeypatch.setattr(D, "JSON_PATH", rep)
    monkeypatch.setattr(D, "PART_PATH", part)
    monkeypatch.setattr(D, "OUT_DIR", tmp_path / "out")


def test_matched_at_requires_both_conf_and_iou():
    pairs = [[0.9, 0.2], [0.05, 0.9]]                   # never both at once for IoU 0.3/0.5
    assert D.matched_at(pairs, 0.25, 0.5) is False      # conf ok but IoU too low, IoU ok but conf too low
    assert D.matched_at(pairs, 0.01, 0.01) is True      # the second candidate clears a minimal IoU gate
    assert D.matched_at(pairs, 0.25, 0.3) is False
    assert D.matched_at(pairs, 0.9, 0.3) is False
    assert D.matched_at([[0.9, 0.4]], 0.25, 0.3) is True
    assert D.matched_at([], 0.01, 0.5) is False


def test_conf_grid_is_monotone():
    assert D.CONF_GRID == sorted(D.CONF_GRID)
    assert D.CONF_GRID[0] == 0.01 and D.CONF_GRID[-1] == 0.25


def test_main_writes_curve_and_net_px(monkeypatch, tmp_path, capsys):
    gt = [_gt("a.jpg", 0, 3.3, "<4", 1, 1, 1, 0.665, 0.325),
          _gt("b.jpg", 0, 7.5, "6-8", 1, 0, 1, 0.60, 0.130)]
    block = _block(gt, {"a.jpg": _pairs([[0.325, 0.665]]), "b.jpg": _pairs([[0.130, 0.60]])},
                   cand=2, cand_op=1)
    part = _part(tmp_path, [
        {"basename": "a.jpg", "rel_path": "loc_hard/images/train/a.jpg", "bucket": "<4",
         "equiv_size_640": 3.3, "net_px_1024": 5.28, "v2_matched_op_1024": "False",
         "v2_best_iou_weak_1024": "0.0", "v2_best_conf_weak_1024": ""},
        {"basename": "b.jpg", "rel_path": "loc_hard/images/train/b.jpg", "bucket": "6-8",
         "equiv_size_640": 7.5, "net_px_1024": 12.0, "v2_matched_op_1024": "True",
         "v2_best_iou_weak_1024": "0.9", "v2_best_conf_weak_1024": "0.7"}])
    _wire(monkeypatch, tmp_path, _report(tmp_path, {"1024": block}), part)
    assert D.main() == 0

    gt_csv = tmp_path / "out" / "resolution_response_remote_v1_gt.csv"
    rows = list(csv.DictReader(gt_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    small = [r for r in rows if r["bucket"] == "<4"][0]
    big = [r for r in rows if r["bucket"] == "6-8"][0]
    assert small["hit_iou0.5_conf0.25"] == "True"
    assert small["hit_iou0.5_conf0.2"] == "True"
    assert small["hit_iou0.3_conf0.25"] == "True"
    assert small["net_px"] == "5.28"                      # 1.6 * 3.3 * 1024/1024
    assert small["ref_v2_matched_op_1024"] == "False"
    # b.jpg only has a 0.13-conf candidate with IoU 0.60 -> hit up to conf 0.10, miss from 0.15 on
    assert big["hit_iou0.5_conf0.1"] == "True"          # column names come from "%s" % 0.10 -> 0.1
    assert big["hit_iou0.5_conf0.15"] == "False"
    assert big["hit_iou0.5_conf0.25"] == "False"
    assert big["ref_v2_matched_op_1024"] == "True"

    summary = list(csv.DictReader((tmp_path / "out" / "resolution_response_remote_v1_summary.csv").open(encoding="utf-8")))
    assert len(summary) == 1
    s = summary[0]
    assert s["n_gt"] == "2" and s["op_confident_candidates"] == "1" and s["op_tp"] == "1"
    assert s["op_unmatched"] == "0" and s["images_with_no_confident_candidate"] == "1"
    assert s["hit_iou0.5_conf0.1"] == "2" and s["hit_iou0.5_conf0.15"] == "1"
    assert s["n_best_iou_ge_0.5"] == "2" and s["n_best_iou_ge_0.3"] == "2" and s["n_best_iou_lt_0.01"] == "0"
    assert float(s["latency_e2e_median_ms"]) == pytest.approx(39.69)
    assert float(s["speed_inference_median_ms"]) == pytest.approx(10.59)
    assert s["net_input_px"] == "1024"
    out = capsys.readouterr().out
    # the fixture deliberately disagrees with the carried local references, so agreement must be reported as 0/2
    assert "best-IoU agreement within 0.02: 0/2" in out
    assert "matched@0.25/0.5  reference=1  probe=1" in out


def test_main_rejects_candidate_count_mismatch(monkeypatch, tmp_path):
    gt = [_gt("a.jpg", 0, 7.0, "6-8", 3, 0, 0, 0.0, 0.0)]
    block = _block(gt, {"a.jpg": _pairs([[0.5, 0.9]])}, cand=1, cand_op=0)
    part = _part(tmp_path, [{"basename": "a.jpg", "rel_path": "l/images/train/a.jpg", "bucket": "6-8",
                             "equiv_size_640": 7.0, "net_px_1024": 11.2}])
    _wire(monkeypatch, tmp_path, _report(tmp_path, {"1024": block}), part)
    with pytest.raises(AssertionError):
        D.main()


def test_main_reports_per_resolution_latency_split(monkeypatch, tmp_path, capsys):
    gt = [_gt("a.jpg", 0, 6.2, "6-8", 0, 0, 0, 0.0, 0.0)]
    part = _part(tmp_path, [{"basename": "a.jpg", "rel_path": "l/images/train/a.jpg", "bucket": "6-8",
                             "equiv_size_640": 6.2, "net_px_1024": 9.9}])
    res = {"1024": _block(gt, {"a.jpg": _pairs([])}, cand=0, cand_op=0, e2e=39.69, inf=10.59, pre=4.68, post=1.57, vram=203.0),
           "1536": _block(gt, {"a.jpg": _pairs([])}, cand=0, cand_op=0, e2e=51.77, inf=12.38, pre=9.73,
                                 post=0.86, vram=364.0, net_input=1536)}
    _wire(monkeypatch, tmp_path, _report(tmp_path, res), part)
    assert D.main() == 0
    out = capsys.readouterr().out
    assert "preprocess=9.73 inference=12.38 postprocess=0.86" in out
    summary = list(csv.DictReader((tmp_path / "out" / "resolution_response_remote_v1_summary.csv").open(encoding="utf-8")))
    assert [s["imgsz"] for s in summary] == ["1024", "1536"]
    assert [s["images_with_no_confident_candidate"] for s in summary] == ["1", "1"]
    assert [s["net_input_px"] for s in summary] == ["1024", "1536"]
    assert "image_had_none= 1" in out


def test_real_probe_json_and_gt_csv_are_consistent():
    """Artifact test: the committed probe JSON and the committed GT CSV must describe the same run."""
    probe = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_remote_v1_probe.json"
    gt_csv = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_remote_v1_gt.csv"
    if not probe.is_file() or not gt_csv.is_file():
        pytest.skip("resolution-response artifacts not built yet")
    rep = json.loads(probe.read_text(encoding="utf-8"))
    assert rep["weights_sha256"] == "3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7"
    assert sorted(rep["resolutions"]) == ["1024", "1280", "1536"]
    rows = list(csv.DictReader(gt_csv.open(encoding="utf-8")))
    assert len(rows) == 3 * 20 == 60
    for R in ("1024", "1280", "1536"):
        block = rep["resolutions"][R]
        rs = [r for r in rows if r["imgsz"] == R]
        assert len(rs) == 20
        assert sum(1 for r in rs if r["hit_iou0.5_conf0.25"] == "True") == sum(
            1 for g in block["gt"] if g["best_iou_weak"] >= 0.5 and g["best_conf_at_best_iou"] >= 0.25)
        assert sum(int(r["n_candidates"]) for r in rs) == block["n_candidates"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
