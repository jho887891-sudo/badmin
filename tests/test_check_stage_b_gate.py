#!/usr/bin/env python3
"""Tests for tools/check_stage_b_gate.py (gate logic v2).

These are pure-function tests: they never touch a live run, a checkpoint or a training process.
They pin the three fixes:
  H1 collapse  - a catastrophic metric collapse FAILs even when box_loss does not rise
  H2 memory    - MemAvailable / swap IO / cgroup participate in the verdict; SwapFree==0 alone does not
  H3 LR limits - limits come from the design targets in lr_config.json x1.10

Run:  python tests/test_check_stage_b_gate.py -v   (or: pytest -q tests/test_check_stage_b_gate.py)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_stage_b_gate as gate  # noqa: E402

BASE_MAP50 = 0.871
BASE_RECALL = 0.76012
STAGE_A_BEST = ("/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/"
                "gate6A_imgsz1024_b16_w8_20260926-083331/weights/best.pt")


def healthy_series():
    return {"epoch": [1, 2, 3, 4, 5],
            "box_loss": [1.385, 1.358, 1.354, 1.347, 1.347],
            "recall": [0.746, 0.700, 0.711, 0.684, 0.684],
            "map50": [0.849, 0.800, 0.793, 0.783, 0.825]}


def healthy_lrcfg():
    groups = []
    for _ in range(4):
        groups.append({"initial_lr": 0.003, "lr": 0.0015, "group": len(groups)})
        groups.append({"initial_lr": 0.001, "lr": 0.0005, "group": len(groups)})
    return {"backbone_unfrozen": True, "lr0": 0.001, "param_groups": groups}


def healthy_lrs():
    return {ep: {"start": [0.0] * 8, "end": [0.003, 0.001] * 4, "max": [0.00291, 0.00097] * 4}
            for ep in range(1, 6)}


def healthy_mem():
    return {"memavailable_minmax_GB": [21.6, 30.9], "swapfree_minmax_GB": [0.0, 0.53],
            "trainer_rss_max_GB": 9.59, "cgroup_frac_max": None, "swap_kbps_max": 1132.3}


def run_eval(series=None, lrs=None, mem=None, lrcfg=None, init=STAGE_A_BEST, nan=False,
             required=5, override_normal=None, override_head=None):
    # NOTE: explicit None checks - an intentionally empty lrcfg={} must reach evaluate() as-is
    return gate.evaluate(healthy_series() if series is None else series,
                         healthy_lrs() if lrs is None else lrs,
                         healthy_mem() if mem is None else mem,
                         healthy_lrcfg() if lrcfg is None else lrcfg,
                         init, nan=nan,
                         baseline_map50=BASE_MAP50, baseline_recall=BASE_RECALL,
                         required_epochs=required, override_normal_lr=override_normal,
                         override_head_lr=override_head, label="unit", run="<synthetic>")


class HealthyRun(unittest.TestCase):
    def test_healthy_run_passes(self):
        out = run_eval()
        self.assertEqual(out["verdict"], "PASS_CONTINUE_STAGE_B", out["reasons"])
        self.assertFalse(out["memory_fail"])
        self.assertEqual(out["lr_violations"], [])

    def test_swapfree_zero_alone_does_not_fail(self):
        mem = healthy_mem()
        mem["swapfree_minmax_GB"] = [0.0, 0.0]
        out = run_eval(mem=mem)
        self.assertEqual(out["verdict"], "PASS_CONTINUE_STAGE_B", out["reasons"])
        self.assertFalse(out["memory_fail"])

    def test_not_enough_epochs_is_unknown(self):
        out = run_eval(required=9)
        self.assertEqual(out["verdict"], "UNKNOWN_NEED_MORE_EVIDENCE")


class TestA_CatastrophicMetricCollapse(unittest.TestCase):
    def test_recall_collapse_fails_without_box_loss_rise(self):
        series = {"epoch": [1, 2, 3], "box_loss": [1.30, 1.28, 1.25],
                  "recall": [0.70, 0.35, 0.15], "map50": [0.82, 0.50, 0.20]}
        out = run_eval(series=series, required=3)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(out["catastrophic_metric_collapse"])
        self.assertLess(out["max_consecutive_box_loss_rise"], gate.TREND_RISE_EPOCHS)

    def test_map50_collapse_alone_fails(self):
        series = {"epoch": [1, 2, 3], "box_loss": [1.30, 1.29, 1.28],
                  "recall": [0.75, 0.74, 0.73], "map50": [0.82, 0.50, 0.20]}
        out = run_eval(series=series, required=3)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(out["catastrophic_metric_collapse"])
        self.assertFalse(out["trend_collapse"])


class TestB_MemoryFail(unittest.TestCase):
    def test_low_memavailable_fails(self):
        mem = healthy_mem()
        mem["memavailable_minmax_GB"] = [3.0, 12.0]
        out = run_eval(mem=mem)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(out["memory_fail"])
        self.assertTrue(any("MemAvailable" in r for r in out["memory_reasons"]), out["memory_reasons"])

    def test_high_swap_io_fails(self):
        mem = healthy_mem()
        mem["swap_kbps_max"] = 31.2 * 1024.0
        out = run_eval(mem=mem)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(any("swap IO" in r for r in out["memory_reasons"]), out["memory_reasons"])

    def test_cgroup_over_90pct_fails(self):
        mem = healthy_mem()
        mem["cgroup_frac_max"] = 0.931
        out = run_eval(mem=mem)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(any("cgroup" in r for r in out["memory_reasons"]), out["memory_reasons"])


class TestC_LRLimit(unittest.TestCase):
    def test_normal_group_above_target_fails(self):
        lrs = healthy_lrs()
        lrs[3]["max"] = [0.0030, 0.0015] * 4          # normal group at 0.0015 > 0.001 x 1.10
        out = run_eval(lrs=lrs)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(out["lr_violations"], out["reasons"])
        self.assertIn("0.0011", json_dumps(out["lr_violations"]))

    def test_head_group_above_target_fails(self):
        lrs = healthy_lrs()
        lrs[2]["max"] = [0.0040, 0.0010] * 4          # head group at 0.004 > 0.003 x 1.10
        out = run_eval(lrs=lrs)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])

    def test_limits_come_from_lr_config_targets(self):
        out = run_eval()
        lim = out["lr_limits"]
        self.assertAlmostEqual(lim["target_normal"], 0.001, places=8)
        self.assertAlmostEqual(lim["target_head"], 0.003, places=8)
        self.assertAlmostEqual(lim["normal_limit"], 0.0011, places=8)
        self.assertAlmostEqual(lim["head_limit"], 0.0033, places=8)
        self.assertIn("lr_config.json", lim["source"])

    def test_warmup_observed_lr_is_not_used_as_target(self):
        # param_groups carry both the live (warmup) lr and the design initial_lr; only initial_lr counts
        cfg = healthy_lrcfg()
        for g in cfg["param_groups"]:
            g["lr"] = 0.00001                     # bogus live value
        out = run_eval(lrcfg=cfg)
        self.assertAlmostEqual(out["lr_limits"]["target_head"], 0.003, places=8)
        self.assertAlmostEqual(out["lr_limits"]["target_normal"], 0.001, places=8)

    def test_fallback_limits_when_no_config(self):
        out = run_eval(lrcfg={})
        lim = out["lr_limits"]
        self.assertEqual(lim["source"], "plan B fallback constants")
        self.assertAlmostEqual(lim["normal_limit"], 0.0011, places=8)
        self.assertAlmostEqual(lim["head_limit"], 0.0033, places=8)


class TrendCollapse(unittest.TestCase):
    def test_box_loss_rise_plus_moderate_metric_drop_fails(self):
        series = {"epoch": [1, 2, 3, 4, 5], "box_loss": [1.30, 1.32, 1.34, 1.36, 1.38],
                  "recall": [0.72, 0.68, 0.64, 0.62, 0.60], "map50": [0.84, 0.82, 0.80, 0.79, 0.78]}
        out = run_eval(series=series)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])
        self.assertTrue(out["trend_collapse"])
        self.assertFalse(out["catastrophic_metric_collapse"])


class HardFail(unittest.TestCase):
    def test_nan_fails(self):
        out = run_eval(nan=True)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])

    def test_wrong_init_fails(self):
        out = run_eval(init="/somewhere/else/best.pt")
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])

    def test_backbone_frozen_fails(self):
        cfg = healthy_lrcfg()
        cfg["backbone_unfrozen"] = False
        out = run_eval(lrcfg=cfg)
        self.assertEqual(out["verdict"], "FAIL_STOP_STAGE_B", out["reasons"])


def json_dumps(x):
    import json
    return json.dumps(x)


if __name__ == "__main__":
    unittest.main(verbosity=2)