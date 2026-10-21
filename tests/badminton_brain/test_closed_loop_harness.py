#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-10: the pre-registered closed-loop HARNESS - refusal behaviour and determinism.

These tests pin the ONE rule that matters about this harness: it is a harness,
so it refuses to produce a hit rate without a racket collision proxy and a
calibration, and even a complete mock run never claims a real hit rate.

Run:
    python -m pytest tests/badminton_brain/test_closed_loop_harness.py -q
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.badminton_brain.harness.closed_loop import (  # noqa: E402
    SCENARIOS, HarnessConfig, RacketCollisionProxy, run_closed_loop,
)
from src.badminton_brain.reachability import Calibration  # noqa: E402


def verified():
    return Calibration(court_from_base=np.eye(4), verified=True)


def config(scenario="LOW_SPEED_INCOMING", **kwargs):
    kwargs.setdefault("calibration", verified())
    kwargs.setdefault("racket_proxy", RacketCollisionProxy())
    return HarnessConfig(scenario=scenario, **kwargs)


class RefusalTests(unittest.TestCase):
    def test_refuses_without_the_racket_collision_proxy(self):
        report = run_closed_loop(config(racket_proxy=None))
        self.assertEqual(report["status"], "REFUSED")
        self.assertIn("RACKET_COLLISION_PROXY_MISSING", report["refusal_reason"])
        self.assertIsNone(report["hit_rate"])
        self.assertIsNone(report["synthetic_contact_rate"])
        self.assertIsNone(report["contacts"])

    def test_refuses_without_calibration(self):
        report = run_closed_loop(config(calibration=None))
        self.assertEqual(report["status"], "REFUSED")
        self.assertIn("CALIBRATION_MISSING", report["refusal_reason"])
        self.assertIsNone(report["hit_rate"])

    def test_refuses_with_an_unverified_calibration(self):
        report = run_closed_loop(config(calibration=Calibration()))
        self.assertEqual(report["status"], "REFUSED")
        self.assertIn("CALIBRATION_MISSING", report["refusal_reason"])

    def test_refuses_when_everything_is_missing(self):
        report = run_closed_loop(HarnessConfig(scenario="FIXED_BALL"))
        self.assertEqual(report["status"], "REFUSED")
        self.assertEqual(sorted(report["refusal_reason"]),
                         ["CALIBRATION_MISSING", "RACKET_COLLISION_PROXY_MISSING"])

    def test_refusal_is_json_serialisable(self):
        json.dumps(run_closed_loop(config(racket_proxy=None)), sort_keys=True)


class MockRunTests(unittest.TestCase):
    def test_every_scenario_runs_deterministically(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                first = run_closed_loop(config(scenario))
                second = run_closed_loop(config(scenario))
                self.assertEqual(json.dumps(first, sort_keys=True),
                                 json.dumps(second, sort_keys=True))
                self.assertEqual(first["status"], "MOCK_RUN_COMPLETE")
                self.assertEqual(first["evidence_class"], "SYNTHETIC-MOCK-ONLY")

    def test_never_claims_a_real_hit_rate(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                report = run_closed_loop(config(scenario))
                self.assertIsNone(report["hit_rate"])
                self.assertFalse(report["real_hit_rate_claimed"])
                self.assertIsNone(report["physics_engine"])
                self.assertIsNone(report["camera"])
                self.assertIsNone(report["robot"])
                if report["synthetic_contact_rate"] is not None:
                    self.assertLessEqual(report["synthetic_contact_rate"], 1.0)
                self.assertLessEqual(report["contacts"], report["executed_commands"])

    def test_scenarios_are_distinguishable(self):
        fixed = run_closed_loop(config("FIXED_BALL"))
        incoming = run_closed_loop(config("LOW_SPEED_INCOMING"))
        perturbed = run_closed_loop(config("PERTURBED"))
        self.assertNotEqual(fixed["ball_positions_m"][1],
                            incoming["ball_positions_m"][1])
        self.assertNotEqual(incoming["ball_positions_m"][1],
                            perturbed["ball_positions_m"][1])

    def test_perturbation_is_seeded(self):
        a = run_closed_loop(config("PERTURBED", seed=7))
        b = run_closed_loop(config("PERTURBED", seed=7))
        c = run_closed_loop(config("PERTURBED", seed=8))
        self.assertEqual(a["ball_positions_m"], b["ball_positions_m"])
        self.assertNotEqual(a["ball_positions_m"], c["ball_positions_m"])

    def test_fsm_report_is_embedded(self):
        report = run_closed_loop(config("FIXED_BALL"))
        self.assertIn("fsm_report", report)
        self.assertIn(report["fsm_report"]["state"], {"EXECUTE", "HOLD"})
        self.assertGreaterEqual(len(report["fsm_report"]["events"]), 1)

    def test_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            HarnessConfig(scenario="REAL_ROBOT")


if __name__ == "__main__":
    unittest.main()
