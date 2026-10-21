#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-08: latency budget accounting and stale-frame rejection tests.

The policy under test is pre-registered (0.100 s max source-frame age); the tests
pin both sides of the boundary so it cannot drift silently.

Run:
    python -m pytest tests/badminton_brain/test_latency_budget.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.badminton_brain.timing.latency_budget import (  # noqa: E402
    END_TO_END_BUDGET_S,
    MAX_SOURCE_FRAME_AGE_S,
    REJECT_AGE_EXCEEDED,
    REJECT_FRAME_REUSE,
    REJECT_FUTURE_FRAME,
    RESERVE_S,
    STAGE_BUDGET_S,
    STAGES,
    LatencyError,
    StalenessGuard,
    StageTimestamps,
    StaleFrameRejected,
    latency_report,
)


def stamps(capture=0.0, detect=0.008, filter=0.012, predict=0.020, command=0.024,
           frame_id=7):
    # stage durations 8 / 4 / 8 / 4 ms, all inside the pre-registered budgets
    return StageTimestamps(capture=capture, detect=detect, filter=filter,
                           predict=predict, command=command, frame_id=frame_id)


class PreRegistrationTests(unittest.TestCase):
    def test_threshold_and_budgets_are_the_documented_values(self):
        self.assertEqual(MAX_SOURCE_FRAME_AGE_S, 0.100)
        self.assertEqual(END_TO_END_BUDGET_S, 0.100)
        self.assertEqual(RESERVE_S, 0.040)
        self.assertEqual(sorted(STAGE_BUDGET_S), sorted(STAGES))
        self.assertAlmostEqual(sum(STAGE_BUDGET_S.values()) + RESERVE_S,
                               END_TO_END_BUDGET_S, places=9)


class StageTimestampTests(unittest.TestCase):
    def test_durations_sum_to_end_to_end(self):
        record = stamps()
        durations = record.durations_s()
        self.assertEqual(sorted(durations), sorted(STAGES))
        self.assertAlmostEqual(sum(durations.values()), record.end_to_end_s, places=9)
        self.assertAlmostEqual(record.end_to_end_s, 0.024, places=9)

    def test_out_of_order_timestamps_are_rejected(self):
        with self.assertRaises(LatencyError):
            StageTimestamps(capture=0.0, detect=0.02, filter=0.01,
                            predict=0.03, command=0.04)

    def test_non_finite_timestamps_are_rejected(self):
        with self.assertRaises(LatencyError):
            StageTimestamps(capture=0.0, detect=float("nan"),
                            filter=0.02, predict=0.03, command=0.04)

    def test_decision_age_is_command_minus_capture(self):
        record = stamps(capture=10.0, detect=10.012, filter=10.014,
                        predict=10.026, command=10.031)
        self.assertAlmostEqual(record.decision_age_s, 0.031, places=9)
        self.assertEqual(record.source_frame_timestamp_s, 10.0)


class StalenessTests(unittest.TestCase):
    def test_boundary_is_inclusive_on_the_threshold(self):
        guard = StalenessGuard()
        admitted = guard.admit(prediction_id="p-at-threshold", frame_id=1,
                               source_frame_timestamp_s=0.0, now_s=0.100)
        self.assertEqual(admitted.decision, "ADMITTED")
        self.assertAlmostEqual(admitted.age_s, 0.100, places=9)

    def test_a_frame_just_over_the_threshold_is_refused(self):
        guard = StalenessGuard()
        with self.assertRaises(StaleFrameRejected) as ctx:
            guard.admit(prediction_id="p-stale", frame_id=2,
                        source_frame_timestamp_s=0.0, now_s=0.1001)
        self.assertEqual(ctx.exception.reason, REJECT_AGE_EXCEEDED)
        self.assertEqual(ctx.exception.admission.decision, "REFUSED:" + REJECT_AGE_EXCEEDED)
        self.assertEqual(guard.report()["refusals_by_reason"], {REJECT_AGE_EXCEEDED: 1})

    def test_a_stale_prediction_is_never_silently_reused(self):
        guard = StalenessGuard()
        guard.admit(prediction_id="p1", frame_id=3, source_frame_timestamp_s=0.0, now_s=0.05)
        with self.assertRaises(StaleFrameRejected) as ctx:
            guard.admit(prediction_id="p1", frame_id=3,
                        source_frame_timestamp_s=0.0, now_s=0.30)
        # stale AND reused: staleness is checked first, so the reason is age
        self.assertEqual(ctx.exception.reason, REJECT_AGE_EXCEEDED)
        self.assertEqual(guard.report()["admitted"], 1)

    def test_reuse_within_the_threshold_is_still_refused(self):
        guard = StalenessGuard()
        guard.admit(prediction_id="p2", frame_id=4, source_frame_timestamp_s=0.0, now_s=0.01)
        with self.assertRaises(StaleFrameRejected) as ctx:
            guard.admit(prediction_id="p2", frame_id=4,
                        source_frame_timestamp_s=0.0, now_s=0.02)
        self.assertEqual(ctx.exception.reason, REJECT_FRAME_REUSE)

    def test_a_future_source_frame_is_refused(self):
        guard = StalenessGuard()
        with self.assertRaises(StaleFrameRejected) as ctx:
            guard.admit(prediction_id="p3", frame_id=5,
                        source_frame_timestamp_s=10.0, now_s=9.9)
        self.assertEqual(ctx.exception.reason, REJECT_FUTURE_FRAME)

    def test_empty_prediction_id_is_refused(self):
        guard = StalenessGuard()
        with self.assertRaises(StaleFrameRejected):
            guard.admit(prediction_id="  ", frame_id=6,
                        source_frame_timestamp_s=0.0, now_s=0.01)

    def test_threshold_can_be_tightened_but_not_disabled(self):
        tight = StalenessGuard(threshold_s=0.05)
        with self.assertRaises(StaleFrameRejected):
            tight.admit(prediction_id="p4", frame_id=7,
                        source_frame_timestamp_s=0.0, now_s=0.06)
        with self.assertRaises(LatencyError):
            StalenessGuard(threshold_s=0.0)

    def test_report_counts_admitted_and_refused(self):
        guard = StalenessGuard()
        guard.admit(prediction_id="ok", frame_id=8, source_frame_timestamp_s=0.0, now_s=0.05)
        try:
            guard.admit(prediction_id="bad", frame_id=9,
                        source_frame_timestamp_s=0.0, now_s=0.5)
        except StaleFrameRejected:
            pass
        report = guard.report()
        self.assertEqual(report["admitted"], 1)
        self.assertEqual(report["refused"], 1)
        self.assertEqual(report["threshold_s"], MAX_SOURCE_FRAME_AGE_S)
        self.assertEqual(report["evidence_class"], "OFFLINE-SYNTHETIC")


class BudgetAccountingTests(unittest.TestCase):
    def test_a_within_budget_chain_reports_within_budget(self):
        report = latency_report([stamps()])
        self.assertEqual(report["end_to_end"]["status"], "WITHIN_BUDGET")
        self.assertEqual(report["end_to_end"]["over_budget_frames"], 0)
        self.assertEqual(report["per_stage"]["capture"]["status"],
                         "NOT_MEASURABLE_FROM_STAGE_TIMESTAMPS")
        for stage in STAGES[1:]:
            self.assertEqual(report["per_stage"][stage]["status"], "WITHIN_BUDGET")

    def test_an_over_budget_stage_is_reported_not_rounded_away(self):
        slow = StageTimestamps(capture=0.0, detect=0.080, filter=0.081,
                               predict=0.090, command=0.095, frame_id=11)
        report = latency_report([stamps(), slow])
        self.assertEqual(report["per_stage"]["detect"]["over_budget_frames"], 1)
        self.assertEqual(report["per_stage"]["detect"]["status"], "OVER_BUDGET")
        self.assertAlmostEqual(report["per_stage"]["detect"]["over_budget_max_s"], 0.080, places=9)
        self.assertEqual(report["end_to_end"]["over_budget_frames"], 0)  # 0.095 <= 0.100

    def test_end_to_end_over_budget_is_detected(self):
        slow = StageTimestamps(capture=0.0, detect=0.010, filter=0.011,
                               predict=0.012, command=0.150, frame_id=12)
        report = latency_report([slow])
        self.assertEqual(report["end_to_end"]["status"], "OVER_BUDGET")
        self.assertEqual(report["end_to_end"]["max_frame_id"], 12)

    def test_budget_table_must_add_up(self):
        with self.assertRaises(LatencyError):
            latency_report([stamps()], stage_budget_s={"capture": 0.2})
        broken = dict(STAGE_BUDGET_S)
        broken["detect"] = 0.5
        with self.assertRaises(LatencyError):
            latency_report([stamps()], stage_budget_s=broken)

    def test_empty_batch_is_an_error(self):
        with self.assertRaises(LatencyError):
            latency_report([])

    def test_report_carries_pre_registration_and_evidence_class(self):
        report = latency_report([stamps()])
        self.assertEqual(report["pre_registered"]["max_source_frame_age_s"],
                         MAX_SOURCE_FRAME_AGE_S)
        self.assertEqual(report["evidence_class"], "OFFLINE-SYNTHETIC")
        self.assertIn("UNKNOWN", report["pre_registered"]["measured_replacements"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
