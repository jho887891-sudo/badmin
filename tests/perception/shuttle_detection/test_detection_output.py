#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-04 acceptance: detection output schema, JSON round-trip and unit guards.

Every check here uses synthetic payloads only.  Nothing in this file measures a
detector; it pins the *contract* that later real detections must satisfy.

Run:
    python -m pytest tests/perception/shuttle_detection/test_detection_output.py -q
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.detection_output import (  # noqa: E402
    MIN_PIXEL_VARIANCE_PX2,
    SCHEMA_ID,
    SCHEMA_VERSION,
    DetectionObservation,
    DetectionSchemaError,
    ImageSize,
    ValidityFlags,
    validate_detection_payload,
)

SIZE = ImageSize(width_px=1280, height_px=720)


def make_observation(**overrides):
    kwargs = dict(
        timestamp_s=1.25,
        frame_id=37,
        camera_id="cam_left",
        bbox_xyxy_px=(640.0, 360.0, 646.0, 366.0),
        confidence=0.42,
        pixel_covariance=[[1.5, 0.25], [0.25, 2.5]],
        image_size=SIZE,
    )
    kwargs.update(overrides)
    return DetectionObservation(**kwargs)


class TestJsonRoundTrip(unittest.TestCase):
    def test_round_trip_through_real_json(self):
        obs = make_observation()
        text = json.dumps(obs.to_json_dict())
        rebuilt = DetectionObservation.from_json_dict(json.loads(text))
        self.assertEqual(rebuilt.timestamp_s, obs.timestamp_s)
        self.assertEqual(rebuilt.frame_id, obs.frame_id)
        self.assertEqual(rebuilt.camera_id, obs.camera_id)
        self.assertEqual(rebuilt.confidence, obs.confidence)
        self.assertEqual(rebuilt.image_size, obs.image_size)
        self.assertEqual(rebuilt.validity, obs.validity)
        np.testing.assert_allclose(rebuilt.bbox_xyxy_px, obs.bbox_xyxy_px)
        np.testing.assert_allclose(rebuilt.pixel_covariance, obs.pixel_covariance)

    def test_payload_declares_units_and_identity(self):
        payload = make_observation().to_json_dict()
        self.assertEqual(payload["schema"], SCHEMA_ID)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["units"]["bbox_xyxy_px"], "px")
        self.assertEqual(payload["units"]["pixel_covariance"], "px^2")
        self.assertEqual(payload["units"]["timestamp_s"], "s")
        # JSON-safe: no numpy scipy scalars left behind
        json.dumps(payload)

    def test_centre_and_extent_are_pixel_quantities(self):
        obs = make_observation()
        np.testing.assert_allclose(obs.bbox_centre_px, (643.0, 363.0))
        np.testing.assert_allclose(obs.bbox_extent_px, (6.0, 6.0))
        self.assertEqual(obs.units["bbox_xyxy_px"], "px")

    def test_unknown_key_rejected_in_strict_mode(self):
        payload = make_observation().to_json_dict()
        payload["extra_metric_depth"] = 3.0
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(payload)
        # lenient mode still rebuilds the observation
        self.assertEqual(validate_detection_payload(payload, strict=False).frame_id, 37)

    def test_schema_identity_must_match(self):
        payload = make_observation().to_json_dict()
        payload["schema_version"] = SCHEMA_VERSION + 1
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(payload)
        payload = make_observation().to_json_dict()
        payload["schema"] = "something.else"
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(payload)

    def test_missing_unrecoverable_fields_rejected(self):
        for key in ("timestamp_s", "frame_id", "bbox_xyxy_px", "confidence",
                    "pixel_covariance", "units", "validity"):
            payload = make_observation().to_json_dict()
            del payload[key]
            with self.assertRaises(DetectionSchemaError, msg=key):
                validate_detection_payload(payload)


class TestUnitConfusion(unittest.TestCase):
    """Pixels and metres must never be interchangeable."""

    def test_metre_bbox_key_is_rejected(self):
        payload = make_observation().to_json_dict()
        payload["bbox_xyxy_m"] = [0.1, 0.2, 0.15, 0.25]
        with self.assertRaises(DetectionSchemaError) as ctx:
            validate_detection_payload(payload)
        self.assertIn("unit confusion", str(ctx.exception).lower())

    def test_metre_squared_covariance_is_rejected(self):
        # A believable stereo covariance in m^2 written into the pixel field.
        with self.assertRaises(DetectionSchemaError) as ctx:
            make_observation(pixel_covariance=[[1e-4, 0.0], [0.0, 1e-4]])
        self.assertIn("unit confusion", str(ctx.exception).lower())
        self.assertLess(1e-4, MIN_PIXEL_VARIANCE_PX2)

    def test_units_block_must_say_pixels(self):
        payload = make_observation().to_json_dict()
        payload["units"]["bbox_xyxy_px"] = "m"
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(payload)
        payload = make_observation().to_json_dict()
        payload["units"]["pixel_covariance"] = "m^2"
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(payload)

    def test_normalised_coordinates_are_rejected(self):
        with self.assertRaises(DetectionSchemaError):
            make_observation(bbox_xyxy_px=(-0.5, 0.1, 0.6, 0.9))


class TestNonFiniteRejection(unittest.TestCase):
    def test_nan_and_inf_in_every_float_field(self):
        bad = [
            dict(timestamp_s=float("nan")),
            dict(timestamp_s=float("inf")),
            dict(confidence=float("nan")),
            dict(confidence=float("inf")),
            dict(bbox_xyxy_px=(640.0, float("nan"), 646.0, 666.0)),
            dict(bbox_xyxy_px=(640.0, 360.0, float("inf"), 366.0)),
            dict(pixel_covariance=[[float("nan"), 0.0], [0.0, 1.0]]),
            dict(pixel_covariance=[[1.0, float("inf")], [float("inf"), 1.0]]),
        ]
        for override in bad:
            with self.assertRaises(DetectionSchemaError, msg=str(override)):
                make_observation(**override)

    def test_json_nan_is_rejected_not_round_tripped(self):
        payload = make_observation().to_json_dict()
        payload["confidence"] = float("nan")
        with self.assertRaises(DetectionSchemaError):
            validate_detection_payload(json.loads(json.dumps(payload, allow_nan=True)))


class TestRangesAndGeometry(unittest.TestCase):
    def test_confidence_range(self):
        for value in (-0.01, 1.01, 2.0):
            with self.assertRaises(DetectionSchemaError, msg=str(value)):
                make_observation(confidence=value)
        for value in (0.0, 0.5, 1.0):
            self.assertEqual(make_observation(confidence=value).confidence, value)

    def test_bbox_ordering_and_shape(self):
        for bbox in ((646.0, 360.0, 640.0, 366.0),      # x2 < x1
                     (640.0, 366.0, 646.0, 360.0),      # y2 < y1
                     (640.0, 360.0, 640.0, 366.0),      # zero width
                     (640.0, 360.0, 646.0)):            # wrong length
            with self.assertRaises(DetectionSchemaError, msg=str(bbox)):
                make_observation(bbox_xyxy_px=bbox)

    def test_frame_id_and_camera_id(self):
        with self.assertRaises(DetectionSchemaError):
            make_observation(frame_id=-1)
        with self.assertRaises(DetectionSchemaError):
            make_observation(frame_id=1.5)
        with self.assertRaises(DetectionSchemaError):
            make_observation(camera_id="")
        with self.assertRaises(DetectionSchemaError):
            make_observation(camera_id="   ")

    def test_timestamp_must_be_finite_and_non_negative(self):
        with self.assertRaises(DetectionSchemaError):
            make_observation(timestamp_s=-0.001)

    def test_image_size_must_be_pixels(self):
        with self.assertRaises(DetectionSchemaError):
            ImageSize(width_px=0, height_px=720)
        with self.assertRaises(DetectionSchemaError):
            ImageSize(width_px=1280.5, height_px=720)

    def test_bbox_outside_image_sets_flag_and_flag_cannot_lie(self):
        obs = make_observation(bbox_xyxy_px=(1270.0, 700.0, 1290.0, 730.0),
                               validity=ValidityFlags(in_image=False,
                                                      reject_reason="GEOMETRY_OUT_OF_IMAGE"))
        self.assertFalse(obs.validity.in_image)
        self.assertFalse(obs.usable)
        # geometry says inside, flag says outside -> contradiction is refused
        with self.assertRaises(DetectionSchemaError):
            make_observation(validity=ValidityFlags(in_image=False))
        with self.assertRaises(DetectionSchemaError):
            make_observation(bbox_xyxy_px=(1270.0, 700.0, 1290.0, 730.0))


class TestCovarianceValidity(unittest.TestCase):
    def test_positive_semi_definite_required(self):
        with self.assertRaises(DetectionSchemaError):
            make_observation(pixel_covariance=[[1.0, 2.0], [2.0, 1.0]])   # eig 3, -1
        # exactly singular is allowed (PSD, not PD)
        obs = make_observation(pixel_covariance=[[1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(float(np.min(np.linalg.eigvalsh(obs.pixel_covariance))), 0.0)

    def test_symmetry_required(self):
        with self.assertRaises(DetectionSchemaError):
            make_observation(pixel_covariance=[[1.0, 0.3], [0.1, 1.0]])

    def test_shape_required(self):
        for cov in ([[1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    np.eye(4)):
            with self.assertRaises(DetectionSchemaError, msg=str(np.shape(cov))):
                make_observation(pixel_covariance=cov)

    def test_reason_vocabulary_is_closed(self):
        with self.assertRaises(DetectionSchemaError):
            ValidityFlags(reject_reason="WHATEVER")

    def test_usable_requires_every_flag(self):
        self.assertTrue(make_observation().usable)
        self.assertFalse(make_observation(
            validity=ValidityFlags(confidence_ok=False,
                                   reject_reason="LOW_CONFIDENCE")).usable)
        self.assertFalse(make_observation(
            validity=ValidityFlags(detected=False, reject_reason="NOT_DETECTED")).usable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
