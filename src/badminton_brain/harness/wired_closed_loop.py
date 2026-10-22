# -*- coding: utf-8 -*-
"""P7-12: the WIRED simulation closed-loop harness (simulation only).

WHAT CHANGED FROM P7-10
-----------------------
P7-10 (harness/closed_loop.py) drove the FSM with MOCK perception, MOCK prediction
and a SPHERE racket proxy: every stage was a stand-in.  This module keeps the same
refusal rule and replaces the stand-ins with the real P7 interfaces:

    ground-truth ball
      -> SyntheticStereoDetector.detect()          (P7 stereo detection interface)
      -> stereo_geometry.triangulate()             (2-D pixels -> 3-D court point)
      -> ShuttleMeasurement -> ShuttleEstimatorBridge.update()   (the UKF/EKF stage)
      -> PhysicsTrajectoryPredictor.predict()      (the prediction stage)
      -> PredictionInput                           (intercept point/time/approach/sigma)
      -> InteractionFsm (SEARCH..EXECUTE/HOLD)     (P7-10 fail-closed FSM)
      -> assess_reachability(solver=backend IK)    (P7-09 decision, chosen backend)
      -> racket collision proxy LOADED from the committed P6 JSON
      -> SIM_PROXY_GEOMETRY_ONLY contact feasibility at the planned pose

Two things are still SIMULATION: the ball is analytic (no physics engine) and the
racket is the P6 43-primitive collision proxy (no contact model).  The D455 is not
calibrated, so the stereo rig is the ENGINEERING_V0_1 synthetic baseline; the
court->base extrinsic is UNKNOWN, so every run that supplies a Calibration is
declaring an ASSUMED identity and the report says so.

WHAT THIS REFUSES TO REPORT
---------------------------
No hit rate.  Ever.  hit_rate is None and real_hit_rate_claimed is False in every
report, including a complete one.  What it reports instead is what it can actually
defend: states entered, HOLD reasons, reachability verdicts per backend, IK
residual distributions, time-to-target margins, and the SIM_PROXY geometry
clearance.  A geometric clearance is NOT a physical contact and the label says so.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..estimation.shuttle_ukf import ShuttleEstimatorBridge, ShuttleUKF
from ..prediction.physics_predictor import PhysicsTrajectoryPredictor
from ..perception.stereo_geometry import baseline_stereo_setup, triangulate
from ..perception.synthetic_detector import SyntheticStereoDetector
from ..reachability import Calibration, target_from_intercept
from ..reachability.backends import (BACKEND_CHOICES, BACKEND_PROXY,
                                     BackendDampedLeastSquaresIk, BackendMixError,
                                     KinematicsBackend, VerifiedFkUnavailable,
                                     kinematic_limits_for, select_backend, sha256_file,
                                     P6_PROXY_RELPATH)
from ..safety.interaction_fsm import (FrameInput, FsmLimits, FsmState, InteractionFsm,
                                      PlanInput, PredictionInput)
from ..types import ShuttleMeasurement

__all__ = [
    "SCENARIOS", "BALL_START_M", "BALL_VELOCITY_MPS", "STEP_DT_S", "STEPS",
    "PERTURBATION_SIGMA_M", "HIT_PLANE_Z_M", "LEAD_FLOOR_S", "BALL_RADIUS_M",
    "GEOMETRY_LABEL", "RacketPrimitive", "RacketCollisionProxy", "load_racket_proxy",
    "find_proxy_json", "GeometryContact", "WiredConfig", "run_wired_closed_loop",
    "run_episodes", "summarise",
]

SCENARIOS: Tuple[str, ...] = ("FIXED_BALL", "LOW_SPEED_INCOMING", "PERTURBED")

#: Pre-registered scenario constants (P7-10 kept the first three; this slice adds
#: the intercept plane, the lead floor and the ball sphere radius).
BALL_START_M = (0.35, 0.05, 1.35)
BALL_VELOCITY_MPS = (0.25, 0.02, -2.0)   # |v| ~ 2 m/s: low speed by design
STEP_DT_S = 1.0 / 60.0
STEPS = 12
PERTURBATION_SIGMA_M = 0.004
#: The declared intercept plane: the swing is planned where the PREDICTED
#: trajectory crosses this height (metres, court Z).
HIT_PLANE_Z_M = 1.0
#: Minimum planning lead: the intercept sample is never sooner than this.  At the
#: P5-05 velocity cap a full-arm slew costs O(0.15-0.75) s (the required_slew numbers
#: this harness reports), so a plan made later than about 0.45 s out cannot be
#: time-feasible.  Pre-registered, same for both backends.
LEAD_FLOOR_S = 0.45

#: The DECLARED return direction: the direction the racket face must point, i.e.
#: where the shuttle should go (+X is the opponent side in the court frame).  This
#: is a pre-registered geometric choice, NOT a per-backend tuning: the velocity
#: derived alternative (-v, a straight vertical block) was also measured and makes
#: BOTH backends fail, so using it would only hide the geometric difference.
RETURN_DIRECTION_M = (1.0, 0.0, 0.0)

#: Pre-registered nominal postures, one per backend, used as the first seed.
NOMINAL_SEED_RAD = {
    BACKEND_PROXY: (0.0, 1.5, -1.5, 0.0, 0.0, 0.0),
    "verified_p6_fk": (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575),
}
#: Declared SHUTTLE SPHERE radius for the geometric check.  A shuttlecock is a
#: cone ~6.7 cm long with a ~2.5-2.8 cm cork; 0.030 m is a PROXY radius, NOT a
#: measurement (listed in the UNKNOWNs).
BALL_RADIUS_M = 0.030
#: Framing label for the geometric check.  Never drop it from a report.
GEOMETRY_LABEL = "SIM_PROXY_GEOMETRY_ONLY"


def find_proxy_json(*, root: Optional[str] = None,
                    extra_roots: Sequence[str] = ()) -> Optional[Path]:
    """Locate the COMMITTED P6 racket collision proxy JSON, or None."""
    if root:
        candidate = Path(root) / P6_PROXY_RELPATH
        if candidate.is_file():
            return candidate.resolve()
    env_root = os.environ.get("BADMIN_P6_ROOT")
    if env_root:
        candidate = Path(env_root) / P6_PROXY_RELPATH
        if candidate.is_file():
            return candidate.resolve()
    for entry in extra_roots:
        candidate = Path(entry) / P6_PROXY_RELPATH
        if candidate.is_file():
            return candidate.resolve()
    repo = Path(__file__).resolve().parents[3] / P6_PROXY_RELPATH
    return repo.resolve() if repo.is_file() else None


def _point_segment_distance(point: np.ndarray, start: np.ndarray,
                            end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ segment / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def _box_sdf(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    delta = np.abs(point - center) - half
    outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
    inside = float(min(np.max(delta), 0.0))
    return outside + inside


@dataclass(frozen=True)
class RacketPrimitive:
    """One committed collision primitive, expressed in the racket_root frame."""

    primitive_id: str
    kind: str
    role: str
    data: Dict[str, Any]

    def sdf(self, point_local: np.ndarray) -> float:
        """Signed distance from point_local (racket_root frame) to the surface."""
        if self.kind == "capsule":
            start = np.asarray(self.data["axis_start_m"], dtype=float)
            end = np.asarray(self.data["axis_end_m"], dtype=float)
            return _point_segment_distance(point_local, start, end) \
                - float(self.data["radius_m"])
        if self.kind == "box":
            return _box_sdf(point_local, np.asarray(self.data["center_m"], dtype=float),
                            np.asarray(self.data["half_extents_m"], dtype=float))
        if self.kind == "sphere":
            centre = np.asarray(self.data["center_m"], dtype=float)
            return float(np.linalg.norm(point_local - centre)
                         - float(self.data["radius_m"]))
        raise ValueError("unsupported proxy primitive %r of type %r; refusing to "
                         "approximate it silently" % (self.primitive_id, self.kind))


@dataclass(frozen=True)
class GeometryContact:
    """SIM_PROXY geometry verdict for one planned pose.  NOT a physical contact."""

    contacted: bool
    clearance_m: float
    nearest_primitive: str
    ball_radius_m: float = BALL_RADIUS_M
    label: str = GEOMETRY_LABEL

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "contacted": bool(self.contacted),
            "clearance_m": float(self.clearance_m),
            "nearest_primitive": self.nearest_primitive,
            "ball_radius_m": float(self.ball_radius_m),
            "note": "min over the committed racket primitives of the signed distance "
                    "from the ball centre, minus the declared ball radius. A geometry "
                    "test, NOT a physical contact model.",
        }


class RacketCollisionProxy:
    """The COMMITTED P6 racket collision proxy, loaded - never invented.

    Every number comes from
    outputs/simulation/rebot_b601_rs/morph_one/p6_frames/racket_collision_proxy.json
    (P6-07: 43 primitives, 100 percent surface containment within 1 mm).  The
    loader refuses a file with another schema or a primitive count that disagrees
    with the record, so a truncated copy cannot pass as the proxy.
    """

    SCHEMA = "p6_racket_collision_proxy/1"

    def __init__(self, data: Dict[str, Any], *, path: str, sha256: str) -> None:
        self.path = path
        self.sha256 = sha256
        schema = data.get("schema")
        if schema != self.SCHEMA:
            raise ValueError("racket proxy schema %r is not %r" % (schema, self.SCHEMA))
        primitives = data.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise ValueError("racket proxy has no primitives")
        declared = int(data.get("primitive_count", -1))
        if declared != len(primitives):
            raise ValueError("racket proxy declares %d primitives but carries %d: the "
                             "file is inconsistent" % (declared, len(primitives)))
        self.primitives = tuple(
            RacketPrimitive(primitive_id=str(entry["id"]), kind=str(entry["type"]),
                            role=str(entry.get("role", "")),
                            data=dict(entry.get("geometry", {})))
            for entry in primitives
        )
        bed_centres = [float(entry["geometry"]["center_m"][2]) for entry in primitives
                       if str(entry.get("role")) == "string_bed"
                       and str(entry.get("type")) == "box"]
        if not bed_centres:
            raise ValueError("racket proxy has no string_bed box: the face centre cannot "
                             "be derived from the committed geometry")
        #: Face centre = the mean z of the committed string-bed slabs, along +Z from
        #: the handle butt (racket_root origin).  Derived, not assumed.
        self.face_center_from_root_m = float(np.mean(bed_centres))
        self.primitive_count = len(self.primitives)
        self.kind = str(data.get("kind", ""))
        self.role_counts: Dict[str, int] = {}
        for primitive in self.primitives:
            self.role_counts[primitive.role] = self.role_counts.get(primitive.role, 0) + 1

    def check(self, ball_center_m: Sequence[float], face_position_m: Sequence[float],
              face_rotation: np.ndarray,
              ball_radius_m: float = BALL_RADIUS_M) -> GeometryContact:
        """Place the proxy so its face centre is at the planned face pose, then test.

        The planned pose (P7-09 convention) is the RACKET FACE CENTRE with +X the
        face normal and +Z handle->head.  racket_root therefore sits
        face_center_from_root_m back along -Z.  That single placement rule is the
        only assumption here, and its length is derived from the committed geometry.
        """
        ball = np.asarray(ball_center_m, dtype=float).reshape(3)
        face = np.asarray(face_position_m, dtype=float).reshape(3)
        rotation = np.asarray(face_rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("face_rotation must be (3,3), got %r" % (rotation.shape,))
        root = face - float(self.face_center_from_root_m) * rotation[:, 2]
        local = rotation.T @ (ball - root)
        distances = [(primitive.sdf(local), primitive.primitive_id)
                     for primitive in self.primitives]
        clearance, nearest = min(distances, key=lambda pair: pair[0])
        return GeometryContact(contacted=bool(clearance - ball_radius_m <= 0.0),
                               clearance_m=float(clearance - ball_radius_m),
                               nearest_primitive=nearest,
                               ball_radius_m=float(ball_radius_m))

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema": self.SCHEMA,
            "primitive_count": self.primitive_count,
            "role_counts": dict(sorted(self.role_counts.items())),
            "face_center_from_root_m": self.face_center_from_root_m,
            "kind": self.kind,
            "loaded_from_committed_json": True,
        }


def load_racket_proxy(path: Optional[str] = None, *,
                      extra_roots: Sequence[str] = ()) -> RacketCollisionProxy:
    """Load the committed proxy.  A missing file is a REFUSAL, not a default."""
    resolved = Path(path).resolve() if path else find_proxy_json(extra_roots=extra_roots)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(
            "RACKET_COLLISION_PROXY_MISSING: the committed P6 proxy JSON was not found "
            "(looked at $BADMIN_P6_ROOT and this repository for %r); refusing to invent "
            "racket geometry" % P6_PROXY_RELPATH)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return RacketCollisionProxy(data, path=str(resolved), sha256=sha256_file(resolved))


@dataclass(frozen=True)
class WiredConfig:
    """One wired episode's pre-registered configuration."""

    scenario: str = "LOW_SPEED_INCOMING"
    seed: int = 20261022
    steps: int = STEPS
    dt_s: float = STEP_DT_S
    backend: str = BACKEND_PROXY
    p6_root: Optional[str] = None
    fk_path: Optional[str] = None
    proxy_json: Optional[str] = None
    calibration: Optional[Calibration] = None
    hit_plane_z_m: float = HIT_PLANE_Z_M
    lead_floor_s: float = LEAD_FLOOR_S
    ball_radius_m: float = BALL_RADIUS_M
    detection_noise_sigma_px: float = 0.5

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError("scenario must be one of %r, got %r"
                             % (SCENARIOS, self.scenario))
        if self.backend not in BACKEND_CHOICES:
            raise ValueError("backend must be one of %r, got %r"
                             % (BACKEND_CHOICES, self.backend))
        if self.steps <= 0 or not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("steps and dt_s must be positive")


def _nominal_seed(backend: KinematicsBackend) -> np.ndarray:
    """The pre-registered starting posture for this backend, bounds-checked."""
    declared = NOMINAL_SEED_RAD.get(backend.backend_id)
    if declared is None:
        raise ValueError("no pre-registered nominal posture for backend %r"
                         % backend.backend_id)
    seed = np.asarray(declared, dtype=float)
    lower = np.asarray(backend.lower_rad, dtype=float)
    upper = np.asarray(backend.upper_rad, dtype=float)
    if seed.shape != lower.shape or np.any(seed < lower) or np.any(seed > upper):
        raise ValueError("nominal seed for %r is outside the joint limits"
                         % backend.backend_id)
    return seed


def _ball_position(scenario: str, step: int, dt: float,
                   rng: np.random.Generator) -> np.ndarray:
    start = np.asarray(BALL_START_M, dtype=float)
    if scenario == "FIXED_BALL":
        return start.copy()
    velocity = np.asarray(BALL_VELOCITY_MPS, dtype=float)
    position = start + velocity * (step * dt)
    if scenario == "PERTURBED":
        position = position + rng.normal(0.0, PERTURBATION_SIGMA_M, size=3)
    return position


def _intercept_from_trajectory(times: np.ndarray, positions: np.ndarray,
                               velocities: np.ndarray, now_s: float, *,
                               hit_plane_z_m: float, lead_floor_s: float
                               ) -> Tuple[np.ndarray, float, np.ndarray, str]:
    """Pick the intercept sample from the PREDICTED trajectory (deterministic)."""
    times = np.asarray(times, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    velocities = np.asarray(velocities, dtype=float).reshape(-1, 3)
    eligible = np.nonzero(times >= now_s + lead_floor_s)[0]
    if eligible.size == 0:
        eligible = np.arange(times.size)
    crossing = [int(i) for i in eligible if positions[i, 2] <= hit_plane_z_m]
    if crossing:
        index = crossing[0]
        rule = "HIT_PLANE_CROSSING"
    else:
        index = int(min(eligible, key=lambda i: abs(positions[i, 2] - hit_plane_z_m)))
        rule = "NEAREST_TO_HIT_PLANE"
    point = positions[index].copy()
    approach = np.asarray(RETURN_DIRECTION_M, dtype=float)
    approach = approach / float(np.linalg.norm(approach))
    return point, float(times[index]), approach, rule


def _percentiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _build_backend(config: WiredConfig) -> KinematicsBackend:
    return select_backend(config.backend, root=config.p6_root, fk_path=config.fk_path)


def run_wired_closed_loop(config: WiredConfig) -> Dict[str, Any]:
    """Run ONE wired episode.  Returns a report that never contains a hit rate."""
    backend = _build_backend(config)
    solver = BackendDampedLeastSquaresIk(backend)
    limits = kinematic_limits_for(backend)
    proxy: Optional[RacketCollisionProxy]
    try:
        proxy = load_racket_proxy(config.proxy_json, extra_roots=(
            (config.p6_root,) if config.p6_root else ()))
    except FileNotFoundError:
        proxy = None

    missing: List[str] = []
    if proxy is None:
        missing.append("RACKET_COLLISION_PROXY_MISSING")
    calibration = config.calibration
    if calibration is None or not calibration.usable():
        missing.append("CALIBRATION_MISSING")

    setup = baseline_stereo_setup()
    detector = SyntheticStereoDetector(noise_sigma_px=config.detection_noise_sigma_px,
                                       seed=config.seed)
    # fuse_velocity=False: one stereo frame measures a POSITION, and a fabricated
    # per-frame velocity would be an invented measurement.  The measurement contract
    # still requires a velocity field, so it carries a declared zero that the filter
    # does not fuse (see the P7-12 doc).
    bridge = ShuttleEstimatorBridge(num_envs=1,
                                    ukf=ShuttleUKF(num_envs=1, fuse_velocity=False))
    predictor = PhysicsTrajectoryPredictor(horizon_s=1.0, dt_s=1.0 / 120.0)
    # The FSM carries its OWN thresholds (FsmLimits); they must be the ones the
    # chosen backend's limits declare, or the plan gate and the execute gate would
    # disagree about the same joint vector.
    fsm_limits = FsmLimits(joint_margin_rad=limits.joint_margin_rad,
                           velocity_cap_rad_s=min(float(v) for v in limits.velocity_cap_rad_s))
    fsm = InteractionFsm(calibration=calibration, ik_solver=solver,
                         kinematic_limits=limits, limits=fsm_limits)
    rng = np.random.default_rng(config.seed)

    states_entered: Dict[str, int] = {}
    hold_reasons: Dict[str, int] = {}
    verdicts: List[Dict[str, Any]] = []
    ik_residuals_position: List[float] = []
    ik_residuals_rotation: List[float] = []
    margins: List[float] = []
    revolute_margins: List[float] = []
    geometry: List[Dict[str, Any]] = []
    steps_run = 0
    commands = 0
    seed = _nominal_seed(backend)
    initialised = False
    mix_ids = {backend.backend_id}

    for step in range(config.steps):
        now_s = step * config.dt_s
        ball = _ball_position(config.scenario, step, config.dt_s, rng)
        states_entered["SEARCH"] = states_entered.get("SEARCH", 0) + 1

        # 1. detection interface (pixels), 2. triangulation (3-D court point)
        detection = detector.detect(ball.reshape(1, 3))
        if not bool(detection.valid[0]):
            # An invisible shuttle is a reported sensor event, never a 3-D guess.
            steps_run += 0
            states_entered["DETECTION_DROPPED"] = states_entered.get("DETECTION_DROPPED", 0) + 1
            break
        point = triangulate(detection.uv_left, detection.uv_right,
                            setup.intrinsics_left.matrix(),
                            setup.extrinsics.T_left_right(),
                            T_court_left=setup.extrinsics.T_court_left(),
                            valid=detection.valid)
        point = np.asarray(point, dtype=float).reshape(-1, 3)[0]
        if not np.all(np.isfinite(point)):
            states_entered["TRIANGULATION_INVALID"] = \
                states_entered.get("TRIANGULATION_INVALID", 0) + 1
            break

        # 3. estimation interface (UKF bridge)
        pixel_residual = float(np.linalg.norm(point - ball))
        sigma_m = float(max(pixel_residual, 0.002))
        if not initialised:
            bridge.ukf.initialize(position=point.reshape(1, 3),
                                  velocity=np.zeros((1, 3)), timestamp=now_s)
            initialised = True
        measurement = ShuttleMeasurement(
            position=point.reshape(1, 3), velocity=np.zeros((1, 3)),
            covariance=np.full((1, 3, 3), sigma_m ** 2),
            timestamp=now_s, valid_mask=np.ones(1))
        estimate = bridge.update(measurement)
        est_pos = np.asarray(estimate["position"], dtype=float).reshape(1, 3)[0]
        est_vel = np.asarray(estimate["velocity"], dtype=float).reshape(1, 3)[0]
        state_cov = np.asarray(estimate["covariance"], dtype=float)
        state_cov = state_cov.reshape(1, state_cov.shape[-1], state_cov.shape[-1])[0]
        # The FULL filter covariance projected onto the measured position block:
        # this is the sigma the FSM's uncertainty gate must see, not the raw
        # triangulation residual.
        est_sigma = float(math.sqrt(max(float(np.trace(state_cov[:3, :3])) / 3.0, 1e-12)))

        # 4. prediction interface
        trajectory = predictor.predict(est_pos, est_vel, timestamp=now_s)
        intercept, intercept_time, approach, intercept_rule = _intercept_from_trajectory(
            trajectory.times, trajectory.position[0], trajectory.velocity[0], now_s,
            hit_plane_z_m=config.hit_plane_z_m, lead_floor_s=config.lead_floor_s)

        # 5. FSM: frame -> prediction -> plan -> command
        frame = FrameInput(frame_id=step, capture_time_s=now_s,
                           received_at_s=now_s + 0.002, detection_dim=3,
                           position_m=point, sigma_m=max(est_sigma, 1e-3))
        state = fsm.on_frame(frame, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        prediction = PredictionInput(
            prediction_id="ekf+physics-%d" % step, frame_id=step,
            reference_time_s=now_s + 0.002, intercept_time_s=float(intercept_time),
            intercept_point_m=intercept, approach_direction=approach,
            sigma_start_m=max(est_sigma, 1e-3), sigma_growth_mps=0.02)
        state = fsm.on_prediction(prediction, now_s + 0.002)
        if state is FsmState.HOLD:
            break

        # The plan is the backend's own IK solution for that intercept, so the
        # command and the verdict can never come from different kinematics.
        try:
            plan_target = target_from_intercept(intercept, approach)
            planned = solver.solve(plan_target, seed)
            command = np.asarray(planned.joints_rad, dtype=float)
        except Exception:  # noqa: BLE001 - the FSM re-solves and will HOLD
            command = seed.copy()
        state = fsm.on_plan(PlanInput(planned_at_s=now_s + 0.002, joints_rad=command,
                                      seed_rad=seed, duration_s=0.0), now_s + 0.002)
        verdict = fsm.last_verdict
        if verdict is not None:
            mix_ids.add(backend.backend_id)
            record: Dict[str, Any] = {
                "step": step,
                "backend": backend.backend_id,
                "backend_source_class": backend.source_class,
                "frame_name": backend.frame_name,
                "decision": verdict.decision.value,
                "reason": verdict.reason.value,
                "time_to_target_s": verdict.time_to_target_s,
                "joint_margin": verdict.joint_margin_rad,
                "required_slew_s": verdict.required_slew_s,
                "intercept_point_m": [round(float(v), 6) for v in intercept],
                "intercept_rule": intercept_rule,
                "approach_source": "DECLARED_RETURN_DIRECTION",
                "pixel_residual_m": round(pixel_residual, 9),
            }
            if verdict.ik is not None:
                record["ik_solver"] = verdict.ik.solver
                record["ik_residual_position_m"] = float(verdict.ik.residual_position_m)
                record["ik_residual_rotation_rad"] = float(verdict.ik.residual_rotation_rad)
                record["ik_iterations"] = int(verdict.ik.iterations)
                revolute = [i for i, name in enumerate(backend.joint_names)
                            if not name.startswith("gripper")]
                q_ik = np.asarray(verdict.ik.joints_rad, dtype=float)
                lo = np.asarray(backend.lower_rad, dtype=float)
                hi = np.asarray(backend.upper_rad, dtype=float)
                rev_margin = float(np.min(np.minimum(q_ik[revolute] - lo[revolute],
                                                     hi[revolute] - q_ik[revolute])))
                record["revolute_joint_margin"] = rev_margin
                revolute_margins.append(rev_margin)
                ik_residuals_position.append(float(verdict.ik.residual_position_m))
                ik_residuals_rotation.append(float(verdict.ik.residual_rotation_rad))
                if verdict.time_to_target_s is not None:
                    available = (float(verdict.time_to_target_s)
                                 - float(verdict.required_slew_s or 0.0))
                    margins.append(float(available))
                    record["time_to_target_margin_s"] = float(available)
                # 6. SIM_PROXY geometry at the planned pose, whether or not the FSM
                # later refuses to execute it.
                if proxy is not None:
                    T = np.asarray(backend.pose(command), dtype=float)
                    # The ball is tested where it is PLANNED to be (the predicted
                    # intercept), not where it is now: a pose 0.45 s ahead compared
                    # against the present ball would be a meaningless distance.
                    contact = proxy.check(intercept, T[:3, 3], T[:3, :3],
                                          ball_radius_m=config.ball_radius_m)
                    entry = contact.as_dict()
                    entry.update({"step": step, "plan_stage": "PLANNED",
                                  "ball_reference": "PREDICTED_INTERCEPT",
                                  "backend": backend.backend_id,
                                  "face_position_m": [round(float(v), 6) for v in T[:3, 3]]})
                    geometry.append(entry)
                    record["geometry_clearance_m"] = contact.clearance_m
                    record["geometry_contact"] = contact.contacted
                    record["geometry_nearest_primitive"] = contact.nearest_primitive
            verdicts.append(record)
        if state is FsmState.HOLD:
            break

        state = fsm.on_execute_command(command, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        commands += 1
        steps_run += 1
        seed = command

    for event in fsm.events:
        states_entered[event.to_state.value] = states_entered.get(event.to_state.value, 0) + 1
        if event.reason is not None:
            hold_reasons[event.reason.value] = hold_reasons.get(event.reason.value, 0) + 1

    if len(mix_ids) > 1:
        raise BackendMixError("one run produced verdicts from %r; a report may never mix "
                              "backends" % (sorted(mix_ids),))

    report: Dict[str, Any] = {
        "schema": "badmin.p7.wired_closed_loop",
        "schema_version": 1,
        "scenario": config.scenario,
        "seed": config.seed,
        "backend": backend.backend_id,
        "backend_provenance": backend.provenance(),
        "evidence_class": "SIMULATION-ONLY",
        "real_hit_rate_claimed": False,
        "hit_rate": None,
        "physics_engine": None,
        "camera": "SYNTHETIC_ANALYTIC_STEREO",
        "robot": None,
        "calibration_used": None if calibration is None else {
            "verified": bool(calibration.verified),
            "source": calibration.source,
            "court_from_base": None if calibration.court_from_base is None
            else "provided (ASSUMED, not measured)",
        },
        "racket_proxy": None if proxy is None else proxy.describe(),
        "geometry_check_label": GEOMETRY_LABEL,
        "steps_requested": config.steps,
        "steps_completed": steps_run,
        "commands_executed": commands,
        "states_entered": dict(sorted(states_entered.items())),
        "hold_reasons": dict(sorted(hold_reasons.items())),
        "final_state": fsm.state.value,
        "hold_latched": fsm.holding,
        "verdicts": verdicts,
        "reachability_verdicts": _count(
            (verdict["decision"], verdict["reason"]) for verdict in verdicts),
        "ik_residual_position_m": _percentiles(ik_residuals_position),
        "ik_residual_rotation_rad": _percentiles(ik_residuals_rotation),
        "time_to_target_margin_s": _percentiles(margins),
        "revolute_joint_margin_rad": _percentiles(revolute_margins),
        "geometry_evaluations": len(geometry),
        "geometry_contacts": int(sum(1 for entry in geometry if entry["contacted"])),
        "geometry": geometry,
        "fsm_report": fsm.report(),
        "note": "hit_rate is None by contract: without a calibrated frame, a measured "
                "collision model and a physics engine there is no referent for a hit rate. "
                "geometry_contacts is a SIM_PROXY surface-distance statistic, not a hit.",
    }
    if missing:
        report.update({
            "status": "REFUSED",
            "refusal_reason": missing,
            "hit_rate": None,
            "geometry_contacts": 0,
            "geometry_evaluations": 0,
            "refusal_note": "the run still drove the FSM so that the HOLD reason is real; "
                            "it reports the counts it can defend and no rate",
        })
    else:
        report["status"] = "WIRED_RUN_COMPLETE"
    return report


def _count(pairs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for decision, reason in pairs:
        key = decision + ":" + reason
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def run_episodes(*, scenarios: Sequence[str] = SCENARIOS, seeds: Sequence[int] = (20261022,),
                 backend: str = BACKEND_PROXY, **kwargs: Any) -> Dict[str, Any]:
    """Run every (scenario, seed) pair with ONE backend and aggregate the counts."""
    episodes: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            config = WiredConfig(scenario=scenario, seed=int(seed), backend=backend, **kwargs)
            episodes.append(run_wired_closed_loop(config))
    return summarise(episodes, backend=backend)


def summarise(episodes: Sequence[Dict[str, Any]], *, backend: str) -> Dict[str, Any]:
    """Aggregate episode reports.  Counts only - never a rate."""
    backends = {episode["backend"] for episode in episodes}
    if len(backends) > 1:
        raise BackendMixError("summary would mix backends %r" % (sorted(backends),))
    hold_per_reason: Dict[str, int] = {}
    states: Dict[str, int] = {}
    verdicts: Dict[str, int] = {}
    residuals: List[float] = []
    margins: List[float] = []
    revolute_margins: List[float] = []
    contacts = 0
    evaluations = 0
    refused = 0
    for episode in episodes:
        if episode["status"] == "REFUSED":
            refused += 1
        for reason, count in episode["hold_reasons"].items():
            hold_per_reason[reason] = hold_per_reason.get(reason, 0) + count
        for state, count in episode["states_entered"].items():
            states[state] = states.get(state, 0) + count
        for key, count in episode["reachability_verdicts"].items():
            verdicts[key] = verdicts.get(key, 0) + count
        for verdict in episode["verdicts"]:
            if "ik_residual_position_m" in verdict:
                residuals.append(float(verdict["ik_residual_position_m"]))
            if "time_to_target_margin_s" in verdict:
                margins.append(float(verdict["time_to_target_margin_s"]))
            if "revolute_joint_margin" in verdict:
                revolute_margins.append(float(verdict["revolute_joint_margin"]))
        contacts += int(episode["geometry_contacts"])
        evaluations += int(episode["geometry_evaluations"])
    episodes_in_hold = sum(1 for episode in episodes if episode["hold_latched"])
    return {
        "schema": "badmin.p7.wired_closed_loop.summary",
        "schema_version": 1,
        "backend": backend,
        "evidence_class": "SIMULATION-ONLY",
        "hit_rate": None,
        "real_hit_rate_claimed": False,
        "episodes": len(episodes),
        "episodes_refused": refused,
        "episodes_ending_in_hold": episodes_in_hold,
        "hold_reasons_total": dict(sorted(hold_per_reason.items())),
        "states_entered_total": dict(sorted(states.items())),
        "reachability_verdicts_total": dict(sorted(verdicts.items())),
        "ik_residual_position_m": _percentiles(residuals),
        "time_to_target_margin_s": _percentiles(margins),
        "revolute_joint_margin_rad": _percentiles(revolute_margins),
        "geometry_check_label": GEOMETRY_LABEL,
        "geometry_evaluations": evaluations,
        "geometry_contacts": contacts,
        "episode_reports": list(episodes),
    }
