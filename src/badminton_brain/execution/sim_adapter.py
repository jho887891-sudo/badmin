# -*- coding: utf-8 -*-
"""T9 - simulation execution adapter: SafeCommand -> Morph One wheels + PiPER joints -> Feedback.

Spec:
  docs/superpowers/plans/2026-09-13-brain-modules.md (T9 row)
  docs/simulation/BADMINTON_ROBOT.md S9 (two drive modes), S10 (four-steer inverse kinematics),
                                    S11 (steer-angle optimisation), S24/S25 (robot command API)
  docs/architecture/ROBOT_BRAIN.md   S8 (feedback path), S11/S13 (one duty per layer)

Duties: this module is the only place where a SafeCommand becomes actuator-level targets.
It does NOT plan, does NOT clamp (Safety owns the limits) and does NOT call any Isaac/Kit/Omni
API - the real environment driver feeds SafeCommand in and reads Feedback out.

Frame convention (coordinator ruling with the T7 fix) - FROZEN:
  WholeBodyTarget.base_twist and SafeCommand.base_twist are the robot_base (body) frame twist
  [vx_body, vy_body, wz]: +x_body points forward, +y_body points left of the robot, wz is the
  yaw rate about +z_body.  The court -> body rotation is the planner's duty (T7, done inside
  planning); this adapter hands the numbers it receives straight to the body-frame four-steer
  inverse kinematics and never re-rotates them - equally it must never be fed a court-frame
  velocity.  Worked example: a robot at yaw = 90 deg with the court-frame intent (0.5, 0) m/s
  must be commanded the body twist (0.0, -0.5, 0.0).  Feeding (0.5, 0, 0) straight into the
  body IK steers every wheel wrong by exactly the yaw angle (90 deg); for a pure translation
  (wz = 0) the wheel-rate magnitudes coincide in both frames, so only the steer angle reveals
  the mistake - with wz != 0 the wheel rates differ too (TEMP geometry, wz = 0.5 rad/s: about
  35% on the worst wheel).  Regression guard: tests/badminton_brain/test_execution_adapter.py
  ::FrameConventionGuardTests.
  Caveat: the message-level frame attribute stays 'court' (the frozen check_court_frame
  contract in types.py checks every message), so the body frame of the base_twist VALUES can
  only be documented, not encoded in the message.

Drive modes (BADMINTON_ROBOT.md S9):
  BODY_TWIST_ACTUATOR      the actuator consumes the robot_base (body) frame twist
                           [vx_body, vy_body, wz] documented above.
                           S10 inverse kinematics is still evaluated and exposed, but only as a
                           diagnostic: the adapter keeps no steering state (nothing to optimise).
  STEER_DRIVE_WHEEL_MODEL  the actuator consumes steer angle (N,4) + wheel rate (N,4).  The S11
                           minimum-steering solution is applied against the remembered steering
                           angle of each wheel, so a reversal is realised by spinning the wheel
                           backwards instead of rotating the steer module by pi.

Wheel order is the frozen topology order of the canonical kinematics module:
(FL, FR, RL, RR) - every (N,4) array in this module uses it.

Geometry policy (S12 + DEC-017 single source, DEC-018 visibility):
  The robot configuration owns the geometry: MorphOneGeometry.wheel_positions_robot and
  MorphOneGeometry.wheel_radius_m of badminton_robot_cfg are read first (pass cfg=... to use a
  specific configuration; the default engineering configuration is used otherwise).  A value that
  the config really resolved is adopted with its own status/source and is NOT re-labelled.  Only
  when the config carries no value (wheel_positions_robot is REQUIRES_MEASUREMENT by design, value
  None) does the adapter fall back to an explicitly TEMP_PARAMETERIZED_PROXY engineering baseline,
  or to the config's own TEMP wheel radius - never to a second, unlabelled copy of the geometry.
  Resolution order for both inputs, identical for each: explicit Param > explicit raw value (kept
  TEMP: "caller-supplied ... still TEMP until measured and re-labelled") > resolved config value >
  labelled TEMP fallback.  An explicit None means exactly "not supplied" and follows the same path
  as omitting the argument.
  Whenever any geometry is still a proxy, the constructor raises a RuntimeWarning naming the TEMP
  fields (DEC-018), and the values stay queryable for the final-mode gate:
  param_limits() / unresolved_limits() / measurement_requirements() (same API as T5/T6/T7).
  Injecting measured geometry (a resolved cfg, or a Param with a measured status) removes the
  corresponding requirement.

Feedback policy (DEC-015) - two different residuals, never mixed up again:
  prediction_error is the PREDICTION residual (predicted shuttle position/velocity minus the
  measured one, metres / metres per second) and is consumed by the adaptation layer's drag/wind
  estimation.  It is produced by the estimation/prediction side, so this adapter leaves it UNSET
  (None): an execution module has no measurement and must never fabricate a prediction.
  tracking_residual (N,) is the EXECUTION tracking residual: how far the command that was
  actually sent stayed from the command that was requested.  Here it is the Euclidean distance
  in the robot_base frame between the commanded base twist and the twist reconstructed from the
  emitted wheel targets (kinematics.wheel_targets_to_body_twist): at most 1e-12 in floating
  point (measured ~4e-16) unless a wheel-level solution (S11) changes it.  Finite and per env.
  Feedback.contact_detected is a latched execution event flag (default False), set only through
  SimExecutionAdapter.report_contact.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import ExecutionModule
from ..status import AssetStatus, Param, UNRESOLVED_STATUSES
from ..types import BrainBoundaryError, Feedback, Layer, SafeCommand

#: Last-resort TEMP proxy wheel centres in robot_base, used only because the single source of
#: truth (badminton_robot_cfg.MorphOneGeometry.wheel_positions_robot) carries NO value: it is
#: Param(None, REQUIRES_MEASUREMENT, 'wheel center coordinates in robot_base'), and this module
#: never resolves it.  Same numbers as the phase-3 kinematics test proxy; giving cfg=... with a
#: resolved wheel_positions_robot replaces them entirely (DEC-017).
TEMP_WHEEL_POSITIONS_M: Dict[str, Tuple[float, float]] = {
    'FL': (0.25, 0.20),
    'FR': (0.25, -0.20),
    'RL': (-0.25, 0.20),
    'RR': (-0.25, -0.20),
}
TEMP_WHEEL_POSITIONS_SOURCE = (
    "TEMP engineering proxy from docs/simulation/BADMINTON_ROBOT.md S10 (same proxy as "
    "tests/simulation/robots/test_morph_one_kinematics.py): "
    "badminton_robot_cfg.MorphOneGeometry.wheel_positions_robot is REQUIRES_MEASUREMENT "
    "(value None), so the measured chassis geometry must be supplied through the robot config")

#: S8 + DEC-015: this adapter sees neither the shuttle nor its own realised state, so it can
#: report neither a prediction residual nor a sensor-based contact event.
SENSOR_FEEDBACK_SOURCE = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "sim execution adapter has no sensor: Feedback.prediction_error (predicted shuttle minus "
    "measurement) belongs to the estimation/prediction side and stays None here, and the contact "
    "event comes from the environment adapter")


def _find_simulation_dir() -> Optional[Path]:
    """Locate <repo>/simulation (the canonical kinematics lives there)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'simulation' / 'robots' / 'badminton_robot' / 'morph_one'
        if (candidate / 'kinematics.py').is_file():
            return parent / 'simulation'
    return None


@lru_cache(maxsize=1)
def load_kinematics() -> Tuple[Any, Any]:
    """Return (robot cfg module, canonical four-steer kinematics module). No Isaac import.

    Note: the canonical kinematics lives under <repo>/simulation as the package
    robots.badminton_robot.morph_one.kinematics, so this function adds <repo>/simulation to
    sys.path once.  The insertion is idempotent (checked before inserting) and process-global,
    exactly like ExpertPlanner._morph_one(); the result is cached, so the control cycle never
    touches sys.path again.
    """
    path = _find_simulation_dir()
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    errors: List[str] = []
    for module_name in ('robots.badminton_robot.badminton_robot_cfg',
                        'robots.badminton_robot.morph_one.kinematics'):
        try:
            importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on checkout layout
            errors.append(module_name + ': ' + str(exc))
    if errors:
        raise ImportError(
            "SimExecutionAdapter needs the canonical Morph One kinematics "
            "(simulation/robots/badminton_robot/morph_one/kinematics.py); " + '; '.join(errors))
    cfg = importlib.import_module('robots.badminton_robot.badminton_robot_cfg')
    kinematics = importlib.import_module('robots.badminton_robot.morph_one.kinematics')
    return cfg, kinematics


def _coerce_drive_mode(value: Any, cfg: Any) -> Any:
    """Accept a DriveMode member or its documented string value (S9)."""
    drive_mode = cfg.DriveMode
    if isinstance(value, drive_mode):
        return value
    if isinstance(value, str):
        try:
            return drive_mode(value)
        except ValueError as exc:
            raise ValueError(
                "unsupported drive_mode " + repr(value) + "; expected one of "
                + repr([member.value for member in drive_mode])) from exc
    raise ValueError(
        "drive_mode must be a DriveMode or its string value, got " + type(value).__name__)


def _adopt_geometry_param(name: str, supplied: Any, cfg_param: Any, fallback: Param) -> Param:
    """Resolve one geometry input, keeping its provenance intact (DEC-017).

    Order: explicit Param > explicit raw value (TEMP-labelled) > value resolved by the robot
    config > the supplied labelled TEMP fallback.  An explicit None means "not supplied".
    """
    if isinstance(supplied, Param):
        return supplied
    if supplied is not None:
        return Param(supplied, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                     "caller-supplied " + name + "; still TEMP until measured and re-labelled")
    if isinstance(cfg_param, Param) and cfg_param.value is not None:
        return cfg_param
    source = fallback.source
    if isinstance(cfg_param, Param):
        source = source + "; the configuration value is " + str(cfg_param.status.value)
    return Param(fallback.value, fallback.status, source)


def _normalise_wheel_positions(wheel_positions: Any, cfg: Any) -> Dict[Any, Tuple[float, float]]:
    """Map WheelId -> (x, y) in robot_base, rejecting incomplete/ill-formed geometry."""
    out: Dict[Any, Tuple[float, float]] = {}
    for wheel in cfg.WheelId:
        if wheel in wheel_positions:
            key = wheel
        elif wheel.value in wheel_positions:
            key = wheel.value
        else:
            raise ValueError("wheel_positions is missing " + wheel.value)
        xy = tuple(float(v) for v in wheel_positions[key])
        if len(xy) != 2:
            raise ValueError("wheel_positions[" + wheel.value + "] must be (x, y)")
        if not all(np.isfinite(xy)):
            raise ValueError("wheel_positions[" + wheel.value + "] must be finite")
        out[wheel] = xy
    return out


@dataclass(frozen=True)
class ExecutionCommand:
    """One environment's last executed command, in the frozen actuator vocabulary.

    body_twist is the robot_base (body) frame twist [vx_body, vy_body, wz] (module docstring:
    frame convention); steer/wheel arrays are body-frame wheel quantities in the frozen order.

    Shapes: body_twist (3,), joint_position_target (6,), steer_angle_rad (4,),
    wheel_speed_rad_s (4,), wheel_tangential_speed_mps (4,) - all in the frozen
    wheel order (FL, FR, RL, RR).  Every array is stored read-only, so the cached record cannot
    be mutated from outside, and the record is explicitly unhashable (it holds arrays).
    """
    env_id: int = 0
    drive_mode: Any = None
    timestamp: float = 0.0
    body_twist: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_position_target: np.ndarray = field(default_factory=lambda: np.zeros(6))
    steer_angle_rad: np.ndarray = field(default_factory=lambda: np.zeros(4))
    wheel_speed_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(4))
    wheel_tangential_speed_mps: np.ndarray = field(default_factory=lambda: np.zeros(4))
    wheel_level_authoritative: bool = False
    limited: bool = False
    violations: tuple = ()

    def __post_init__(self) -> None:
        # A cached command must be immutable: copy each array and freeze it, so a reader cannot
        # corrupt the record through the returned view.
        for name in ('body_twist', 'joint_position_target', 'steer_angle_rad',
                     'wheel_speed_rad_s', 'wheel_tangential_speed_mps'):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def actuator_target(self) -> Dict[str, np.ndarray]:
        """The quantities the drive mode actually sends to the robot."""
        if self.wheel_level_authoritative:
            return {'steer_angle_rad': self.steer_angle_rad,
                    'wheel_speed_rad_s': self.wheel_speed_rad_s}
        return {'body_twist': self.body_twist}

    def as_dict(self) -> Dict[str, Any]:
        """Flat record for logging / dataset handles."""
        return {
            'env_id': self.env_id,
            'drive_mode': getattr(self.drive_mode, 'value', self.drive_mode),
            'timestamp': self.timestamp,
            'body_twist': self.body_twist,
            'joint_position_target': self.joint_position_target,
            'steer_angle_rad': self.steer_angle_rad,
            'wheel_speed_rad_s': self.wheel_speed_rad_s,
            'wheel_tangential_speed_mps': self.wheel_tangential_speed_mps,
            'wheel_level_authoritative': self.wheel_level_authoritative,
            'limited': self.limited,
            'violations': self.violations,
        }


# numpy arrays have no meaningful value hash, so the record is explicitly unhashable instead of
# failing deep inside numpy with a confusing "unhashable type: 'numpy.ndarray'".
ExecutionCommand.__hash__ = None


class SimExecutionAdapter(ExecutionModule):
    """SafeCommand -> Morph One wheel/arm targets -> Feedback (pure numpy, no Isaac)."""
    layer = Layer.EXECUTION
    name = 'sim_execution_adapter'
    is_implemented = True

    def __init__(self, num_envs: int = 1, drive_mode: Any = None, wheel_positions: Any = None,
                 wheel_radius_m: Any = None, joint_names: Optional[Sequence[str]] = None,
                 *, cfg: Any = None) -> None:
        sim_module, kinematics = load_kinematics()
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1, got " + str(num_envs))
        self.sim_module = sim_module
        self.kinematics = kinematics
        # DEC-017: the robot configuration owns the geometry.  Pass cfg=... to hand in a specific
        # configuration (e.g. one whose wheel centres are measured); otherwise the default
        # engineering baseline configuration is used and its unresolved fields stay unresolved.
        self.cfg = cfg if cfg is not None else sim_module.make_default_robot_cfg()
        self.wheel_order: Tuple[Any, ...] = tuple(kinematics.WHEEL_ORDER)
        self.drive_mode = _coerce_drive_mode(
            sim_module.DriveMode.BODY_TWIST_ACTUATOR if drive_mode is None else drive_mode,
            sim_module)

        geometry = getattr(getattr(self.cfg, 'morph_one', None), 'geometry', None)
        # Both geometry inputs follow exactly the same path: an explicit None means "not supplied".
        self.wheel_positions_param = _adopt_geometry_param(
            'wheel_positions_robot', wheel_positions,
            getattr(geometry, 'wheel_positions_robot', None),
            Param(dict(TEMP_WHEEL_POSITIONS_M), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                  TEMP_WHEEL_POSITIONS_SOURCE))
        self.wheel_radius_param = _adopt_geometry_param(
            'wheel_radius_m', wheel_radius_m, getattr(geometry, 'wheel_radius_m', None),
            sim_module.MorphOneCfg().geometry.wheel_radius_m)
        self.wheel_positions = _normalise_wheel_positions(self.wheel_positions_param.value,
                                                          sim_module)
        self.wheel_radius_m = self._positive_radius(self.wheel_radius_param)

        if joint_names is not None:
            self.joint_names: Tuple[str, ...] = tuple(joint_names)
        else:
            # S13: the PiPER joint order is frozen by the robot config, never re-defined here
            piper = getattr(self.cfg, 'piper', None) or sim_module.PiperCfg()
            self.joint_names = tuple(piper.joint_names)

        self._steer_state: List[Dict[Any, float]] = [
            {wheel: 0.0 for wheel in self.wheel_order} for _ in range(self.num_envs)]
        self._contact: np.ndarray = np.zeros((self.num_envs,), dtype=bool)
        self._last_command: List[Optional[ExecutionCommand]] = [None] * self.num_envs

        # DEC-018: TEMP geometry must never be silent.
        self._warn_unresolved_geometry()

    # ------------------------------------------------------------------ parameter reporting
    def param_limits(self) -> Dict[str, Param]:
        """Every geometry parameter of this adapter as a Param, by name (T5/T6/T7 API)."""
        return {'wheel_positions_robot': self.wheel_positions_param,
                'wheel_radius_m': self.wheel_radius_param}

    def unresolved_limits(self) -> Tuple[Tuple[str, Param], ...]:
        """Geometry not backed by a real measurement, sorted by name (T5/T6/T7 API).

        A TEMP proxy or a value-less parameter shows up here, so the final-mode gate can see
        that this adapter still runs on an engineering baseline.
        """
        return tuple((name, param) for name, param in sorted(self.param_limits().items())
                     if param.status in UNRESOLVED_STATUSES or param.value is None)

    def measurement_requirements(self) -> Dict[str, Param]:
        """One Param(None, REQUIRES_MEASUREMENT, source) per geometry that is still unresolved."""
        return {name: Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                            'measure ' + name + ': ' + param.source)
                for name, param in self.unresolved_limits()}

    @staticmethod
    def _positive_radius(param: Param) -> float:
        if param.value is None:
            raise ValueError(
                "wheel_radius_m has no value: supply it or use a robot cfg whose "
                "MorphOneGeometry.wheel_radius_m is set (source: " + param.source + ")")
        value = float(param.value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("wheel_radius_m must be finite and > 0, got " + str(param.value))
        return value

    def _warn_unresolved_geometry(self) -> None:
        unresolved = self.unresolved_limits()
        if not unresolved:
            return
        warnings.warn(
            "[SimExecutionAdapter] running with TEMP_PARAMETERIZED_PROXY geometry: "
            + ", ".join(name for name, _ in unresolved)
            + "; measured Morph One wheel centres / radius are REQUIRES_MEASUREMENT "
              "(docs/simulation/BADMINTON_ROBOT.md S4/S12).",
            RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------ properties
    @property
    def wheel_level_authoritative(self) -> bool:
        """True iff this mode sends wheel-level targets to the robot (S9.2)."""
        return self.drive_mode is self.sim_module.DriveMode.STEER_DRIVE_WHEEL_MODEL

    # ------------------------------------------------------------------ API
    def process(self, command: SafeCommand) -> Feedback:
        """Consume the only message Execution may receive and return Feedback (S15)."""
        if not isinstance(command, SafeCommand):
            raise BrainBoundaryError(
                type(self).__name__ + ".process expects a SafeCommand, got "
                + type(command).__name__)
        twist = np.asarray(command.base_twist, dtype=float)
        joints = np.asarray(command.joint_position_target, dtype=float)
        num_envs = int(twist.shape[0])
        if num_envs != self.num_envs:
            raise BrainBoundaryError(
                type(self).__name__ + " was built for " + str(self.num_envs)
                + " environments but received a batch of " + str(num_envs))

        tracking_residual = np.zeros((num_envs,), dtype=float)
        for env_id in range(num_envs):
            record = self._build_record(env_id, twist[env_id], joints[env_id], command)
            self._last_command[env_id] = record
            realised = self.kinematics.wheel_targets_to_body_twist(
                self._wheel_target_map(record), self.wheel_positions, self.wheel_radius_m)
            tracking_residual[env_id] = float(np.linalg.norm(realised - record.body_twist))

        # DEC-015: report the execution tracking residual; prediction_error stays unset because
        # the prediction-vs-measurement residual is not the execution layer's to produce.
        return Feedback(timestamp=float(command.timestamp), prediction_error=None,
                        tracking_residual=tracking_residual,
                        contact_detected=self._contact.copy())

    def get_last_command(self, env_ids: Optional[Sequence[int]] = None
                         ) -> Tuple[Optional[ExecutionCommand], ...]:
        """Last executed command per environment (None where nothing was commanded yet)."""
        return tuple(self._last_command[env_id] for env_id in self._env_ids(env_ids))

    def get_steer_state(self, env_ids: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, ...]:
        """Remembered steering angle per wheel (S11 memory), in the frozen wheel order."""
        return tuple(
            np.asarray([self._steer_state[env_id][wheel] for wheel in self.wheel_order],
                       dtype=float)
            for env_id in self._env_ids(env_ids))

    def report_contact(self, env_ids: Sequence[int], detected: bool = True) -> None:
        """Latch an execution-level contact event for the given environments.

        The adapter has no sensor (see SENSOR_FEEDBACK_SOURCE): contact may only enter through
        this hook, and the latched flag is cleared by reset().
        """
        for env_id in self._env_ids(env_ids):
            self._contact[env_id] = bool(detected)

    def set_drive_mode(self, drive_mode: Any) -> None:
        """Switch S9 mode at runtime; the steering state survives (reset clears it)."""
        self.drive_mode = _coerce_drive_mode(drive_mode, self.sim_module)

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear the command cache / steering state / contact latch of the selected envs only."""
        for env_id in self._env_ids(env_ids):
            self._last_command[env_id] = None
            self._steer_state[env_id] = {wheel: 0.0 for wheel in self.wheel_order}
            self._contact[env_id] = False

    # ------------------------------------------------------------------ internals
    def _env_ids(self, env_ids: Optional[Sequence[int]]) -> List[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, (int, np.integer)):
            env_ids = [int(env_ids)]
        selected: List[int] = []
        for raw in env_ids:
            env_id = int(raw)
            if env_id < 0 or env_id >= self.num_envs:
                raise ValueError("env id " + str(env_id) + " out of range for "
                                 + str(self.num_envs) + " environments")
            selected.append(env_id)
        return selected

    def _wheel_target_map(self, record: ExecutionCommand) -> Dict[Any, Any]:
        return {wheel: self.kinematics.WheelTargets(record.steer_angle_rad[index],
                                                    record.wheel_speed_rad_s[index],
                                                    record.wheel_tangential_speed_mps[index])
                for index, wheel in enumerate(self.wheel_order)}

    def _build_record(self, env_id: int, twist: np.ndarray, joints: np.ndarray,
                      command: SafeCommand) -> ExecutionCommand:
        wheel_level = self.wheel_level_authoritative
        current = self._steer_state[env_id] if wheel_level else None
        targets = self.kinematics.body_twist_to_wheel_targets(
            twist, self.wheel_positions, self.wheel_radius_m, current_steer_rad=current)
        steer = np.asarray([targets[wheel].steer_angle_rad for wheel in self.wheel_order],
                           dtype=float)
        drive = np.asarray([targets[wheel].wheel_speed_rad_s for wheel in self.wheel_order],
                           dtype=float)
        tangential = np.asarray(
            [targets[wheel].tangential_speed_mps for wheel in self.wheel_order], dtype=float)
        if wheel_level:
            self._steer_state[env_id] = {wheel: float(steer[index])
                                         for index, wheel in enumerate(self.wheel_order)}
        return ExecutionCommand(
            env_id=env_id,
            drive_mode=self.drive_mode,
            timestamp=float(command.timestamp),
            body_twist=np.asarray(twist, dtype=float).copy(),
            joint_position_target=np.asarray(joints, dtype=float).copy(),
            steer_angle_rad=steer,
            wheel_speed_rad_s=drive,
            wheel_tangential_speed_mps=tangential,
            wheel_level_authoritative=wheel_level,
            limited=bool(command.limited),
            violations=tuple(command.violations),
        )


__all__ = ["SimExecutionAdapter", "ExecutionCommand", "load_kinematics",
           "SENSOR_FEEDBACK_SOURCE", "TEMP_WHEEL_POSITIONS_M", "TEMP_WHEEL_POSITIONS_SOURCE"]
