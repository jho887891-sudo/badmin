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

Drive modes (BADMINTON_ROBOT.md S9):
  BODY_TWIST_ACTUATOR      the actuator consumes the court-frame body twist (vx, vy, wz).
                           S10 inverse kinematics is still evaluated and exposed, but only as a
                           diagnostic: the adapter keeps no steering state (nothing to optimise).
  STEER_DRIVE_WHEEL_MODEL  the actuator consumes steer angle (N,4) + wheel rate (N,4).  The S11
                           minimum-steering solution is applied against the remembered steering
                           angle of each wheel, so a reversal is realised by spinning the wheel
                           backwards instead of rotating the steer module by pi.

Wheel order is the frozen topology order of the canonical kinematics module:
(FL, FR, RL, RR) - every (N,4) array in this module uses it.

Geometry policy (S12): the chassis geometry is NOT measured.  wheel_positions_param defaults
to an explicitly TEMP parameterised proxy and wheel_radius_param reuses the Param of
badminton_robot_cfg.MorphOneGeometry (also TEMP); neither is presented as measured.

Feedback policy: a pure adapter has no sensor, so Feedback.prediction_error carries the only
error it can honestly observe - the realisation residual between the commanded base twist and the
twist reconstructed from the emitted wheel targets (wheel_targets_to_body_twist).  It is never a
measured prediction error; that value is REQUIRES_MEASUREMENT and is filled by the environment
adapter.  Feedback.contact_detected is a latched execution event flag (default False), set only
through SimExecutionAdapter.report_contact.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import ExecutionModule
from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, Feedback, Layer, SafeCommand

#: TEMP proxy wheel centres in robot_base (same values as
#: tests/simulation/robots/test_morph_one_kinematics.py).  The real geometry is
#: Param(None, REQUIRES_MEASUREMENT, 'wheel center coordinates in robot_base') in
#: badminton_robot_cfg.MorphOneGeometry and is NOT resolved by this module.
_TEMP_WHEEL_POSITIONS_M: Dict[str, Tuple[float, float]] = {
    'FL': (0.25, 0.20),
    'FR': (0.25, -0.20),
    'RL': (-0.25, 0.20),
    'RR': (-0.25, -0.20),
}
_TEMP_WHEEL_POSITIONS_SOURCE = (
    "TEMP proxy from docs/simulation/BADMINTON_ROBOT.md S10 (same proxy as "
    "tests/simulation/robots/test_morph_one_kinematics.py); chassis wheel centres are not measured")

#: S8: this adapter cannot measure prediction error - it only sees what it commanded.
SENSOR_FEEDBACK_SOURCE = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "sim execution adapter has no sensor; the measured prediction/contact signal comes from the "
    "environment adapter")


def _find_simulation_dir() -> Optional[Path]:
    """Locate <repo>/simulation (the canonical kinematics lives there)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'simulation' / 'robots' / 'badminton_robot' / 'morph_one'
        if (candidate / 'kinematics.py').is_file():
            return parent / 'simulation'
    return None


@lru_cache(maxsize=1)
def load_kinematics() -> Tuple[Any, Any]:
    """Return (robot cfg module, canonical four-steer kinematics module). No Isaac import."""
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

    Shapes: body_twist (3,), joint_position_target (6,), steer_angle_rad (4,),
    wheel_speed_rad_s (4,), wheel_tangential_speed_mps (4,) - all in the frozen
    wheel order (FL, FR, RL, RR).
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


class SimExecutionAdapter(ExecutionModule):
    """SafeCommand -> Morph One wheel/arm targets -> Feedback (pure numpy, no Isaac)."""
    layer = Layer.EXECUTION
    name = 'sim_execution_adapter'
    is_implemented = True

    def __init__(self, num_envs: int = 1, drive_mode: Any = None, wheel_positions: Any = None,
                 wheel_radius_m: Any = None, joint_names: Optional[Sequence[str]] = None) -> None:
        cfg, kinematics = load_kinematics()
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1, got " + str(num_envs))
        self.kinematics = kinematics
        self.cfg = cfg
        self.wheel_order: Tuple[Any, ...] = tuple(kinematics.WHEEL_ORDER)
        self.drive_mode = _coerce_drive_mode(
            cfg.DriveMode.BODY_TWIST_ACTUATOR if drive_mode is None else drive_mode, cfg)

        if wheel_positions is None:
            self.wheel_positions_param = Param(
                dict(_TEMP_WHEEL_POSITIONS_M), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                _TEMP_WHEEL_POSITIONS_SOURCE)
        else:
            self.wheel_positions_param = Param(
                dict(wheel_positions), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "wheel centres supplied by the caller for this run (chassis geometry is still "
                "REQUIRES_MEASUREMENT in badminton_robot_cfg)")
        self.wheel_positions = _normalise_wheel_positions(self.wheel_positions_param.value, cfg)

        if wheel_radius_m is None:
            # single source: never invent a second wheel radius
            radius_param = cfg.MorphOneCfg().geometry.wheel_radius_m
            if radius_param.value is None:
                raise ValueError(
                    "wheel_radius_m was not supplied and badminton_robot_cfg carries no value "
                    "(status=" + str(radius_param.status.value) + "); pass the measured radius")
            self.wheel_radius_param = radius_param
        else:
            value = float(wheel_radius_m)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("wheel_radius_m must be finite and > 0, got " + str(wheel_radius_m))
            self.wheel_radius_param = Param(
                value, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "wheel radius supplied by the caller for this run")
        self.wheel_radius_m = float(self.wheel_radius_param.value)

        if joint_names is not None:
            self.joint_names: Tuple[str, ...] = tuple(joint_names)
        else:
            # S13: the PiPER joint order is frozen by the robot config, never re-defined here
            self.joint_names = tuple(cfg.PiperCfg().joint_names)

        self._steer_state: List[Dict[Any, float]] = [
            {wheel: 0.0 for wheel in self.wheel_order} for _ in range(self.num_envs)]
        self._contact: np.ndarray = np.zeros((self.num_envs,), dtype=bool)
        self._last_command: List[Optional[ExecutionCommand]] = [None] * self.num_envs

    # ------------------------------------------------------------------ properties
    @property
    def wheel_level_authoritative(self) -> bool:
        """True iff this mode sends wheel-level targets to the robot (S9.2)."""
        return self.drive_mode is self.cfg.DriveMode.STEER_DRIVE_WHEEL_MODEL

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

        residual = np.zeros((num_envs,), dtype=float)
        for env_id in range(num_envs):
            record = self._build_record(env_id, twist[env_id], joints[env_id], command)
            self._last_command[env_id] = record
            realised = self.kinematics.wheel_targets_to_body_twist(
                self._wheel_target_map(record), self.wheel_positions, self.wheel_radius_m)
            residual[env_id] = float(np.linalg.norm(realised - record.body_twist))

        return Feedback(timestamp=float(command.timestamp), prediction_error=residual,
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
        self.drive_mode = _coerce_drive_mode(drive_mode, self.cfg)

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
           "SENSOR_FEEDBACK_SOURCE", "Mapping"]
