"""P4-11: dependency-free wheel-drive command interface (P4_PROXY).

Why this file exists
--------------------
P4-08/P4-09 validated the wheel drive inside 'scripts/simulation/p4_proxy_wheel_drive.py', but
every command was built inline from argparse values with min()/max() calls.  That works for a
script and is unsafe for a robot: a unit mistake, a NaN, a zero radius or an over-limit command
silently becomes "some" wheel target instead of an error.  This module is the single, testable
place where a chassis command becomes four joint velocity targets, and it FAILS CLOSED: every bad
input raises a typed error, nothing is repaired silently.

Nothing here imports numpy, torch, Isaac or USD.  It is stdlib-only so the GPU script can import
it BEFORE the simulator launches (see the pre-flight in p4_proxy_wheel_drive.py).

Sign / axis convention (PROVEN ON GPU, do not "fix" it)
-------------------------------------------------------
* Every wheel joint is authored about world +X (P1 v2, verified to 0.0043 deg by PCA).
* A wheel spinning +omega about world +X drives the chassis along -Y: the contact point moves
  +Y, so the reaction on the chassis is -Y.  Rolling is therefore along Y, not X.
* A straight roll needs the SAME sign on all four wheels.  P1 v2's outward-normal pattern
  +1/+1/-1/-1 is a MOTOR-CHANNEL convention (per-side sign), NOT a joint-axis one; using it as
  joint targets made the chassis yaw 1.915 rad instead of translating (P4p_fwd_r02).
* Chassis speed convention used here: speed_m_s > 0 means travel along -Y, which needs
  omega = +speed / radius on all four wheels.  This is exactly the GPU feed-forward
  omega_ff = -vy_target / r with vy_target = -speed.
* Yaw: the wheel at lateral offset x must roll at the local surface speed psi_dot * x, so
  omega_i = -(psi_dot * x_i) / r.  P4p_turn_r03 ran targets [+3, +3, -3, -3] (i.e. psi_dot < 0)
  and measured a NEGATIVE yaw change of -1.4175 rad.

Evidence (committed GPU runs, outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/):
  * P4p_rollneg_r02: targets [4,4,4,4]     -> chassis dY = -1.2147367 m     (-Y travel)
  * P4p_rollpos_r02: targets [-4,-4,-4,-4] -> chassis dY = +1.215509057 m   (+Y travel)
  * P4p_turn_r03:    targets [3,3,-3,-3]   -> yaw 0 -> -1.417513835 rad
  * P4V_030_r10:     speed cmd 0.30 m/s    -> dY = -1.041028857 m, dtheta = +10.0358 rad
  * P4V_clamp_r11:   cmd 0.80 / limit 0.50 -> commanded_after_clamp_m_s = 0.50

Documented UNKNOWNs (do not read these as verified):
  * the wheel targets produced here are KINEMATIC; the real loaded radius, slip and motor
    torque/current limits are not modelled (P4_PROXY, not P4_REAL);
  * the four-wheel chassis must scrub laterally to turn (its wheels only roll along Y), so the
    achieved yaw rate is lower than the kinematic command: P4p_turn_r03 commanded
    psi_dot ~= -1.236 rad/s and its measured freeze-frame average was -0.472 rad/s.  The
    interface commands the kinematics, not the achieved yaw;
  * a degrees-vs-radians or mm-vs-m mistake whose magnitude still lands inside the valid range
    is indistinguishable and is NOT caught (see UNIT GUARDS below for what is caught).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

__all__ = [
    "AUDIT_SCHEMA",
    "AXLE_POSITIONS_M",
    "CommandValidationError",
    "EFFECTIVE_ROLLING_RADIUS_M",
    "HALF_TRACK_M",
    "InvalidLimitError",
    "InvalidLimitModeError",
    "InvalidRadiusError",
    "InvalidWheelCountError",
    "LIMIT_MODES",
    "NotFiniteError",
    "SpeedLimitExceededError",
    "UnitConfusionError",
    "WHEEL_COUNT",
    "WHEEL_LATERAL_X_M",
    "WHEEL_ORDER",
    "WHEEL_RADIUS_GEOM_M",
    "WheelCommand",
    "WheelRateError",
    "WheelTargetsOutOfRangeError",
    "YawLimitExceededError",
    "audit_round_trip",
]

# ---------------------------------------------------------------------------------------------
# Verified geometry (single source: P1 v2, mirrored from p4_proxy_wheel_drive.py)
# ---------------------------------------------------------------------------------------------
AXLE_POSITIONS_M = (
    (0.250002, 0.297, -0.244),
    (0.250002, -0.297, -0.244),
    (-0.250002, 0.297, -0.244),
    (-0.250002, -0.297, -0.244),
)
WHEEL_ORDER = ("wheel1", "wheel2", "wheel3", "wheel4")
WHEEL_COUNT = 4
#: Lateral (X) offset of each wheel = the lever arm that turns a yaw rate into a wheel speed.
#: Rolling is along Y, so the X spread is the TRACK and the Y spread is the wheelbase.
WHEEL_LATERAL_X_M = tuple(axle[0] for axle in AXLE_POSITIONS_M)
#: |x| used by WheelCommand.from_yaw_rate; 0.250002 m from P1 v2.
HALF_TRACK_M = 0.250002
#: Geometric (unloaded) tread radius, P1 v2 torus equator R + r = 103.000000 mm.
WHEEL_RADIUS_GEOM_M = 0.103
#: MEASURED effective rolling radius from the P4-08 matrix (the GPU feed-forward radius).
EFFECTIVE_ROLLING_RADIUS_M = 0.101459

ROLL_AXIS = "Y"
JOINT_AXIS = "X"
#: +omega about the joint axis produces chassis travel along -Y.
TRAVEL_SIGN_OF_POSITIVE_OMEGA = -1.0

LIMIT_MODES = ("clamp", "reject")
AUDIT_SCHEMA = "p4_11_wheel_command/v1"

# ---------------------------------------------------------------------------------------------
# Documented ranges used by the UNIT GUARDS.
#   * 50 rad/s is the hard actuator velocity_limit in the GPU script, so no legitimate rad/s
#     wheel target exists above it.
#   * 50 rad/s * 0.103 m = 5.15 m/s is therefore the mechanical speed ceiling of this chassis.
#   * 5.15 m/s / 0.250002 m = 20.6 rad/s is the same ceiling expressed as a yaw rate.
# Values ABOVE these ceilings cannot be a valid command in the declared unit, so they are
# rejected as unit confusion instead of being clamped into something plausible.
# ---------------------------------------------------------------------------------------------
MAX_WHEEL_RATE_RAD_S = 50.0
SPEED_CEILING_M_S = MAX_WHEEL_RATE_RAD_S * WHEEL_RADIUS_GEOM_M          # 5.15 m/s
YAW_CEILING_RAD_S = SPEED_CEILING_M_S / HALF_TRACK_M                    # ~20.6 rad/s
#: |v| > ceiling but <= this is almost certainly mm/s (300 mm/s -> 0.300 m/s).
SPEED_MM_S_MAX = SPEED_CEILING_M_S * 1000.0                             # 5150 mm/s
#: radius > this is almost certainly mm (103 mm -> 0.103 m).
RADIUS_MM_MIN, RADIUS_MM_MAX = 10.0, 1000.0
RADIUS_SANE_MIN_M, RADIUS_SANE_MAX_M = 0.005, 5.0
#: |yaw rate| > ceiling but <= this is almost certainly deg/s.
YAW_DEG_S_MAX = math.degrees(YAW_CEILING_RAD_S)                         # ~1180 deg/s
#: Wheel rate ceiling expressed in rpm (50 rad/s ~= 477.46 rpm).
WHEEL_RPM_MAX = MAX_WHEEL_RATE_RAD_S * 60.0 / (2.0 * math.pi)


# ---------------------------------------------------------------------------------------------
# Typed errors: every failure mode is its own class so a caller can branch without string tests.
# ---------------------------------------------------------------------------------------------
class CommandValidationError(ValueError):
    """Base class for every fail-closed rejection of a wheel command."""

    code = "command_invalid"


class NotFiniteError(CommandValidationError):
    """A value was NaN or +/-inf."""

    code = "not_finite"


class InvalidRadiusError(CommandValidationError):
    """The rolling radius was <= 0 or non-finite."""

    code = "invalid_radius"


class InvalidLimitError(CommandValidationError):
    """The speed/yaw limit was <= 0 or non-finite."""

    code = "invalid_limit"


class InvalidLimitModeError(CommandValidationError):
    """limit_mode was not one of LIMIT_MODES."""

    code = "invalid_limit_mode"


class InvalidWheelCountError(CommandValidationError):
    """The command did not carry exactly one target per wheel."""

    code = "invalid_wheel_count"


class WheelTargetsOutOfRangeError(CommandValidationError):
    """A wheel joint velocity target exceeded the actuator velocity limit."""

    code = "wheel_targets_out_of_range"


class WheelRateError(WheelTargetsOutOfRangeError):
    """Alias kept for callers written against the draft name."""


class SpeedLimitExceededError(CommandValidationError):
    """limit_mode='reject' and |speed| > limit: the command must abort."""

    code = "speed_limit_exceeded"


class YawLimitExceededError(CommandValidationError):
    """limit_mode='reject' and |yaw rate| > limit: the command must abort."""

    code = "yaw_limit_exceeded"


class UnitConfusionError(CommandValidationError):
    """A value that cannot be the declared unit (looks like mm, mm/s, deg/s or rpm)."""

    code = "unit_confusion"


# ---------------------------------------------------------------------------------------------
# Validation primitives.  Each one is deliberately narrow and raises, never repairs.
# ---------------------------------------------------------------------------------------------
def _require_finite(value, field_name):
    if isinstance(value, bool):
        raise NotFiniteError("%s must be a number, got bool %r" % (field_name, value))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NotFiniteError("%s must be a real number, got %r (%s)"
                             % (field_name, value, type(exc).__name__)) from exc
    if not math.isfinite(number):
        raise NotFiniteError("%s must be finite, got %r" % (field_name, number))
    return number


def _require_limit_mode(mode):
    if mode not in LIMIT_MODES:
        raise InvalidLimitModeError(
            "limit_mode must be one of %s, got %r (there is no default: clamping vs aborting "
            "must be an explicit decision)" % (list(LIMIT_MODES), mode))
    return mode


def _require_radius(radius_m):
    try:
        radius = _require_finite(radius_m, "radius_m")
    except NotFiniteError as exc:
        # field-level error class: "the radius was bad", whatever made it bad
        raise InvalidRadiusError(str(exc)) from exc
    if radius <= 0.0:
        raise InvalidRadiusError("radius_m must be > 0, got %r (radius 0 makes wheel speed "
                                 "infinite; there is no silent fallback)" % radius)
    if RADIUS_MM_MIN <= radius <= RADIUS_MM_MAX:
        raise UnitConfusionError(
            "radius_m=%r looks like a millimetre value: %.6f mm should be passed as %.6f m"
            % (radius, radius, radius / 1000.0))
    if not (RADIUS_SANE_MIN_M <= radius <= RADIUS_SANE_MAX_M):
        raise UnitConfusionError(
            "radius_m=%r is outside the documented range [%r, %r] m for this chassis"
            % (radius, RADIUS_SANE_MIN_M, RADIUS_SANE_MAX_M))
    return radius


def _require_limit(limit, field_name):
    try:
        value = _require_finite(limit, field_name)
    except NotFiniteError as exc:
        raise InvalidLimitError(str(exc)) from exc
    if value <= 0.0:
        raise InvalidLimitError("%s must be > 0, got %r (a zero/negative limit would either "
                                "forbid all motion or remove the limit)" % (field_name, value))
    return value


def _guard_speed_unit(speed_m_s):
    magnitude = abs(speed_m_s)
    if magnitude <= SPEED_CEILING_M_S:
        return speed_m_s
    if magnitude <= SPEED_MM_S_MAX:
        raise UnitConfusionError(
            "speed %r m/s is above the %.4f m/s chassis ceiling but matches mm/s: "
            "%.6f mm/s should be passed as %.6f m/s"
            % (speed_m_s, SPEED_CEILING_M_S, magnitude, speed_m_s / 1000.0))
    raise UnitConfusionError(
        "speed %r m/s is implausible in any unit for this chassis (ceiling %.4f m/s)"
        % (speed_m_s, SPEED_CEILING_M_S))


def _guard_yaw_unit(yaw_rate_rad_s):
    magnitude = abs(yaw_rate_rad_s)
    if magnitude <= YAW_CEILING_RAD_S:
        return yaw_rate_rad_s
    if magnitude <= YAW_DEG_S_MAX:
        raise UnitConfusionError(
            "yaw rate %r rad/s is above the %.4f rad/s chassis ceiling but matches deg/s: "
            "%.6f deg/s should be passed as %.6f rad/s"
            % (yaw_rate_rad_s, YAW_CEILING_RAD_S, magnitude,
               math.radians(yaw_rate_rad_s)))
    raise UnitConfusionError(
        "yaw rate %r rad/s is implausible in any unit for this chassis (ceiling %.4f rad/s)"
        % (yaw_rate_rad_s, YAW_CEILING_RAD_S))


def _guard_wheel_rate(rate_rad_s, wheel_name):
    magnitude = abs(rate_rad_s)
    if magnitude <= MAX_WHEEL_RATE_RAD_S + 1e-12:
        return rate_rad_s
    if magnitude <= WHEEL_RPM_MAX:
        raise UnitConfusionError(
            "%s target %r rad/s is above the %.1f rad/s actuator velocity limit but matches "
            "rpm: %.6f rpm should be passed as %.6f rad/s"
            % (wheel_name, rate_rad_s, MAX_WHEEL_RATE_RAD_S,
               magnitude * 60.0 / (2.0 * math.pi), rate_rad_s))
    raise WheelTargetsOutOfRangeError(
        "%s target %r rad/s is implausible in any unit (actuator velocity limit %.1f rad/s)"
        % (wheel_name, rate_rad_s, MAX_WHEEL_RATE_RAD_S))


# ---------------------------------------------------------------------------------------------
# The command object
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WheelCommand:
    """Four wheel joint velocity targets plus the provenance needed to audit them.

    All fields are validated in __post_init__; an instance either exists and is valid or the
    constructor raised.  There is no "partially valid" command object.
    """

    wheel_targets_rad_s: tuple
    speed_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    limit_mode: str = "reject"
    radius_m: float = EFFECTIVE_ROLLING_RADIUS_M
    requested_speed_m_s: float = 0.0
    requested_yaw_rate_rad_s: float = 0.0
    clamped: bool = False
    half_track_m: float = HALF_TRACK_M
    source: str = "manual"

    def __post_init__(self):
        object.__setattr__(self, "limit_mode", _require_limit_mode(self.limit_mode))
        object.__setattr__(self, "radius_m", _require_radius(self.radius_m))
        object.__setattr__(self, "half_track_m",
                           _require_limit(self.half_track_m, "half_track_m"))

        targets = self.wheel_targets_rad_s
        if isinstance(targets, (str, bytes)):
            raise InvalidWheelCountError("wheel_targets_rad_s must be a sequence of numbers, "
                                         "got %r" % (targets,))
        try:
            targets = tuple(targets)
        except TypeError as exc:
            raise InvalidWheelCountError("wheel_targets_rad_s must be a sequence, got %r"
                                         % (targets,)) from exc
        if len(targets) != WHEEL_COUNT:
            raise InvalidWheelCountError(
                "expected exactly %d wheel targets (%s) from the P1 v2 axle table, got %d: %r"
                % (WHEEL_COUNT, ", ".join(WHEEL_ORDER), len(targets), targets))
        checked = []
        for index, value in enumerate(targets):
            rate = _require_finite(value, "wheel_targets_rad_s[%d]" % index)
            checked.append(_guard_wheel_rate(rate, WHEEL_ORDER[index]))
        object.__setattr__(self, "wheel_targets_rad_s", tuple(checked))

        object.__setattr__(self, "speed_m_s", _require_finite(self.speed_m_s, "speed_m_s"))
        object.__setattr__(self, "yaw_rate_rad_s",
                           _require_finite(self.yaw_rate_rad_s, "yaw_rate_rad_s"))
        object.__setattr__(self, "requested_speed_m_s",
                           _require_finite(self.requested_speed_m_s, "requested_speed_m_s"))
        object.__setattr__(self, "requested_yaw_rate_rad_s",
                           _require_finite(self.requested_yaw_rate_rad_s,
                                           "requested_yaw_rate_rad_s"))
        if not isinstance(self.clamped, bool):
            raise CommandValidationError("clamped must be a bool, got %r" % (self.clamped,))

    # -- derived quantities ------------------------------------------------------------------
    @property
    def travel_direction(self):
        """'-Y' for a positive speed, '+Y' for a negative one, 'none' for zero."""
        if self.speed_m_s > 0.0:
            return "-Y"
        if self.speed_m_s < 0.0:
            return "+Y"
        return "none"

    @property
    def yaw_direction(self):
        """'+Z' (counter-clockwise seen from +Z) or '-Z', 'none' for zero."""
        if self.yaw_rate_rad_s > 0.0:
            return "+Z"
        if self.yaw_rate_rad_s < 0.0:
            return "-Z"
        return "none"

    def chassis_velocity_m_s(self):
        """Kinematic chassis velocity implied by the straight-roll command."""
        return (0.0, TRAVEL_SIGN_OF_POSITIVE_OMEGA * self.speed_m_s, 0.0)

    def joint_velocity_vector(self, joint_names):
        """Zero-filled joint-aligned vector with the four wheel joints filled in.

        Mirrors the GPU script: the chassis is not actuated, only joint_wheel<N> is.  A missing
        wheel joint raises instead of being silently skipped.
        """
        names = list(joint_names)
        vector = [0.0] * len(names)
        missing = []
        for wheel_name, target in zip(WHEEL_ORDER, self.wheel_targets_rad_s):
            joint_name = "joint_" + wheel_name
            if joint_name not in names:
                missing.append(joint_name)
                continue
            vector[names.index(joint_name)] = float(target)
        if missing:
            raise InvalidWheelCountError("wheel joints missing from the joint list: %s"
                                         % missing)
        return vector

    # -- constructors ------------------------------------------------------------------------
    @classmethod
    def from_speed(cls, v_m_s, radius_m, limit_m_s, limit_mode):
        """Straight roll: v_m_s > 0 travels along -Y and needs +v/r on all wheels.

        limit_mode='clamp' pins the magnitude to the limit and sets clamped=True;
        limit_mode='reject' raises SpeedLimitExceededError (fail closed).
        """
        mode = _require_limit_mode(limit_mode)
        radius = _require_radius(radius_m)
        limit = _require_limit(limit_m_s, "limit_m_s")
        speed = _guard_speed_unit(_require_finite(v_m_s, "v_m_s"))

        clamped = abs(speed) > limit
        if clamped:
            if mode == "reject":
                raise SpeedLimitExceededError(
                    "speed %r m/s exceeds limit %r m/s with limit_mode='reject': abort before "
                    "any simulation write (use limit_mode='clamp' to pin the magnitude instead)"
                    % (speed, limit))
            effective = math.copysign(limit, speed)
        else:
            effective = speed

        wheel_speed = effective / radius
        return cls(wheel_targets_rad_s=tuple([wheel_speed] * WHEEL_COUNT),
                   speed_m_s=effective, yaw_rate_rad_s=0.0, limit_mode=mode, radius_m=radius,
                   requested_speed_m_s=speed, requested_yaw_rate_rad_s=0.0, clamped=clamped,
                   source="from_speed")

    @classmethod
    def from_yaw_rate(cls, yaw_rate_rad_s, radius_m, limit_rad_s, limit_mode,
                      half_track_m=HALF_TRACK_M):
        """In-place turn: omega_i = -(psi_dot * x_i) / r with x_i from the P1 v2 axles.

        psi_dot > 0 is counter-clockwise seen from +Z and needs the x > 0 wheels to roll
        negatively.  P4p_turn_r03 recorded [+3, +3, -3, -3] and a measured yaw change of
        -1.4175 rad, i.e. the same convention with psi_dot < 0.
        """
        mode = _require_limit_mode(limit_mode)
        radius = _require_radius(radius_m)
        limit = _require_limit(limit_rad_s, "limit_rad_s")
        half_track = _require_limit(half_track_m, "half_track_m")
        psi_dot = _guard_yaw_unit(_require_finite(yaw_rate_rad_s, "yaw_rate_rad_s"))
        if half_track != HALF_TRACK_M:
            # the per-wheel lever arms come from AXLE_POSITIONS_M; a different half_track would
            # silently contradict them, so only the verified value is accepted
            raise InvalidLimitError(
                "half_track_m=%r contradicts the verified P1 v2 axle table (%.6f m); the lever "
                "arm is not a free parameter" % (half_track, HALF_TRACK_M))

        clamped = abs(psi_dot) > limit
        if clamped:
            if mode == "reject":
                raise YawLimitExceededError(
                    "yaw rate %r rad/s exceeds limit %r rad/s with limit_mode='reject': abort "
                    "before any simulation write (use limit_mode='clamp' to pin it instead)"
                    % (psi_dot, limit))
            effective = math.copysign(limit, psi_dot)
        else:
            effective = psi_dot

        targets = tuple(-(effective * x) / radius for x in WHEEL_LATERAL_X_M)
        return cls(wheel_targets_rad_s=targets, speed_m_s=0.0, yaw_rate_rad_s=effective,
                   limit_mode=mode, radius_m=radius, requested_speed_m_s=0.0,
                   requested_yaw_rate_rad_s=psi_dot, clamped=clamped,
                   half_track_m=HALF_TRACK_M, source="from_yaw_rate")

    @classmethod
    def from_audit(cls, audit):
        """audit() -> equal WheelCommand (the audit round trip)."""
        if not isinstance(audit, dict):
            raise CommandValidationError("audit must be a dict, got %r"
                                         % (type(audit).__name__,))
        if audit.get("schema") != AUDIT_SCHEMA:
            raise CommandValidationError("audit schema must be %r, got %r"
                                         % (AUDIT_SCHEMA, audit.get("schema")))
        targets = audit.get("wheel_targets_rad_s_per_wheel")
        if targets is None:
            raise CommandValidationError("audit is missing 'wheel_targets_rad_s_per_wheel'")
        if list(audit.get("wheel_order", ())) != list(WHEEL_ORDER):
            raise CommandValidationError("audit wheel_order %r does not match %r"
                                         % (audit.get("wheel_order"), list(WHEEL_ORDER)))
        return cls(wheel_targets_rad_s=tuple(float(t) for t in targets),
                   speed_m_s=float(audit["speed_effective_m_s"]),
                   yaw_rate_rad_s=float(audit["yaw_rate_effective_rad_s"]),
                   limit_mode=audit["limit_mode"],
                   radius_m=float(audit["radius_m"]),
                   requested_speed_m_s=float(audit["speed_requested_m_s"]),
                   requested_yaw_rate_rad_s=float(audit["yaw_rate_requested_rad_s"]),
                   clamped=bool(audit["clamped"]),
                   half_track_m=float(audit["half_track_m"]),
                   source=str(audit.get("source", "audit")))

    # -- audit -------------------------------------------------------------------------------
    def audit(self):
        """A plain-JSON dict safe to embed in a run manifest / result.json."""
        return {
            "schema": AUDIT_SCHEMA,
            "source": self.source,
            "limit_mode": self.limit_mode,
            "clamped": bool(self.clamped),
            "wheel_order": list(WHEEL_ORDER),
            "wheel_targets_rad_s_per_wheel": [float(t) for t in self.wheel_targets_rad_s],
            "speed_requested_m_s": float(self.requested_speed_m_s),
            "speed_effective_m_s": float(self.speed_m_s),
            "travel_direction": self.travel_direction,
            "chassis_velocity_m_s": [float(v) for v in self.chassis_velocity_m_s()],
            "yaw_rate_requested_rad_s": float(self.requested_yaw_rate_rad_s),
            "yaw_rate_effective_rad_s": float(self.yaw_rate_rad_s),
            "yaw_direction": self.yaw_direction,
            "radius_m": float(self.radius_m),
            "half_track_m": float(self.half_track_m),
            "wheel_lateral_x_m": [float(x) for x in WHEEL_LATERAL_X_M],
            "convention": {
                "joint_axis": "world +%s for all four wheels" % JOINT_AXIS,
                "roll_axis": "world %s (perpendicular to the axle)" % ROLL_AXIS,
                "positive_omega_travels": "%sY" % ("-" if TRAVEL_SIGN_OF_POSITIVE_OMEGA < 0
                                                   else "+"),
                "straight_roll": "same sign on all four wheels",
                "p1_outward_normal_signs_is_motor_channel_not_joint": [1.0, 1.0, -1.0, -1.0],
                "yaw_law": "omega_i = -(psi_dot * x_i) / r",
                "speed_law": "omega = +v / r for v > 0 (travel along -Y)",
            },
            "limits": {
                "max_wheel_rate_rad_s": MAX_WHEEL_RATE_RAD_S,
                "speed_ceiling_m_s": SPEED_CEILING_M_S,
                "yaw_ceiling_rad_s": YAW_CEILING_RAD_S,
            },
            "gpu_evidence": {
                "roll_neg_y": "P4p_rollneg_r02 targets [4,4,4,4] -> dY -1.2147367 m",
                "roll_pos_y": "P4p_rollpos_r02 targets [-4,-4,-4,-4] -> dY +1.215509057 m",
                "turn": "P4p_turn_r03 targets [3,3,-3,-3] -> yaw -1.417513835 rad",
                "speed_loop": "P4V_030_r10 cmd 0.30 m/s -> dY -1.041028857 m",
                "clamp": "P4V_clamp_r11 cmd 0.80 with limit 0.50 -> 0.50",
            },
            "unknowns": [
                "kinematic targets only: loaded radius, slip, motor torque/current not modelled",
                "achieved yaw rate is lower than commanded (four fixed axles must scrub)",
            ],
        }

    def to_json(self):
        """The audit dict serialised with sorted keys (stable, diffable manifests)."""
        return json.dumps(self.audit(), sort_keys=True, ensure_ascii=False)


def audit_round_trip(command):
    """Convenience: WheelCommand -> audit dict -> WheelCommand and assert equality."""
    restored = WheelCommand.from_audit(command.audit())
    if restored != command:
        raise CommandValidationError("audit round trip changed the command: %r != %r"
                                     % (restored, command))
    return restored
