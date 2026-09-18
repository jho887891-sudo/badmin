"""The speed bound, the duty cycle, and the record that makes a measurement interpretable.

A racket-head speed on its own is not a result. Three things have to travel with it:

  THE BOUND      omega * reach from the real motors. Not a prediction - an upper bound that ignores joint
                 coupling and stops at the flange - but enough to say whether a number is plausible. The
                 shipped asset permitted 153 m/s, which is the size of the error this work exists to remove.

  THE DUTY CYCLE The official testing terminated EVERY run on motor-2 overheating: 2.5 kg at 40 minutes, a
                 1.5 kg hover at full reach in 3 minutes, 1.5 kg at 70 percent reach in 18 minutes. A swing
                 test that ignores this is measuring a duty cycle the arm cannot sustain.

  THE BIASES     Two are known and both push the same way. The arm is modelled at 61 percent of its real mass
                 (Task 4), so it accelerates faster than the real one; and the racket mass and inertia are
                 UNKNOWN (Task 6). A measured speed is therefore an OVERESTIMATE, and saying so is part of
                 reporting it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .joint_limits import rad_s_from_rpm
from .motor_mass import motor_mass_audit
from .racket import racket_mass_properties

# README_zh.md:176, the B601-DM maximum reach.
REACH_M = 0.767

_DM4310_NO_LOAD_RAD_S = rad_s_from_rpm(200.0)
_DM4340_NO_LOAD_RAD_S = rad_s_from_rpm(52.5)

_VARIANT_FACTORS = {"real": 1.0, "recommended_70": 0.70}


def limit_variant_factor(variant: str) -> float:
    """The fraction of the real no-load speed a patched asset enforces."""
    try:
        return _VARIANT_FACTORS[str(variant)]
    except KeyError:
        raise ValueError(
            "unknown limit variant " + repr(variant) + "; expected one of "
            + ", ".join(sorted(_VARIANT_FACTORS))
        ) from None


def tip_speed_bound(omega_rad_s: float, reach_m: float = REACH_M) -> float:
    """Upper bound on the flange speed for a joint rate, as omega * reach.

    An over-estimate in two directions at once: it assumes one joint carries all the motion and that the arm
    is a rigid radius, whereas real joint coupling makes the tip slower. It stops at the flange, so a racket
    beyond it moves faster still, not slower.
    """
    if omega_rad_s < 0:
        raise ValueError("angular speed must not be negative, got " + repr(omega_rad_s))
    if reach_m <= 0:
        raise ValueError("reach must be positive, got " + repr(reach_m))
    return float(omega_rad_s) * float(reach_m)


def bound_at_fraction(fraction: float) -> float:
    """The bound for the fastest joint at a fraction of its real no-load speed."""
    if fraction <= 0:
        raise ValueError("fraction must be positive, got " + repr(fraction))
    return tip_speed_bound(_DM4310_NO_LOAD_RAD_S * fraction)


@dataclass
class DutyCycle:
    """What a run asks the arm to do, checked against the published recommendations.

    Defaults are inside them. Anything outside raises at construction: the official testing is specific enough
    that an unsustainable duty cycle is a mistake rather than a choice.
    """

    payload_kg: float = 1.0
    reach_fraction: float = 0.60
    speed_fraction: float = 0.60
    ambient_c: float = 25.0
    evidence: tuple = (
        "Performance_Testing_zh.md:59 - 1.5 kg, 5-70 percent reach reciprocating, ran > 2 h and was stopped by",
        "an operator when motor 2 reached 90 C",
        "Performance_Testing_zh.md:60 - 2.5 kg under the same motion tripped motor 2 thermal protection in",
        "40 minutes",
        "Performance_Testing_zh.md:68 - a 1.5 kg HOVER at full reach tripped thermal protection in 3 minutes",
        "Performance_Testing_zh.md:85-88 - official recommendation: payload < 1.5 kg, working radius < 70",
        "percent of reach, speed < 70 percent of maximum, ambient 15-35 C",
    )

    MAX_PAYLOAD_KG = 1.5
    MAX_REACH_FRACTION = 0.70
    MAX_SPEED_FRACTION = 0.70

    def __post_init__(self) -> None:
        if self.payload_kg >= self.MAX_PAYLOAD_KG:
            raise ValueError(
                "payload " + repr(self.payload_kg) + " kg is at or above the recommended maximum of "
                + repr(self.MAX_PAYLOAD_KG) + " kg; the official testing shows the arm cannot sustain it"
            )
        if self.reach_fraction >= self.MAX_REACH_FRACTION:
            raise ValueError(
                "reach fraction " + repr(self.reach_fraction) + " is at or beyond the recommended "
                + repr(self.MAX_REACH_FRACTION) + "; motor 2 protected itself in 3 minutes at full reach"
            )
        if self.speed_fraction > self.MAX_SPEED_FRACTION:
            raise ValueError(
                "speed fraction " + repr(self.speed_fraction) + " exceeds the recommended "
                + repr(self.MAX_SPEED_FRACTION)
            )
        if not 15.0 <= self.ambient_c <= 35.0:
            raise ValueError("ambient " + repr(self.ambient_c) + " C is outside the recommended 15-35 C")

    @property
    def is_compliant(self) -> bool:
        return (
            self.payload_kg < self.MAX_PAYLOAD_KG
            and self.reach_fraction < self.MAX_REACH_FRACTION
            and self.speed_fraction <= self.MAX_SPEED_FRACTION
            and 15.0 <= self.ambient_c <= 35.0
        )


DEFAULT_DUTY_CYCLE = DutyCycle()


def known_biases() -> list:
    """Everything known to push a measured speed away from the truth, with its direction."""
    audit = motor_mass_audit()
    return [
        "arm mass: modelled at "
        + "{:.0%}".format(audit["modelled_arm_mass_kg"] / audit["specified_arm_mass_kg"])
        + " of the real arm (" + "{:.4f}".format(audit["modelled_arm_mass_kg"]) + " kg against "
        + "{:.1f}".format(audit["specified_arm_mass_kg"])
        + " kg), so lighter distal links accelerate faster and the measured speed is OPTIMISTIC",
        "racket mass and inertia are UNKNOWN, so the racket contributes no inertia at all and the measured "
        "speed is OPTIMISTIC",
        "the bound is omega * reach, which ignores joint coupling and therefore OVERSTATES the flange speed",
    ]


def experiment_record(*, limit_variant: str, torque_convention: str, duty_cycle: DutyCycle,
                      measured_peak_m_s: float) -> dict:
    """The minimum that must travel with a measured speed for it to mean anything."""
    factor = limit_variant_factor(limit_variant)
    bound = bound_at_fraction(factor)
    inside = measured_peak_m_s <= bound
    props = racket_mass_properties()
    return {
        "limit_variant": limit_variant,
        "limit_variant_factor": factor,
        "torque_convention": torque_convention,
        "duty_cycle": asdict(duty_cycle),
        "measured_peak_m_s": measured_peak_m_s,
        "bound_m_s": bound,
        "inside_bound": inside,
        "racket_mass_status": props["status"],
        "known_biases": known_biases(),
        "verdict": (
            "inside the bound"
            if inside
            else "ABOVE the bound: the asset is still wrong, not the arm fast. A speed beyond "
            + "{:.2f}".format(bound)
            + " m/s cannot come from these motors and means the limits are not the ones loaded."
        ),
    }
