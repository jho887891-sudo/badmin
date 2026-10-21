# -*- coding: utf-8 -*-
"""P8-03: difficulty curriculum -- level -> parameter ranges, split-gated BY CONSTRUCTION.

A curriculum level declares (a) a name, (b) an ordinal level number, (c) which seed split it
belongs to, and (d) the parameter ranges that define it.  The split is not advice: a level whose
split is "val" or "test" can never be handed to training code, because

  * CurriculumSchedule.training_levels() returns only split == "train" levels, and
  * CurriculumSchedule.require_training_level(name) raises CurriculumSplitViolation for any
    validation/test level, so a training manifest cannot be assembled from one by accident,
  * episodes_from_ledger() (evaluation.py) refuses an episode list whose level split differs
    from the requested seed split.

Parameter sampling is deterministic: numpy default_rng(seed) with the parameter names sorted, so
the same (level, seed) always yields the same parameters on every machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .seed_ledger import SPLITS

CURRICULUM_VERSION = "p8-curriculum-v0.1"
TRAIN_SPLIT = "train"
EVAL_SPLITS: Tuple[str, ...] = ("val", "test")


class CurriculumError(ValueError):
    """Base class for curriculum refusals."""


class CurriculumDefinitionError(CurriculumError):
    """The schedule itself is malformed (duplicate names, mixed parameters, ...)."""


class CurriculumSplitViolation(CurriculumError):
    """A validation/test level was requested for training (or vice versa)."""


@dataclass(frozen=True)
class ParameterRange:
    """A closed interval for one curriculum parameter, with its unit."""
    name: str
    lo: float
    hi: float
    unit: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise CurriculumDefinitionError("parameter name must be non-empty")
        lo, hi = float(self.lo), float(self.hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise CurriculumDefinitionError(
                "parameter {0!r}: bounds must be finite, got [{1}, {2}]".format(self.name, lo, hi))
        if hi < lo:
            raise CurriculumDefinitionError(
                "parameter {0!r}: hi ({1}) must be >= lo ({2})".format(self.name, hi, lo))

    def contains(self, value: float, tol: float = 1e-12) -> bool:
        return (float(self.lo) - tol) <= float(value) <= (float(self.hi) + tol)

    def sample(self, rng: np.random.Generator) -> float:
        if float(self.hi) == float(self.lo):
            return float(self.lo)
        return float(self.lo) + (float(self.hi) - float(self.lo)) * float(rng.random())

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lo": float(self.lo), "hi": float(self.hi), "unit": self.unit}


@dataclass(frozen=True)
class CurriculumLevel:
    """One rung of the curriculum.  split decides which seed set it may be paired with."""
    name: str
    level: int
    split: str
    parameters: Tuple[ParameterRange, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise CurriculumDefinitionError("level name must be non-empty")
        if self.split not in SPLITS:
            raise CurriculumSplitViolation(
                "level {0!r}: split must be one of {1}, got {2!r}"
                .format(self.name, SPLITS, self.split))
        if int(self.level) < 1:
            raise CurriculumDefinitionError(
                "level {0!r}: ordinal level must be >= 1, got {1}".format(self.name, self.level))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise CurriculumDefinitionError(
                "level {0!r} declares a parameter twice: {1}".format(self.name, names))
        if not names:
            raise CurriculumDefinitionError("level {0!r} declares no parameters".format(self.name))

    @property
    def is_training_level(self) -> bool:
        return self.split == TRAIN_SPLIT

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def range_for(self, name: str) -> ParameterRange:
        for p in self.parameters:
            if p.name == name:
                return p
        raise CurriculumDefinitionError("level {0!r} has no parameter {1!r}"
                                        .format(self.name, name))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": int(self.level), "split": self.split,
                "description": self.description,
                "parameters": [p.to_dict() for p in self.parameters]}


@dataclass(frozen=True)
class CurriculumSchedule:
    """An ordered set of levels and the split gate that keeps val/test out of training."""
    task_family: str
    levels: Tuple[CurriculumLevel, ...]
    version: str = CURRICULUM_VERSION

    def __post_init__(self) -> None:
        if not self.task_family:
            raise CurriculumDefinitionError("task_family must be non-empty")
        names = [lv.name for lv in self.levels]
        if not names:
            raise CurriculumDefinitionError("curriculum declares no levels")
        if len(set(names)) != len(names):
            raise CurriculumDefinitionError("duplicate curriculum level names: {0}".format(names))
        reference = self.levels[0].parameter_names
        for lv in self.levels:
            if lv.parameter_names != reference:
                raise CurriculumDefinitionError(
                    "levels must declare the same parameter names; {0!r} has {1}, {2!r} has {3}"
                    .format(self.levels[0].name, reference, lv.name, lv.parameter_names))
        if not any(lv.is_training_level for lv in self.levels):
            raise CurriculumDefinitionError("curriculum declares no training level")

    # -- split gate ----------------------------------------------------------------
    def level(self, name: str) -> CurriculumLevel:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise CurriculumDefinitionError("unknown curriculum level {0!r}".format(name))

    def training_levels(self) -> Tuple[CurriculumLevel, ...]:
        """Only the levels a training run is allowed to see."""
        return tuple(lv for lv in self.levels if lv.is_training_level)

    def eval_levels(self) -> Tuple[CurriculumLevel, ...]:
        return tuple(lv for lv in self.levels if not lv.is_training_level)

    def require_training_level(self, name: str) -> CurriculumLevel:
        """Return a training level or REFUSE: validation/test levels never enter training."""
        lv = self.level(name)
        if not lv.is_training_level:
            raise CurriculumSplitViolation(
                "level {0!r} belongs to split {1!r} and can never be used for training; "
                "training levels are {2}"
                .format(name, lv.split, tuple(v.name for v in self.training_levels())))
        return lv

    def require_eval_level(self, name: str) -> CurriculumLevel:
        lv = self.level(name)
        if lv.is_training_level:
            raise CurriculumSplitViolation(
                "level {0!r} is a training level and cannot be reported as a validation/test "
                "result".format(name))
        return lv

    # -- sampling ------------------------------------------------------------------
    def sample(self, name: str, seed: int) -> Dict[str, float]:
        """Deterministic parameter draw for (level, seed) -- sorted names, fixed rng order."""
        lv = self.level(name)
        rng = np.random.default_rng(int(seed))
        out: Dict[str, float] = {}
        for p in sorted(lv.parameters, key=lambda r: r.name):
            value = p.sample(rng)
            if not p.contains(value):
                raise CurriculumDefinitionError(
                    "level {0!r} sampled {1}={2} outside [{3}, {4}]"
                    .format(name, p.name, value, p.lo, p.hi))
            out[p.name] = value
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "task_family": self.task_family,
                "levels": [lv.to_dict() for lv in self.levels],
                "training_levels": [lv.name for lv in self.training_levels()],
                "eval_levels": [lv.name for lv in self.eval_levels()]}


def _r(name: str, lo: float, hi: float, unit: str = "1") -> ParameterRange:
    return ParameterRange(name=name, lo=lo, hi=hi, unit=unit)


def default_schedule() -> CurriculumSchedule:
    """Reference curriculum for the synthetic contact-targeting task family.

    Train levels L1..L3 only ever see the train seed split.  V1/V2 are the validation rungs
    (edge-of-workspace targets and a tight reachability budget, i.e. the FSM must refuse many of
    them) and T1/T2 are the test rungs (workspace corners, tight timing).  No parameter range is
    a statement about the real court: these are synthetic episode ranges.
    """
    params_near = (_r("target_x_m", -1.20, -0.80, "m"), _r("target_y_m", -0.40, 0.40, "m"),
                   _r("target_z_m", 0.45, 0.75, "m"), _r("reach_radius_m", 1.60, 2.60, "m"))
    params_mid = (_r("target_x_m", -1.80, -1.00, "m"), _r("target_y_m", -0.80, 0.80, "m"),
                  _r("target_z_m", 0.35, 0.95, "m"), _r("reach_radius_m", 1.60, 2.60, "m"))
    params_far = (_r("target_x_m", -2.20, -0.60, "m"), _r("target_y_m", -1.20, 1.20, "m"),
                  _r("target_z_m", 0.35, 0.95, "m"), _r("reach_radius_m", 1.60, 2.60, "m"))
    params_edge = (_r("target_x_m", -2.40, -1.80, "m"), _r("target_y_m", 1.20, 1.40, "m"),
                   _r("target_z_m", 0.90, 1.00, "m"), _r("reach_radius_m", 1.60, 2.60, "m"))
    params_tight = (_r("target_x_m", -2.20, -1.20, "m"), _r("target_y_m", -1.00, 1.00, "m"),
                    _r("target_z_m", 0.35, 0.95, "m"), _r("reach_radius_m", 0.60, 1.10, "m"))
    params_corner = (_r("target_x_m", -2.55, -2.20, "m"), _r("target_y_m", -1.45, -1.20, "m"),
                     _r("target_z_m", 0.20, 0.30, "m"), _r("reach_radius_m", 1.60, 2.60, "m"))
    params_late = (_r("target_x_m", -1.60, -0.60, "m"), _r("target_y_m", -0.60, 0.60, "m"),
                   _r("target_z_m", 0.55, 0.85, "m"), _r("reach_radius_m", 2.40, 2.60, "m"))
    levels = (
        CurriculumLevel("L1-near", 1, "train", params_near, "close targets, generous reach budget"),
        CurriculumLevel("L2-mid", 2, "train", params_mid, "mid targets"),
        CurriculumLevel("L3-far", 3, "train", params_far, "far targets, wide lateral spread"),
        CurriculumLevel("V1-edge", 4, "val", params_edge, "workspace edge, high racket z"),
        CurriculumLevel("V2-tight-reach", 5, "val", params_tight,
                        "reachability budget below the target distance: FSM must refuse"),
        CurriculumLevel("T1-corner", 6, "test", params_corner, "workspace corner target"),
        CurriculumLevel("T2-late", 7, "test", params_late, "late contact, tight xy spread"),
    )
    return CurriculumSchedule(task_family="racket_contact_targeting", levels=levels)


__all__ = [
    "CURRICULUM_VERSION", "TRAIN_SPLIT", "EVAL_SPLITS", "CurriculumError",
    "CurriculumDefinitionError", "CurriculumSplitViolation", "ParameterRange",
    "CurriculumLevel", "CurriculumSchedule", "default_schedule",
]
