# -*- coding: utf-8 -*-
"""P5-04: an EXPLICIT joint-name -> runtime DOF index map, built only from the runtime name list.

WHY THIS EXISTS
    On the formal arm the three orders are all different:

      * USD stage traversal order  joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3,
                                   joint2, joint1, root_joint   (P5 F2, pxr export)
      * URDF / kinematic order     joint1 .. joint6, gripper_joint1, gripper_joint2
      * RUNTIME DOF order          whatever articulation.joint_names returns - the only order that
                                   indexes a joint_pos / joint_vel tensor or a PhysX write.

    A vector built by walking the USD stage (or by enumerating a dict literal) therefore looks
    perfectly consistent and still lands on the wrong joint. Every value in this repo that is
    addressed by index has to come through one explicit name -> index resolution, and that
    resolution must be driven by the RUNTIME list, never by asset traversal order.

WHAT THIS MODULE GUARANTEES
    JointIndex.build(runtime_names, required=...) either produces a map that is a bijection between
    the required name set and positions 0..len(runtime)-1, or it raises. It never guesses:

      * DuplicateJointName  a name occurs more than once in the runtime list (positions listed)
      * MissingJoint        a required name does not occur in the runtime list
      * ExtraJoint          the runtime list carries a name the model does not define

    Names are compared EXACTLY (no strip, no case folding, no aliasing). 'joint1 ' and 'joint1' are
    different strings, and an arm that reports 'Joint1' must be fixed at the source, not here: a
    fuzzy match would turn a silently mis-ordered DOF into a silently mis-ordered DOF.

ORDER INDEPENDENCE
    index_of() and reorder() are defined through the runtime list only, so shuffling the runtime
    list changes the indices and does NOT change which value belongs to which name. The tests prove
    that in both directions, including a reverse control built from the USD traversal order.

NO RUNTIME DEPENDENCIES
    No torch, numpy or pxr: the contract must be checkable on a plain box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


class JointIndexError(RuntimeError):
    """Base class: a name -> DOF index map cannot be built from the names actually reported."""

    code = "joint_index_error"


class DuplicateJointName(JointIndexError):
    """The runtime joint-name list contains the same name at two or more positions."""

    code = "duplicate_joint_name"


class MissingJoint(JointIndexError):
    """A required joint name is absent from the runtime joint-name list."""

    code = "missing_joint"


class ExtraJoint(JointIndexError):
    """The runtime joint-name list has a name the model does not define."""

    code = "extra_joint"


def _names(values: Iterable[object], what: str) -> Tuple[str, ...]:
    """Names are strings, verbatim. Nothing is stripped, folded or aliased."""
    names = tuple(value if isinstance(value, str) else str(value) for value in values)
    if not names:
        raise JointIndexError("%s is empty; cannot build a name -> DOF index map" % what)
    return names


def _duplicates(names: Sequence[str]) -> Dict[str, list]:
    seen: Dict[str, list] = {}
    for position, name in enumerate(names):
        seen.setdefault(name, []).append(position)
    return {name: positions for name, positions in seen.items() if len(positions) > 1}


@dataclass(frozen=True)
class JointIndex:
    """One bijection between joint NAMES and runtime DOF positions.

    names      the runtime joint-name list, in runtime DOF order (position i is DOF i)
    required   the name set the caller/model requires (defaults to names)
    model      free-text label for the arm this map belongs to (for error messages / reports)
    """

    names: Tuple[str, ...]
    required: Tuple[str, ...]
    model: str = ""

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, runtime_names: Sequence[object], required: Optional[Sequence[object]] = None,
              model: str = "") -> "JointIndex":
        """Build the map from the RUNTIME name list, or raise. Never falls back to a default order."""
        names = _names(runtime_names, "the runtime joint-name list")

        duplicates = _duplicates(names)
        if duplicates:
            detail = "; ".join(
                "%r at DOF positions %s" % (name, positions)
                for name, positions in sorted(duplicates.items()))
            raise DuplicateJointName(
                "the runtime joint-name list reports duplicate names: %s (full list %s). "
                "A duplicated DOF cannot be addressed by name - fix the articulation, do not pick "
                "one of the positions." % (detail, list(names)))

        if required is None:
            required_names = names
        else:
            required_names = _names(required, "the required joint-name set")
            required_duplicates = _duplicates(required_names)
            if required_duplicates:
                raise DuplicateJointName(
                    "the required joint-name set itself has duplicates: %s"
                    % sorted(required_duplicates))

        missing = [name for name in required_names if name not in names]
        extra = [name for name in names if name not in required_names]
        if missing or extra:
            raise cls._mismatch(names, required_names, missing, extra, model)

        # The required order is the model/table order; the index order is the runtime order. Keeping
        # both means reorder() can be checked against an independent order instead of against itself.
        return cls(names=names, required=required_names, model=model)

    @staticmethod
    def _mismatch(names, required_names, missing, extra, model) -> JointIndexError:
        parts = []
        if missing:
            parts.append("missing from the runtime list: %s" % missing)
        if extra:
            parts.append("not defined by the model: %s" % extra)
        message = ("joint names do not match%s: %s; runtime list %s, required %s"
                   % (" for %s" % model if model else "", "; ".join(parts),
                      list(names), list(required_names)))
        return MissingJoint(message) if missing else ExtraJoint(message)

    # -- the map -----------------------------------------------------------
    @property
    def dof_count(self) -> int:
        return len(self.names)

    @property
    def mapping(self) -> Dict[str, int]:
        """{joint name: runtime DOF index}. A copy, so a caller cannot edit the map in place."""
        return {name: position for position, name in enumerate(self.names)}

    def index_of(self, name: object) -> int:
        """The runtime DOF index of one name, or raise MissingJoint."""
        key = str(name)
        try:
            return self.names.index(key)
        except ValueError:
            raise MissingJoint(
                "joint %r is not in the runtime joint-name list %s%s; refusing to guess a DOF index"
                % (key, list(self.names), " of %s" % self.model if self.model else "")) from None

    def indices_of(self, names: Sequence[object]) -> Tuple[int, ...]:
        return tuple(self.index_of(name) for name in names)

    def reorder(self, values: Sequence[object], order: Optional[Sequence[object]] = None) -> Tuple:
        """Re-express values, given in 'order' (default: the required/model order), in runtime order.

        This is the ONLY sanctioned way to build a runtime-ordered vector out of a table: it goes
        through names, so the result cannot depend on the order the table happened to be written in.
        """
        order = tuple(self.required if order is None else (str(name) for name in order))
        values = tuple(values)
        if len(values) != len(order):
            raise JointIndexError(
                "reorder(): %d values for %d names %s" % (len(values), len(order), list(order)))
        out: list = [None] * self.dof_count
        for name, value in zip(order, values):
            # index_of() raises for a name the runtime does not have, so a value can never be
            # dropped on the floor (the old dict-comprehension bug) or land on a guessed index.
            position = self.index_of(name)
            if out[position] is not None:
                raise DuplicateJointName(
                    "reorder(): %r was mapped onto DOF %d twice" % (name, position))
            out[position] = value
        if any(value is None for value in out):
            raise MissingJoint(
                "reorder() left DOF positions unfilled for %s: runtime %s, given %s"
                % (self.model or "the arm", list(self.names), list(order)))
        return tuple(out)

    def as_mapping(self, values: Sequence[object],
                   order: Optional[Sequence[object]] = None) -> Dict[str, object]:
        """{name: value} for a runtime-ordered vector - the order-independent view of a command."""
        runtime = self.reorder(values, order) if order is not None else tuple(values)
        if len(runtime) != self.dof_count:
            raise JointIndexError(
                "as_mapping(): %d values for %d runtime DOFs %s"
                % (len(runtime), self.dof_count, list(self.names)))
        return {name: runtime[position] for position, name in enumerate(self.names)}

    def from_mapping(self, mapping: Mapping[str, object]) -> Tuple:
        """Runtime-ordered vector for a {name: value} mapping. Missing or extra keys both raise."""
        extra = [str(name) for name in mapping if str(name) not in self.names]
        missing = [name for name in self.names if name not in mapping]
        if extra and missing:
            raise JointIndexError(
                "from_mapping(): unknown names %s and missing names %s (runtime %s)"
                % (extra, missing, list(self.names)))
        if extra:
            raise ExtraJoint(
                "from_mapping(): names not in the runtime list: %s (runtime %s)"
                % (extra, list(self.names)))
        if missing:
            raise MissingJoint(
                "from_mapping(): no value for %s (runtime %s)" % (missing, list(self.names)))
        return tuple(mapping[name] for name in self.names)


__all__ = [
    "JointIndexError", "DuplicateJointName", "MissingJoint", "ExtraJoint", "JointIndex",
]
