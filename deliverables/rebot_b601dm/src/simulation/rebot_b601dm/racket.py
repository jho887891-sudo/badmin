"""Attach ONE racket to the B601-DM flange, and define the frames downstream code will use.

TWO TRAPS, both found by reading the assets rather than assuming.

1. The GLB is not just a racket. Under one root it carries a Lamp, a whole shuttlecock (Gp_Shuttle with
   Obj_Feather and Obj_Cork), a racket (Obj_Racket with Obj_Strings) and a SECOND racket (Obj_Racket.001 with
   Obj_Strings.001). Attaching "the racket asset" wholesale would weld two rackets, a shuttlecock and a light
   onto the robot. This module selects one subtree and names the exclusions.

2. The gripper slides. gripper_joint1 and gripper_joint2 are prismatic, 0..0.0715 m. A racket on a free slider
   is not a fixed tool, so they are locked.

Attachment point: the URDF chain is link6 -> gripper_joint (fixed) -> gripper_link (lines 251-254), so the
B601-DM flange is gripper_link. BADMINTON_ROBOT.md:169 says "PiPER link6"; that is the same idea, the flange,
on a different arm, and the decision record keeps PiPER as the project identity.

Frames follow BADMINTON_ROBOT.md:182-184: +X the racket face normal, +Z from handle to head, +Y completing a
right-handed set.

MASS AND INERTIA ARE UNKNOWN. Not estimated, not copied from a badminton racket spec sheet: Task 4 already
established that a guessed inertial value is worse than an honest gap, because a racket whose mass is invented
makes every racket-head speed measured with it uninterpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RACKET_GLB = "assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb"

# The B601-DM flange, per the URDF chain link6 -> gripper_joint (fixed) -> gripper_link.
ATTACHMENT_PARENT = "gripper_link"

# A racket bolted to a sliding jaw is not a fixed tool. Locked, not merely preferred.
LOCKED_JOINTS = ("gripper_joint1", "gripper_joint2")

# Substrings that must never be attached: the shuttlecock, the second racket, the studio light.
_EXCLUDE_MARKERS = ("Shuttle", "Feather", "Cork", "Lamp", "Racket.001", "Strings.001")
# Substrings that identify the racket we keep.
_KEEP_MARKERS = ("Obj_Racket", "Obj_Strings")
# Transforms without which the mesh loses its scale and orientation.
_ROOT_MARKERS = ("Sketchfab_model", "Root")


@dataclass(frozen=True)
class Frame:
    """A named frame on the racket, with the axes downstream code may rely on."""

    name: str
    parent: str
    description: str
    translation: tuple = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)


RACKET_FRAMES = (
    Frame(
        name="racket_tcp",
        parent=ATTACHMENT_PARENT,
        description="tool centre point at the base of the racket handle where it meets the flange",
    ),
    Frame(
        name="racket_contact_frame",
        parent="racket_tcp",
        description="centre of the racket face, the reference centre of the main striking area",
    ),
)

_AXES = {
    "+X": "racket face normal (toward the opponent)",
    "+Z": "handle to head",
    "+Y": "right-handed completion of the set",
}


def frame_axes(name: str) -> dict:
    """The axis convention for a frame, from BADMINTON_ROBOT.md:182-184."""
    if name not in {f.name for f in RACKET_FRAMES}:
        raise KeyError("unknown racket frame: " + repr(name))
    return dict(_AXES)


@dataclass(frozen=True)
class RacketSelection:
    """Which nodes of the GLB to bring in, which to leave out, and why."""

    include: tuple = field(default_factory=tuple)
    exclude: tuple = field(default_factory=tuple)
    reason: str = ""


def select_racket_subtree(node_names) -> RacketSelection:
    """Choose exactly one racket from the asset, and name everything rejected.

    node_names is the list of node names in the GLB, so this works off the file rather than a hardcoded guess,
    and a future revision of the asset is handled by inspection instead of by surprise.
    """
    names = [str(n) for n in node_names]
    include, exclude = [], []
    for name in names:
        if any(marker in name for marker in _EXCLUDE_MARKERS):
            exclude.append(name)
        elif any(marker in name for marker in _KEEP_MARKERS) or any(m in name for m in _ROOT_MARKERS):
            include.append(name)
        else:
            exclude.append(name)
    return RacketSelection(
        include=tuple(include),
        exclude=tuple(exclude),
        reason=(
            "the asset holds a racket, a SECOND racket, a whole shuttlecock and a studio lamp under one root. "
            "Wielding the file wholesale would weld all of them to the flange, so one racket subtree is "
            "selected and the shuttle, the second racket and the lamp are excluded by name."
        ),
    )


def racket_mass_properties() -> dict:
    """Mass and inertia of the racket, which are UNKNOWN and stay that way until measured."""
    return {
        "mass_kg": None,
        "inertia": None,
        "status": "UNKNOWN",
        "source": "none; the asset is a mesh with no inertial data",
        "requires": [
            "mass of the actual racket to be used, weighed, in kg",
            "the inertia tensor about its centre of mass, or the geometry and density to compute it",
            "the transform from the flange to the racket centre of mass, which sets the moment arm",
        ],
        "note": (
            "racket mass and inertia directly determine racket-head speed, so an invented value here would "
            "make every speed measured with it uninterpretable. Left unknown deliberately."
        ),
    }


def attachment_plan(gltf_node_names=None) -> dict:
    """Everything needed to attach the racket, with the unknowns named."""
    selection = select_racket_subtree(gltf_node_names or [])
    return {
        "parent": ATTACHMENT_PARENT,
        "joint_type": "fixed",
        "glb": RACKET_GLB,
        "locked_joints": list(LOCKED_JOINTS),
        "selection": {"include": list(selection.include), "exclude": list(selection.exclude)},
        "frames": [f.name for f in RACKET_FRAMES],
        "mass_properties": racket_mass_properties(),
    }
