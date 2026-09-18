"""Task 6: attach ONE racket to the flange, and define its frames.

Two traps this file exists to catch, both found by reading the assets rather than assuming:

1. THE GLB IS NOT JUST A RACKET. badminton_racket_and_shuttlecock_low_poly.glb contains, under one root:
     Lamp              a light
     Gp_Shuttle        a shuttlecock (Obj_Feather + Obj_Cork)
     Obj_Racket        a racket    (Obj_Racket_0 + Obj_Strings)
     Obj_Racket.001    a SECOND racket
   Attaching "the racket asset" wholesale would weld two rackets, a shuttlecock and a lamp onto the robot.

2. THE GRIPPER SLIDES. gripper_joint1 and gripper_joint2 are prismatic, 0..0.0715 m. A racket mounted on a
   free slider is not a fixed tool, so they are locked rather than left free.

Mass and inertia are UNKNOWN and stay that way. The plan forbids assuming them, and Task 4 already
established why a guessed mass is worse than an honest gap.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.simulation.rebot_b601dm.racket import (
    ATTACHMENT_PARENT,
    LOCKED_JOINTS,
    RACKET_FRAMES,
    RACKET_GLB,
    frame_axes,
    racket_mass_properties,
    select_racket_subtree,
)

GLB = Path(RACKET_GLB)
pytestmark = pytest.mark.skipif(not GLB.is_file(), reason="racket asset not present")


def _gltf():
    raw = GLB.read_bytes()
    off = 12
    while off < len(raw):
        clen, ctype = struct.unpack("<I4s", raw[off:off + 8])
        if ctype == b"JSON":
            return json.loads(raw[off + 8:off + 8 + clen].decode("utf-8"))
        off += 8 + clen
    raise AssertionError("no JSON chunk")


class TestItAttachesToOneFlangeOnly:
    def test_the_parent_is_the_b601dm_flange(self):
        """The URDF chain is link6 -> gripper_joint (fixed) -> gripper_link, lines 251-254."""
        assert ATTACHMENT_PARENT == "gripper_link"

    def test_the_gripper_slides_are_locked(self):
        assert set(LOCKED_JOINTS) == {"gripper_joint1", "gripper_joint2"}

    def test_a_second_racket_is_not_attached(self):
        gltf = _gltf()
        names = [str(n.get("name", "")) for n in gltf["nodes"]]
        twice = [n for n in names if n.startswith("Obj_Racket") and n.rstrip("0123456789_.") == "Obj_Racket"]
        assert len({"Obj_Racket", "Obj_Racket.001"} & set(names)) == 2, "the asset really does contain two"
        chosen = select_racket_subtree(names)
        # the precise statement: nothing from the SECOND racket is brought in. Counting names with the
        # "Obj_Racket" prefix would be wrong, because the chosen racket contributes its node AND its mesh.
        assert not [n for n in chosen.include if n.startswith("Obj_Racket.001")]
        assert not [n for n in chosen.include if n.startswith("Obj_Strings.001")]


class TestItDoesNotWeldAShuttlecockToTheRacket:
    def test_the_glb_really_contains_a_shuttlecock(self):
        gltf = _gltf()
        names = [str(n.get("name", "")) for n in gltf["nodes"]]
        assert any("Shuttle" in n for n in names)
        assert any("Feather" in n for n in names)

    def test_the_shuttlecock_is_excluded(self):
        chosen = select_racket_subtree([str(n.get("name", "")) for n in _gltf()["nodes"]])
        joined = " ".join(chosen.include)
        assert "Shuttle" not in joined
        assert "Feather" not in joined
        assert "Cork" not in joined

    def test_the_lamp_is_excluded(self):
        chosen = select_racket_subtree([str(n.get("name", "")) for n in _gltf()["nodes"]])
        assert not any("Lamp" in n for n in chosen.include)

    def test_the_exclusions_are_named_with_a_reason(self):
        chosen = select_racket_subtree([str(n.get("name", "")) for n in _gltf()["nodes"]])
        assert chosen.exclude
        assert "shuttle" in chosen.reason.lower()


class TestTheFramesFollowTheProjectSpec:
    def test_both_frames_are_defined(self):
        names = {f.name for f in RACKET_FRAMES}
        assert names == {"racket_tcp", "racket_contact_frame"}

    def test_the_axes_are_the_ones_badminton_robot_md_specifies(self):
        """BADMINTON_ROBOT.md:182-184: +X racket face normal, +Z handle to head, +Y right-handed."""
        axes = frame_axes("racket_tcp")
        assert axes["+X"] == "racket face normal (toward the opponent)"
        assert axes["+Z"] == "handle to head"
        assert "right-handed" in axes["+Y"]

    def test_the_contact_frame_is_the_face_centre(self):
        contact = next(f for f in RACKET_FRAMES if f.name == "racket_contact_frame")
        assert "face" in contact.description.lower()


class TestMassIsUnknownAndSaysSo:
    def test_mass_and_inertia_are_unknown(self):
        props = racket_mass_properties()
        assert props["mass_kg"] is None
        assert props["inertia"] is None
        assert props["status"] == "UNKNOWN"

    def test_the_properties_name_what_is_needed(self):
        props = racket_mass_properties()
        assert props["requires"]
        assert any("mass" in r.lower() for r in props["requires"])

    def test_nothing_here_pretends_a_racket_mass(self):
        props = racket_mass_properties()
        text = json.dumps(props).lower()
        assert "0.0" not in text.replace("0.05", ""), "no invented numeric mass"


class TestTheSelectionMatchesTheActualFile:
    def test_every_included_node_exists_in_the_glb(self):
        gltf = _gltf()
        names = [str(n.get("name", "")) for n in gltf["nodes"]]
        chosen = select_racket_subtree(names)
        for want in chosen.include:
            assert want in names, want

    def test_the_root_and_scene_wrapper_are_kept(self):
        """Without the root transform the mesh loses its scale and orientation."""
        chosen = select_racket_subtree([str(n.get("name", "")) for n in _gltf()["nodes"]])
        joined = " ".join(chosen.include)
        assert "Root" in joined or "Sketchfab_model" in joined

    def test_the_strings_belonging_to_the_chosen_racket_are_kept(self):
        chosen = select_racket_subtree([str(n.get("name", "")) for n in _gltf()["nodes"]])
        joined = " ".join(chosen.include)
        assert "Obj_Strings" in joined, "a racket without strings is a frame, not a racket"
        assert "Obj_Strings.001" not in joined, "that belongs to the rejected second racket"
