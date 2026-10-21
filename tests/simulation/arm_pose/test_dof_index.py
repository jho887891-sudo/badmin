# -*- coding: utf-8 -*-
"""P5-04 - the joint-name -> runtime DOF index map, and proof that it is order independent.

The trap this file is built around
    The B601-RS USD reports its joints in stage traversal order
    (joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3, joint2, joint1, root_joint)
    while the runtime articulation indexes them in DOF order. A vector assembled by walking the USD
    stage - or by trusting a dict literal - therefore looks consistent and lands on the wrong joint.
    The tests below fail if the map is ever derived from anything but the RUNTIME name list.

No Isaac, no torch, no numpy: dof_index.py is pure python and the USD facts come from the committed
pxr export (outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json).
"""
from __future__ import annotations

import pytest

from pose_data import REBOT_RS_JOINT_NAMES

REQUIRED = REBOT_RS_JOINT_NAMES  # the model table order, used as the 'required' name set


def _shuffled():
    """A plausible runtime order: wrist first, fingers last - nothing like the model table order."""
    return ("joint6", "joint5", "joint4", "joint3", "joint2", "joint1",
            "gripper_joint2", "gripper_joint1")


def _usd_order(usd_facts):
    """USD traversal order minus the fixed root joint: the 8 names, in an order that is NOT the DOF one."""
    return tuple(name for name in usd_facts["stage_traversal_order"] if name != "root_joint")


# --------------------------------------------------------------------------- 1. the map
def test_mapping_follows_the_runtime_list_verbatim():
    from badminton_scene.dof_index import JointIndex

    index = JointIndex.build(REQUIRED, required=REQUIRED, model="rebot_b601_rs")
    assert index.dof_count == 8
    assert index.mapping == {name: position for position, name in enumerate(REQUIRED)}
    assert index.names == REQUIRED
    assert index.required == REQUIRED


def test_a_shuffled_runtime_order_moves_the_indices_not_the_names():
    """The indices must be read off the runtime list, not off a canonical table."""
    from badminton_scene.dof_index import JointIndex

    shuffled = _shuffled()
    index = JointIndex.build(shuffled, required=REQUIRED, model="rebot_b601_rs")
    assert index.names == shuffled
    for position, name in enumerate(shuffled):
        assert index.index_of(name) == position, (name, position)
    # The canonical name set did not change, only where each name sits.
    assert set(index.mapping) == set(REQUIRED)
    assert index.index_of("joint1") == shuffled.index("joint1") != REQUIRED.index("joint1")


def test_reorder_is_order_independent_in_both_directions():
    """Same name -> same value, whichever order the runtime reports and the payload uses."""
    from badminton_scene.dof_index import JointIndex

    values = {name: float(10 + position) for position, name in enumerate(REQUIRED)}
    canonical = JointIndex.build(REQUIRED, required=REQUIRED)
    shuffled = JointIndex.build(_shuffled(), required=REQUIRED)

    payload_canonical = tuple(values[name] for name in REQUIRED)
    runtime_canonical = canonical.reorder(payload_canonical)
    runtime_shuffled = shuffled.reorder(payload_canonical)  # same payload, different runtime order

    assert canonical.as_mapping(runtime_canonical) == values
    assert shuffled.as_mapping(runtime_shuffled) == values
    # The runtime vectors differ (they are in different orders) but describe the same command.
    assert runtime_canonical != runtime_shuffled
    # And each vector places a name's value at that name's runtime index.
    for name, value in values.items():
        assert runtime_canonical[canonical.index_of(name)] == value
        assert runtime_shuffled[shuffled.index_of(name)] == value


def test_a_round_trip_through_mapping_and_back_is_lossless():
    from badminton_scene.dof_index import JointIndex

    index = JointIndex.build(_shuffled(), required=REQUIRED)
    values = {name: 0.25 * position for position, name in enumerate(REQUIRED)}
    assert index.as_mapping(index.from_mapping(values)) == values


# --------------------------------------------------------------------------- 2. the USD trap
def test_the_usd_traversal_order_is_not_the_runtime_order(usd_facts):
    """Reverse control: the asset order is recorded, different, and never used to build indices."""
    usd = _usd_order(usd_facts)
    assert set(usd) == set(REQUIRED), (usd, REQUIRED)
    assert usd != REQUIRED, "the USD facts no longer disagree with the DOF order - re-read P5 F2"

    from badminton_scene.dof_index import JointIndex

    index = JointIndex.build(REQUIRED, required=REQUIRED, model="rebot_b601_rs")
    distinct = tuple(float(position) for position in range(len(usd)))

    # A naive positional zip against the USD order would assign every value to the wrong joint.
    by_usd_position = dict(zip(usd, distinct))
    by_runtime_position = dict(zip(REQUIRED, distinct))
    assert by_usd_position != by_runtime_position

    # reorder(order=usd) is explicit about the order it is given, and agrees with the zip above:
    # the module never assumes that a payload is already in runtime order by accident.
    assert index.as_mapping(distinct, order=usd) == by_usd_position
    # With no order given, the payload is runtime-ordered by definition.
    assert index.as_mapping(distinct) == by_runtime_position


# --------------------------------------------------------------------------- 3. fail loudly
def test_duplicate_runtime_names_raise_and_name_the_positions():
    from badminton_scene.dof_index import DuplicateJointName, JointIndex

    duplicated = _shuffled()[:7] + ("joint3",)  # joint3 twice, gripper_joint1 gone
    with pytest.raises(DuplicateJointName) as excinfo:
        JointIndex.build(duplicated, required=REQUIRED)
    message = str(excinfo.value)
    assert "joint3" in message and "[3, 7]" in message, message
    assert excinfo.value.code == "duplicate_joint_name"


def test_a_required_name_missing_from_the_runtime_list_raises():
    from badminton_scene.dof_index import JointIndex, MissingJoint

    runtime = tuple(name for name in _shuffled() if name != "gripper_joint2")
    with pytest.raises(MissingJoint) as excinfo:
        JointIndex.build(runtime, required=REQUIRED, model="rebot_b601_rs")
    assert "gripper_joint2" in str(excinfo.value), str(excinfo.value)
    assert excinfo.value.code == "missing_joint"


def test_an_extra_runtime_dof_raises():
    from badminton_scene.dof_index import ExtraJoint, JointIndex

    with pytest.raises(ExtraJoint) as excinfo:
        JointIndex.build(REQUIRED + ("joint7",), required=REQUIRED, model="rebot_b601_rs")
    assert "joint7" in str(excinfo.value), str(excinfo.value)
    assert excinfo.value.code == "extra_joint"


def test_joint_names_are_matched_exactly_not_fuzzily():
    """No strip / lower / aliasing: 'joint1 ' is a different joint, and must fail as one."""
    from badminton_scene.dof_index import JointIndex, JointIndexError

    with pytest.raises(JointIndexError) as excinfo:
        JointIndex.build(REQUIRED[:-1] + ("joint1 ",), required=REQUIRED)
    message = str(excinfo.value)
    assert "joint1 " in message and "gripper_joint2" in message, message


def test_an_unknown_lookup_raises_instead_of_guessing_a_position():
    from badminton_scene.dof_index import JointIndex, MissingJoint

    index = JointIndex.build(REQUIRED, required=REQUIRED)
    with pytest.raises(MissingJoint) as excinfo:
        index.index_of("racket_twist")
    assert "racket_twist" in str(excinfo.value), str(excinfo.value)


def test_an_empty_name_list_raises():
    from badminton_scene.dof_index import JointIndex, JointIndexError

    with pytest.raises(JointIndexError):
        JointIndex.build((), required=REQUIRED)


def test_a_payload_with_the_wrong_width_raises_rather_than_being_truncated():
    from badminton_scene.dof_index import JointIndex, JointIndexError

    index = JointIndex.build(REQUIRED, required=REQUIRED)
    with pytest.raises(JointIndexError):
        index.reorder(tuple(range(7)))
    with pytest.raises(JointIndexError):
        index.as_mapping(tuple(range(7)))
