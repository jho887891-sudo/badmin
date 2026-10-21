# -*- coding: utf-8 -*-
"""Fixtures for the model-aware reset-pose tests (P5B).

Everything here is pure python: no Isaac Sim, no torch, no pxr. Two stand-ins make that possible:

  * The real B601-RS limits come from the committed pxr export
    outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json. The 20 MB USD is a
    host resource (AGENTS.md: resources stay on the host, results are committed) and pxr is not
    installed on a plain machine, so the tests read exactly what pxr recorded - and say so.
  * reset.py is imported with a numpy-backed torch stand-in (fake_torch.py), so reset_idx() runs
    without the Isaac Lab runtime while still going through its real code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pose_data  # noqa: E402  (needs HERE on sys.path first)

for _candidate in (str(pose_data.SCENE_DIR), str(HERE)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


@pytest.fixture(scope="session")
def usd_facts():
    return pose_data.read_facts()


@pytest.fixture(scope="session")
def usd_limits():
    return pose_data.read_usd_limits()


@pytest.fixture(scope="session")
def rest_pose():
    """The production module under test (no torch, no pxr)."""
    import badminton_scene.rest_pose as module

    return module


@pytest.fixture
def reset_mod(monkeypatch):
    """reset.py, imported against the numpy torch stand-in.

    Loaded under a private name so it is not cached in sys.modules: every test executes a fresh
    module, and the fake torch disappears with monkeypatch teardown.
    """
    import importlib.util

    import fake_torch

    monkeypatch.setitem(sys.modules, "torch", fake_torch.make_module())
    spec = importlib.util.spec_from_file_location("badminton_scene._reset_under_test", pose_data.RESET_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture
def rs_robot():
    import fake_torch

    return fake_torch.FakeRobot(pose_data.REBOT_RS_JOINT_NAMES)


@pytest.fixture
def piper_robot():
    import fake_torch

    return fake_torch.FakeRobot(pose_data.PIPER_JOINT_NAMES)


@pytest.fixture
def shuttle():
    import fake_torch

    return fake_torch.FakeShuttle()


@pytest.fixture
def env_ids():
    import fake_torch

    return fake_torch.FakeEnvIds([0])
