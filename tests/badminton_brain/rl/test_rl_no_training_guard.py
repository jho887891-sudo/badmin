# -*- coding: utf-8 -*-
"""P8-01/P8-02 guard: "no training was run" -- enforced by static and runtime checks.

The guard is deliberately mechanical: it inspects the RL package's real imports, its
callable namespace, and a fresh interpreter that imports the package.  It must fail if
anyone later drops a trainer, an optimiser or a GPU/RL framework into src/badminton_brain/rl/.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RL_DIR = ROOT / "src" / "badminton_brain" / "rl"
sys.path.insert(0, str(ROOT / "src"))

import badminton_brain.rl as rl  # noqa: E402

FORBIDDEN_MODULES = {
    "torch", "tensorflow", "jax", "gym", "gymnasium", "stable_baselines3", "sb3", "ray",
    "isaacgym", "isaacsim", "omni", "skrl", "tianshou", "accelerate", "transformers",
    "wandb", "optuna", "sklearn",
}
FORBIDDEN_NAME_PREFIXES = ("train", "fit", "optim", "ppo", "learn", "rollout", "replaybuffer")


def _rl_sources():
    files = sorted(RL_DIR.glob("*.py"))
    assert files, "no RL sources found"
    return files


def test_no_rl_framework_import_anywhere_in_the_package():
    found = {}
    for path in _rl_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root in FORBIDDEN_MODULES:
                    found.setdefault(path.name, set()).add(root)
    assert found == {}, "forbidden training/RL framework imports: {0}".format(found)


def test_package_sources_never_call_backward_or_build_an_optimiser():
    offenders = []
    for path in _rl_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("backward", "step", "zero_grad"):
                    offenders.append("{0}:{1} .{2}()".format(
                        path.name, node.lineno, node.func.attr))
    assert offenders == []


def test_public_namespace_contains_no_trainer_or_optimiser():
    # A SUBMODULE is not a trainer.  P8-05 adds the enforcement module
    # badminton_brain.rl.training_gate; importing any submodule sets it as an attribute of the
    # package, so the name scan must look at the API (classes, functions, constants) and not at
    # module objects -- otherwise the guard would forbid the gate module that enforces exactly
    # what the guard is protecting.  The modules themselves are still AST-scanned by the tests
    # above (no torch/gym import, no .backward()/.step()/.zero_grad() call anywhere).
    offenders = [name for name in dir(rl)
                 if not isinstance(getattr(rl, name, None), types.ModuleType)
                 and (name.lower().startswith(FORBIDDEN_NAME_PREFIXES)
                      or name.lower().endswith(("trainer", "optimizer", "policy"))
                      and name.lower() != "ppo_policy")]
    assert offenders == [], offenders
    assert not any(hasattr(rl, name) for name in ("PPO", "Trainer", "Optimizer", "ReplayBuffer"))


def test_fresh_interpreter_imports_only_numpy_and_the_brain_package():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "import badminton_brain.rl as rl\n"
        "forbidden = set({0!r})\n"
        "bad = sorted({{m.split('.')[0] for m in sys.modules}} & forbidden)\n"
        "print('|'.join(bad))\n".format(sorted(FORBIDDEN_MODULES))
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", "forbidden modules imported: " + proc.stdout


def test_package_status_states_that_training_and_hardware_are_not_authorised():
    status = rl.PACKAGE_STATUS
    assert "NOT AUTHORISED" in status["training"].upper()
    assert "NO PATH" in status["hardware"].upper()
    assert "not used" in status["gpu"]
    assert "not imported" in status["isaac"]
    assert status["project_api"].startswith("candidate only")
    assert status["phase"].startswith("P8-01")


def test_no_environment_or_rollout_api_exists():
    for name in ("step", "reset", "rollout", "collect", "play_episode", "Env", "Environment"):
        assert not hasattr(rl, name), name


def test_observation_and_reward_are_importable_without_a_simulator():
    obs = rl.build_observation
    assert callable(obs)
    assert callable(rl.total_reward)
    assert rl.OBSERVATION_DIM == rl.observation_dim() == 70
    assert len(rl.REWARD_TERMS) == 7
    with pytest.raises(rl.ObservationSchemaError):
        obs({})
