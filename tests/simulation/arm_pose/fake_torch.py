# -*- coding: utf-8 -*-
"""A numpy-backed stand-in for torch, so reset.py can be exercised without Isaac Sim.

WHY: reset_idx() lives in the simulation package and does `import torch` at module scope. torch is
the whole Isaac Lab runtime; a plain CI machine does not have it (this repo's local box does not
either), yet the reset-pose contract is exactly the kind of thing that must be checkable there.
The shim implements only the calls reset_idx() makes - zeros, zeros_like, tensor, cat - plus the
elementwise comparisons the limit guard needs. It is NOT a torch emulator: anything else raises, so
a new torch call inside reset_idx() fails loudly here instead of being silently approximated.
"""
from __future__ import annotations

import types

import numpy as np


class FakeTensor:
    """The smallest tensor reset_idx() needs: 2-D float storage, slicing, comparisons."""

    def __init__(self, array):
        self._a = np.asarray(array, dtype=np.float64)
        self.device = "cpu"

    # -- construction / inspection
    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim

    def numel(self):
        return int(self._a.size)

    def tolist(self):
        return self._a.tolist()

    def clone(self):
        return FakeTensor(self._a.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self._a.astype(np.float32).astype(np.float64))

    def double(self):
        return FakeTensor(self._a.astype(np.float64))

    # -- indexing
    def __getitem__(self, key):
        if isinstance(key, FakeEnvIds):
            key = list(key)
        out = self._a[key]
        return FakeTensor(out) if isinstance(out, np.ndarray) else float(out)

    def __setitem__(self, key, value):
        if isinstance(value, FakeTensor):
            value = value._a
        self._a[key] = value

    def __len__(self):
        return len(self._a)

    # -- shape ops
    def repeat(self, *reps):
        return FakeTensor(np.tile(self._a, reps))

    def reshape(self, *shape):
        return FakeTensor(self._a.reshape(*shape))

    # -- arithmetic / comparison
    def __add__(self, other):
        return FakeTensor(self._a + _arr(other))

    def __sub__(self, other):
        return FakeTensor(self._a - _arr(other))

    def __mul__(self, other):
        return FakeTensor(self._a * _arr(other))

    def __lt__(self, other):
        return FakeTensor(self._a < _arr(other))

    def __le__(self, other):
        return FakeTensor(self._a <= _arr(other))

    def __gt__(self, other):
        return FakeTensor(self._a > _arr(other))

    def __ge__(self, other):
        return FakeTensor(self._a >= _arr(other))

    def __or__(self, other):
        return FakeTensor(np.logical_or(self._a.astype(bool), _arr(other).astype(bool)))

    # reset_idx() rejects NaN with the reflexive test (NaN != NaN), so the shim needs elementwise
    # equality/inequality rather than the default identity comparison.
    def __eq__(self, other):
        return FakeTensor(self._a == _arr(other))

    def __ne__(self, other):
        return FakeTensor(self._a != _arr(other))

    def any(self):
        return FakeTensor(np.asarray(self._a.any()))

    def all(self):
        return FakeTensor(np.asarray(self._a.all()))

    def item(self):
        return self._a.item()

    def __bool__(self):
        return bool(self._a.item()) if self._a.size == 1 else bool(self._a.all())

    def __repr__(self):
        return "FakeTensor(%s)" % (self._a.tolist(),)


def _arr(value):
    return value._a if isinstance(value, FakeTensor) else np.asarray(value, dtype=np.float64)


def make_module():
    """Build a module object that quacks like torch for the handful of calls reset.py makes."""

    mod = types.ModuleType("torch")

    def zeros(shape, device=None, dtype=None):
        return FakeTensor(np.zeros(shape, dtype=np.float64))

    def zeros_like(other, **kwargs):
        return FakeTensor(np.zeros_like(_arr(other)))

    def tensor(data, device=None, dtype=None):
        return FakeTensor(np.asarray(data, dtype=np.float64))

    def cat(tensors, dim=0, **kwargs):
        return FakeTensor(np.concatenate([_arr(t) for t in tensors], axis=dim))

    mod.zeros = zeros
    mod.zeros_like = zeros_like
    mod.tensor = tensor
    mod.cat = cat
    mod.Tensor = FakeTensor
    mod.__doc__ = "numpy-backed torch stand-in for the reset-pose tests"
    return mod


class FakeRobot:
    """Records what reset_idx() writes instead of touching PhysX."""

    def __init__(self, joint_names, num_joints=None):
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names) if num_joints is None else num_joints
        self.joint_state_writes = []

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self.joint_state_writes.append(
            {
                "joint_pos": _as_lists(joint_pos),
                "joint_vel": _as_lists(joint_vel),
                "env_ids": None if env_ids is None else list(env_ids),
            }
        )

    # -- convenience for assertions
    @property
    def last_joint_pos(self):
        assert self.joint_state_writes, "reset_idx() wrote nothing"
        return self.joint_state_writes[-1]["joint_pos"]

    def joint_pos_by_name(self, names=None):
        names = self.joint_names if names is None else names
        row = self.last_joint_pos[0]
        return dict(zip(names, [float(v) for v in row]))


class FakeShuttle:
    def __init__(self):
        self.device = "cpu"
        self.root_pose_writes = []
        self.root_velocity_writes = []

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.root_pose_writes.append(_as_lists(pose))

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.root_velocity_writes.append(_as_lists(vel))


class FakeEnvIds:
    """Stand-in for an Isaac Lab env_ids tensor: only numel() and iteration are used."""

    def __init__(self, ids):
        self._ids = list(ids)

    def numel(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def __getitem__(self, key):
        return self._ids[key]

    def __len__(self):
        return len(self._ids)


def _as_lists(value):
    if isinstance(value, FakeTensor):
        return value.tolist()
    return [[float(v) for v in row] for row in value]
