# -*- coding: utf-8 -*-
"""PPO policy slot of the planning layer: explicitly NOT_IMPLEMENTED (plan row T7).

Why this slot is empty instead of guessed
-----------------------------------------
1. Scope.  The current milestone is a numpy-only, CPU-only research baseline: no Isaac/Kit, no
   RL training environment and therefore no PPO rollouts.  Any policy shipped from here could
   not be trained, evaluated or reproduced.
2. No specification exists.  The repository defines no observation vector, no action space, no
   reward and no episode/termination specification for a hitting policy.  Inventing those
   numbers would turn guesses into apparent measurements, which the S4/S12 authenticity policy
   forbids (unknown real values stay Param(None, REQUIRES_MEASUREMENT, source=...)).
3. Review order.  A learned whole-body policy must first be validated against the deterministic
   expert planner (ExpertPlanner) and the Safety shield (T8) on the canonical scenario; until
   then a stub that silently returned plausible-looking twists would be the most dangerous
   component in the chain.
4. Deployment.  A PPO checkpoint must arrive with its training config, seed, commit hash and
   sim-to-real validation record; none of those exist yet.

Consequences, made explicit rather than implicit:
  * is_implemented = False, so validate_architecture(..., mode='final') refuses an architecture
    that uses this module as its planner.
  * process raises immediately - it never returns a fabricated WholeBodyTarget.
  * The pipeline can still register it in development mode to exercise the swap-in path
    (replace an implementation; never bypass the layer).
"""
from __future__ import annotations

from ..interfaces import PlanningModule
from ..types import WholeBodyTarget

NOT_IMPLEMENTED_REASON = (
    "PPO policy slot is NOT_IMPLEMENTED and refuses to fabricate a WholeBodyTarget. Reason: "
    "this milestone is numpy-only (no Isaac/Kit), so no PPO training or evaluation can run; the "
    "repository specifies no observation space, action space, reward or termination spec for a "
    "learned hitting policy; and a learned policy must be validated against the deterministic "
    "ExpertPlanner and the Safety shield first. Use ExpertPlanner until a trained checkpoint "
    "with its training config, seed, commit hash and sim-to-real record exists."
)


class PpoPolicyStub(PlanningModule):
    """Occupied slot for the learned planner; deliberately non-functional.

    Construction is allowed so the architecture, the registry and the validation report can
    name the missing component (development mode warns; final mode errors).  Every call raises.
    """

    name = 'ppo_policy'
    is_implemented = False
    NOT_IMPLEMENTED_REASON = NOT_IMPLEMENTED_REASON

    def __init__(self, num_envs=None, **kwargs) -> None:
        self.num_envs = num_envs
        self.kwargs = dict(kwargs)
        self.call_attempts = 0

    def process(self, state, decision, intercept, trajectory) -> WholeBodyTarget:
        self.call_attempts += 1
        raise NotImplementedError(self.NOT_IMPLEMENTED_REASON)


__all__ = ["PpoPolicyStub", "NOT_IMPLEMENTED_REASON"]
