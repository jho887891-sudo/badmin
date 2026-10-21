# -*- coding: utf-8 -*-
"""RL training-preparation interfaces (P8-01 + P8-02): schemas and offline reward terms.

WHAT THIS PACKAGE IS
--------------------
A *design and test* deliverable: one declarative observation schema with a strict encoder,
one simulation-only action schema, and candidate reward terms that can be scored offline
from recorded frames.  Everything here is CPU-only, numpy-only and simulator-only.

WHAT THIS PACKAGE IS NOT
------------------------
* It is not an environment: there is no step(), no reset(), no rollout and no gym API.
* It is not training: no PPO, no optimiser, no checkpoint, no GPU code, no Isaac import.
* It is not a robot interface: action_schema.SimulatedCommand.to_real_robot() always raises
  RealRobotActionForbidden, and no driver exists in this package.
* It is not project API: docs/observation_action.md and docs/reward_design.md stay
  authoritative and are unchanged.  The candidate spec is tracked by
  management/morph_one_rs/P8_TRAINING_GATE.md.

Training and any physical robot action remain NOT AUTHORISED (see P8-06 / P8-07 in
management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md).
"""
from __future__ import annotations

from .action_schema import (
    ACTION_FIELDS, ACTION_SCHEMA_VERSION, DEFAULT_LIMITS, REAL_ROBOT_FORBIDDEN_REASON,
    SIMULATION_ONLY, TARGET, ActionLimits, ActionSchemaError, ActionValue, SimulatedCommand,
    RealRobotActionForbidden, action_dim, to_command,
)
from .observation_schema import (
    OBSERVATION_DIM, OBSERVATION_FIELDS, OBSERVATION_SCHEMA_VERSION, ObservationSchemaError,
    ObservationValue, build_observation, observation_dim, validate as validate_observation,
)
from .reward_terms import (
    REWARD_SCHEMA_VERSION, REWARD_TERMS, Frame, RewardParams, RewardWeights, TermResult,
    offline_computability_report, total_reward,
)

#: Machine-checkable statement of what this package is allowed to do.
PACKAGE_STATUS = {
    "phase": "P8-01 + P8-02 (preparation only)",
    "training": "NOT AUTHORISED - no training loop exists in this package",
    "hardware": "NO PATH - simulation-only actions; to_real_robot() always raises",
    "gpu": "not used (numpy/CPU only)",
    "isaac": "not imported",
    "project_api": "candidate only; docs/observation_action.md + docs/reward_design.md unchanged",
}

__all__ = [
    "PACKAGE_STATUS",
    "ACTION_FIELDS", "ACTION_SCHEMA_VERSION", "DEFAULT_LIMITS", "REAL_ROBOT_FORBIDDEN_REASON",
    "SIMULATION_ONLY", "TARGET", "ActionLimits", "ActionSchemaError", "ActionValue",
    "SimulatedCommand", "RealRobotActionForbidden", "action_dim", "to_command",
    "OBSERVATION_DIM", "OBSERVATION_FIELDS", "OBSERVATION_SCHEMA_VERSION",
    "ObservationSchemaError", "ObservationValue", "build_observation", "observation_dim",
    "validate_observation",
    "REWARD_SCHEMA_VERSION", "REWARD_TERMS", "Frame", "RewardParams", "RewardWeights",
    "TermResult", "offline_computability_report", "total_reward",
]
