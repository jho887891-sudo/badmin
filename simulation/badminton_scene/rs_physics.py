# -*- coding: utf-8 -*-
"""B601-RS simulation physics settings, based on isolated GPU A/B evidence.

Evidence: management/rebot_b601_rs/REVIEW_RUN_55112d2_VELOCITY.md.
In TGS, enable_external_forces_every_iteration=False created spurious gravity-dependent
DOF velocity readings when the joint position was stationary. Switching it on reduced
the 2-second hold integral mismatch (joint2) from 0.134867 to 0.000143 rad at 240Hz.
This fixes a *measured static-state discrepancy*, not proof of dynamic velocity
accuracy or real-hardware capability.

Keep the experimental B601-RS scene and MultiEnvArm on the same PhysX config. Other
robot models and unrelated court-only simulations are not changed by this helper.
"""
from __future__ import annotations


def rebot_rs_simulation_cfg(*, dt: float, device: str = "cuda:0"):
    """Construct the B601-RS PhysX configuration without modifying USD drive parameters."""
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg

    return sim_utils.SimulationCfg(
        dt=float(dt),
        device=device,
        physics=PhysxCfg(
            solver_type=1,  # TGS, same solver as the isolated A/B probe
            enable_external_forces_every_iteration=True,
        ),
    )


__all__ = ["rebot_rs_simulation_cfg"]
