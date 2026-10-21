# -*- coding: utf-8 -*-
"""Harness-only simulation rigs (no physics, no camera, no robot)."""
from .closed_loop import (SCENARIOS, HarnessConfig, HarnessRefusal, RacketCollisionProxy,
                          run_closed_loop)

__all__ = ["SCENARIOS", "HarnessConfig", "HarnessRefusal", "RacketCollisionProxy",
           "run_closed_loop"]
