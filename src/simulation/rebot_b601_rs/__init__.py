# -*- coding: utf-8 -*-
"""Formal arm package: reBot B601-RS (migrated from PiPER 2026-10).

This package is the B601-RS identity. It must never import from the PiPER stage-0 modules nor from
the B601-DM candidate line: parameters are not transferable between the three arms.
"""
from . import joint_limits  # noqa: F401

__all__ = ["joint_limits"]
