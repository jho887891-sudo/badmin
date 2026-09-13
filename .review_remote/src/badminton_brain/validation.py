# -*- coding: utf-8 -*-
"""Architecture-level validation (ROBOT_BRAIN.md S11/S12/S15 + final-mode discipline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .pipeline import PipelineConfig
from .registry import ModuleRegistry
from .status import UNRESOLVED_STATUSES as UNRESOLVED


@dataclass
class ArchitectureReport:
    ok: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_architecture(registry: ModuleRegistry, config: Optional[PipelineConfig] = None,
                          *, mode: str = 'development') -> ArchitectureReport:
    """Check stage completeness, implementation status and rate resolution."""
    if mode not in ('development', 'final'):
        raise ValueError("mode must be 'development' or 'final'")
    config = config or PipelineConfig()
    errors: List[str] = []
    warnings: List[str] = []

    for layer in registry.missing():
        errors.append(f"baseline stage '{layer.value}' is missing from the registry")

    for module in registry.modules():
        if not getattr(module, 'is_implemented', False):
            message = f"layer '{module.layer.value}' module '{module.name}' is not implemented"
            if mode == 'final':
                errors.append(message + " (final mode requires real algorithms)")
            else:
                warnings.append(message)

    for layer_name, param in config.stage_rate_hz.items():
        if param.status in UNRESOLVED:
            message = f"stage rate for '{layer_name}' is {param.status.value}"
            if mode == 'final':
                errors.append('final mode requires resolved stage rates: ' + message)
            else:
                warnings.append(message)

    return ArchitectureReport(ok=not errors, errors=errors, warnings=warnings)


__all__ = ["ArchitectureReport", "validate_architecture"]
