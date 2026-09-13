# -*- coding: utf-8 -*-
"""Module registry: the switchboard that makes the architecture modular (S11/S15)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .interfaces import BrainModule, interface_layer_of
from .types import BASELINE_ORDER, BrainBoundaryError, Layer


class ModuleRegistry:
    """One module per layer; duplicates and layer mismatches are rejected loudly."""

    def __init__(self) -> None:
        self._modules: Dict[Layer, BrainModule] = {}

    def register(self, module: BrainModule) -> None:
        layer = interface_layer_of(module)
        declared = getattr(module, 'layer', None)
        if declared is not layer:
            raise BrainBoundaryError(
                f"{type(module).__name__}.layer={declared} contradicts its interface layer {layer.value}")
        if layer in self._modules:
            raise BrainBoundaryError(
                f"layer '{layer.value}' already has {type(self._modules[layer]).__name__}; "
                "use replace() to swap an implementation")
        self._modules[layer] = module

    def replace(self, module: BrainModule) -> None:
        """Swap the implementation of an already-registered layer."""
        layer = interface_layer_of(module)
        if getattr(module, 'layer', None) is not layer:
            raise BrainBoundaryError(
                f"{type(module).__name__}.layer contradicts its interface layer {layer.value}")
        if layer not in self._modules:
            raise BrainBoundaryError(f"layer '{layer.value}' is not registered yet")
        self._modules[layer] = module

    def get(self, layer: Layer) -> Optional[BrainModule]:
        return self._modules.get(layer)

    def layers(self) -> Tuple[Layer, ...]:
        ordered: List[Layer] = [l for l in BASELINE_ORDER if l in self._modules]
        if Layer.ADAPTATION in self._modules:
            ordered.append(Layer.ADAPTATION)
        return tuple(ordered)

    def modules(self) -> Tuple[BrainModule, ...]:
        return tuple(self._modules[l] for l in self.layers())

    def missing(self, required=BASELINE_ORDER) -> Tuple[Layer, ...]:
        return tuple(l for l in required if l not in self._modules)


__all__ = ["ModuleRegistry"]
