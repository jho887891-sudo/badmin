"""Convert Isaac Lab tensors to finite host-side numeric arrays for logging.

Torch tensors may live on cuda:0: numpy.asarray(cuda_tensor) raises TypeError.
This small bridge is stdlib-only and independent of Isaac, NumPy and Torch.
"""
from __future__ import annotations

import math


def host_floats(value, count: int) -> list[float]:
    """Read all requested components without passing a CUDA tensor to NumPy.

    The copy is intentional: diagnostics serialize CPU-native floats, never a
    borrowed device buffer whose contents can change before JSON write.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"unsupported vector data type: {type(value).__name__}")
    if len(value) < count:
        raise ValueError(f"expected at least {count} components; got {len(value)}")
    out = [float(item) for item in value[:count]]
    if not all(math.isfinite(x) for x in out):
        raise ValueError("nonfinite telemetry value")
    return out
