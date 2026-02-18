"""
DEPRECATED: V2 Sign Optimization Module.

This module is a backward-compatibility shim. All functionality has been
moved to ``fcpm.reconstruction``.  Please update your imports::

    # Old (deprecated)
    from v2 import OptimizationResult, layer_then_refine

    # New (recommended)
    from fcpm import OptimizationResult
    from fcpm.reconstruction.optimizers import LayerPropagationOptimizer
"""

import warnings as _warnings

_warnings.warn(
    "Importing from 'v2' is deprecated. "
    "Use 'fcpm.reconstruction.optimizers' or 'fcpm' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical locations
from fcpm.reconstruction.base import OptimizationResult  # noqa: E402, F401
from fcpm.reconstruction.energy import compute_gradient_energy  # noqa: E402, F401

# Keep the original V2 function-level exports working
from .sign_optimization_v2 import (  # noqa: E402, F401
    layer_then_refine,
    layer_by_layer_optimization,
    layer_by_layer_vectorized,
    bidirectional_layer_optimization,
    in_plane_refinement,
    combined_v2_optimization,
)

__all__ = [
    'OptimizationResult',
    'compute_gradient_energy',
    'layer_then_refine',
    'layer_by_layer_optimization',
    'layer_by_layer_vectorized',
    'bidirectional_layer_optimization',
    'in_plane_refinement',
    'combined_v2_optimization',
]
