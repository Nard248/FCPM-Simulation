"""
V2 Sign Optimization Module.

This module contains advanced sign optimization approaches for FCPM director reconstruction.

Main exports:
- OptimizationResult: Result dataclass from sign optimization
- layer_then_refine: V2 recommended optimization method
- compute_gradient_energy: Energy computation utility

Submodules:
- approaches: Advanced optimization algorithms (graph cuts, SA, hierarchical, BP)
"""

from .sign_optimization_v2 import (
    OptimizationResult,
    compute_gradient_energy,
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
