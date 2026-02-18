"""
FCPM Reconstruction Module

Methods for reconstructing the director field from FCPM intensity measurements.

Approaches:
1. Direct inversion - algebraic inversion of FCPM formula
2. Q-tensor method - sign-invariant reconstruction via Q-tensor
3. Sign optimization - enforce spatial consistency of director signs

Recommended pipeline:
    1. Use reconstruct_via_qtensor() for initial reconstruction
    2. Apply combined_optimization() for sign consistency
    3. Verify with compute_fcpm_intensity_error()

V2 optimizer classes (all inherit from SignOptimizer):
    CombinedOptimizer, LayerPropagationOptimizer, GraphCutsOptimizer,
    SimulatedAnnealingOptimizer, HierarchicalOptimizer,
    BeliefPropagationOptimizer
"""

from .direct import (
    extract_magnitudes,
    extract_cross_term,
    reconstruct_director_direct,
    reconstruct_director_all_angles,
)

from .qtensor_method import (
    qtensor_from_fcpm,
    qtensor_from_fcpm_exact,
    reconstruct_via_qtensor,
    compute_qtensor_error,
)

# V1 sign optimization functions (backward compatible)
from .sign_optimization import (
    chain_propagation,
    iterative_local_flip,
    wavefront_propagation,
    multi_axis_propagation,
    combined_optimization,
    gradient_energy,
)

# V2 base abstractions
from .base import SignOptimizer, OptimizationResult
from .energy import compute_gradient_energy, FrankConstants, compute_frank_energy_anisotropic

# V2 optimizer classes
from .optimizers import (
    CombinedOptimizer,
    LayerPropagationOptimizer,
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer,
    # Configs
    LayerPropagationConfig,
    GraphCutsConfig,
    SimulatedAnnealingConfig,
    HierarchicalConfig,
    BeliefPropagationConfig,
    # Functional interfaces
    graph_cuts_optimization,
    simulated_annealing_optimization,
    hierarchical_optimization,
    belief_propagation_optimization,
    layer_propagation_optimization,
    combined_v1_optimization,
)

__all__ = [
    # Direct methods
    'extract_magnitudes',
    'extract_cross_term',
    'reconstruct_director_direct',
    'reconstruct_director_all_angles',
    # Q-tensor methods
    'qtensor_from_fcpm',
    'qtensor_from_fcpm_exact',
    'reconstruct_via_qtensor',
    'compute_qtensor_error',
    # V1 sign optimization (backward compatible)
    'chain_propagation',
    'iterative_local_flip',
    'wavefront_propagation',
    'multi_axis_propagation',
    'combined_optimization',
    'gradient_energy',
    # V2 base abstractions
    'SignOptimizer',
    'OptimizationResult',
    'compute_gradient_energy',
    'FrankConstants',
    'compute_frank_energy_anisotropic',
    # V2 optimizer classes
    'CombinedOptimizer',
    'LayerPropagationOptimizer',
    'GraphCutsOptimizer',
    'SimulatedAnnealingOptimizer',
    'HierarchicalOptimizer',
    'BeliefPropagationOptimizer',
    # Configs
    'LayerPropagationConfig',
    'GraphCutsConfig',
    'SimulatedAnnealingConfig',
    'HierarchicalConfig',
    'BeliefPropagationConfig',
    # Functional interfaces
    'graph_cuts_optimization',
    'simulated_annealing_optimization',
    'hierarchical_optimization',
    'belief_propagation_optimization',
    'layer_propagation_optimization',
    'combined_v1_optimization',
]
