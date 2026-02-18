"""
Sign Optimization Algorithms.

All optimizers inherit from :class:`~fcpm.reconstruction.base.SignOptimizer`
and return :class:`~fcpm.reconstruction.base.OptimizationResult`.

Available optimizers:

* :class:`CombinedOptimizer` — V1 chain propagation + iterative flip
* :class:`LayerPropagationOptimizer` — layer-by-layer + refinement
* :class:`GraphCutsOptimizer` — exact min-cut / max-flow
* :class:`SimulatedAnnealingOptimizer` — Metropolis-Hastings
* :class:`HierarchicalOptimizer` — coarse-to-fine multi-scale
* :class:`BeliefPropagationOptimizer` — loopy BP (experimental)
"""

from .combined import CombinedOptimizer, combined_v1_optimization
from .layer_propagation import (
    LayerPropagationConfig,
    LayerPropagationOptimizer,
    layer_propagation_optimization,
)
from .graph_cuts import (
    GraphCutsConfig,
    GraphCutsOptimizer,
    graph_cuts_optimization,
)
from .simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingOptimizer,
    SAHistory,
    simulated_annealing_optimization,
)
from .hierarchical import (
    HierarchicalConfig,
    HierarchicalOptimizer,
    hierarchical_optimization,
)
from .belief_propagation import (
    BeliefPropagationConfig,
    BeliefPropagationOptimizer,
    belief_propagation_optimization,
)

__all__ = [
    # Classes (recommended interface)
    "CombinedOptimizer",
    "LayerPropagationOptimizer",
    "GraphCutsOptimizer",
    "SimulatedAnnealingOptimizer",
    "HierarchicalOptimizer",
    "BeliefPropagationOptimizer",
    # Config dataclasses
    "LayerPropagationConfig",
    "GraphCutsConfig",
    "SimulatedAnnealingConfig",
    "HierarchicalConfig",
    "BeliefPropagationConfig",
    "SAHistory",
    # Functional interfaces
    "combined_v1_optimization",
    "layer_propagation_optimization",
    "graph_cuts_optimization",
    "simulated_annealing_optimization",
    "hierarchical_optimization",
    "belief_propagation_optimization",
]
