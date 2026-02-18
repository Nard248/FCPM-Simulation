"""
DEPRECATED: Advanced Sign Optimization Approaches.

This module is a backward-compatibility shim.  All optimizers have been
moved to ``fcpm.reconstruction.optimizers``.  Please update your imports::

    # Old (deprecated)
    from v2.approaches import GraphCutsOptimizer

    # New (recommended)
    from fcpm.reconstruction.optimizers import GraphCutsOptimizer
    # or simply:
    from fcpm import GraphCutsOptimizer
"""

import warnings as _warnings

_warnings.warn(
    "Importing from 'v2.approaches' is deprecated. "
    "Use 'fcpm.reconstruction.optimizers' or 'fcpm' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from fcpm.reconstruction.optimizers import (  # noqa: E402, F401
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer,
    graph_cuts_optimization,
    simulated_annealing_optimization,
    hierarchical_optimization,
    belief_propagation_optimization,
)
from fcpm.reconstruction.base import OptimizationResult  # noqa: E402, F401

__all__ = [
    'GraphCutsOptimizer',
    'SimulatedAnnealingOptimizer',
    'HierarchicalOptimizer',
    'BeliefPropagationOptimizer',
    'graph_cuts_optimization',
    'simulated_annealing_optimization',
    'hierarchical_optimization',
    'belief_propagation_optimization',
    'OptimizationResult',
]
