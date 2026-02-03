"""
Advanced Sign Optimization Approaches for FCPM Director Reconstruction.

This module implements multiple optimization strategies for resolving the
nematic sign ambiguity (n ≡ -n) after Q-tensor eigendecomposition.

Available Approaches:
--------------------
1. Graph Cuts (graph_cuts.py)
   - Min-cut/max-flow for globally optimal solution
   - Exact solution for submodular energy functions
   - Best for smooth director fields

2. Simulated Annealing (simulated_annealing.py)
   - Metropolis-Hastings with adaptive temperature
   - Can escape local minima
   - Includes cluster (Wolff) moves option

3. Hierarchical (hierarchical.py)
   - Multi-scale coarse-to-fine optimization
   - Fast with global consistency
   - Good balance of speed and quality

4. Belief Propagation (belief_propagation.py)
   - Message passing on factor graph
   - Parallelizable, approximate solution
   - Works well on loopy graphs with damping

Mathematical Foundation:
-----------------------
The sign optimization problem minimizes:

    E(S) = Σ_{(i,j)∈neighbors} |s_i·n̂_i - s_j·n̂_j|²

where s_i ∈ {+1, -1} and n̂_i is the unsigned director at voxel i.

This is equivalent to an Ising model:

    E(S) = Σ_{(i,j)} J_ij · s_i · s_j + const

where J_ij = -2(n̂_i · n̂_j) is the coupling strength.

Usage:
------
>>> from v2.approaches import GraphCutsOptimizer, SimulatedAnnealingOptimizer
>>> from v2.approaches import HierarchicalOptimizer, BeliefPropagationOptimizer
>>>
>>> # Any optimizer follows the same interface:
>>> optimizer = GraphCutsOptimizer()
>>> result = optimizer.optimize(director_field, verbose=True)
>>> optimized_director = result.director

Author: FCPM Simulation Project
"""

from .graph_cuts import GraphCutsOptimizer, graph_cuts_optimization
from .simulated_annealing import (
    SimulatedAnnealingOptimizer,
    simulated_annealing_optimization,
)
from .hierarchical import HierarchicalOptimizer, hierarchical_optimization
from .belief_propagation import (
    BeliefPropagationOptimizer,
    belief_propagation_optimization,
)

# Common result type - use try/except for flexible imports
try:
    from v2.sign_optimization_v2 import OptimizationResult
except ImportError:
    from sign_optimization_v2 import OptimizationResult

__all__ = [
    # Optimizers (class-based interface)
    'GraphCutsOptimizer',
    'SimulatedAnnealingOptimizer',
    'HierarchicalOptimizer',
    'BeliefPropagationOptimizer',
    # Functional interface
    'graph_cuts_optimization',
    'simulated_annealing_optimization',
    'hierarchical_optimization',
    'belief_propagation_optimization',
    # Common types
    'OptimizationResult',
]
