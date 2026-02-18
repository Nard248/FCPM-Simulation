"""
Combined Optimizer (V1 wrapper).

Wraps the original V1 ``combined_optimization`` (chain propagation +
iterative local flip) in a :class:`SignOptimizer` subclass so it can
participate in benchmarking alongside V2 approaches.
"""

from __future__ import annotations

from typing import Optional

from ...core.director import DirectorField
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy
from ..sign_optimization import combined_optimization as _v1_combined


class CombinedOptimizer(SignOptimizer):
    """
    Combined V1 optimizer: chain propagation + iterative local flip.

    This is the original recommended method from V1.  It is wrapped here
    for a uniform interface with V2 optimizers.

    Example::

        optimizer = CombinedOptimizer()
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, max_iter: int = 50):
        self.max_iter = max_iter

    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
    ) -> OptimizationResult:
        initial_energy = compute_gradient_energy(director)

        director_opt, info = _v1_combined(director, verbose=verbose)

        return OptimizationResult(
            director=director_opt,
            initial_energy=initial_energy,
            final_energy=info.get('final_energy', compute_gradient_energy(director_opt)),
            total_flips=0,  # V1 doesn't track this exactly
            method='combined_v1',
            metadata={
                'converged': info.get('converged', False),
                'iterations': info.get('iterations', 0),
            },
        )


def combined_v1_optimization(
    director: DirectorField,
    verbose: bool = False,
) -> OptimizationResult:
    """Functional interface for V1 combined optimization."""
    optimizer = CombinedOptimizer()
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "CombinedOptimizer",
    "combined_v1_optimization",
]
