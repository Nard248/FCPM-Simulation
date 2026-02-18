"""
Layer Propagation Optimizer for Sign Consistency.

Physics-informed layer-by-layer approach: process the volume slice by
slice along z, choosing each voxel's sign to minimise local gradient
energy with respect to already-determined neighbours.

This is the V2 "recommended" approach, combining:
1. Layer-by-layer propagation (fast, establishes z-continuity)
2. Iterative local refinement (catches in-plane inconsistencies)

Extracted from ``v2/sign_optimization_v2.py`` and wrapped in a
:class:`SignOptimizer` subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.director import DirectorField, DTYPE
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy


@dataclass
class LayerPropagationConfig:
    """Configuration for layer propagation optimizer."""

    max_refine_iter: int = 20
    bidirectional: bool = False


class LayerPropagationOptimizer(SignOptimizer):
    """
    Layer-by-layer propagation followed by iterative refinement.

    Phase 1 establishes z-continuity by aligning each layer with the
    previous one (fast, O(N)).  Phase 2 performs global iterative
    flipping to smooth remaining in-plane inconsistencies.

    Example::

        optimizer = LayerPropagationOptimizer()
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[LayerPropagationConfig] = None):
        self.config = config or LayerPropagationConfig()

    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
    ) -> OptimizationResult:
        n = director.to_array().astype(DTYPE).copy()
        ny_dim, nx_dim, nz_dim = n.shape[:3]

        initial_energy = compute_gradient_energy(n)
        total_flips = 0

        if verbose:
            print("=" * 60)
            print("Layer-Then-Refine Optimization")
            print("=" * 60)
            print(f"Initial energy: {initial_energy:.2f}")

        # Phase 1: Layer-by-layer propagation (z-1 reference only)
        if verbose:
            print("\nPhase 1: Layer propagation (z-1 reference)...")

        for z in range(1, nz_dim):
            dot = np.sum(n[:, :, z] * n[:, :, z - 1], axis=-1)
            flip_mask = dot < 0
            n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
            total_flips += int(np.sum(flip_mask))

        if self.config.bidirectional:
            for z in range(nz_dim - 2, -1, -1):
                dot = np.sum(n[:, :, z] * n[:, :, z + 1], axis=-1)
                flip_mask = dot < 0
                n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
                total_flips += int(np.sum(flip_mask))

        energy_after_layer = compute_gradient_energy(n)
        if verbose:
            print(f"  After layer propagation: energy = {energy_after_layer:.2f}")

        # Phase 2: Iterative refinement
        if verbose:
            print("\nPhase 2: Iterative refinement...")

        for iteration in range(self.config.max_refine_iter):
            cost_curr = np.zeros((ny_dim, nx_dim, nz_dim), dtype=DTYPE)
            cost_flip = np.zeros((ny_dim, nx_dim, nz_dim), dtype=DTYPE)

            for axis in range(3):
                n_fwd = np.roll(n, -1, axis=axis)
                cost_curr += np.sum((n - n_fwd) ** 2, axis=-1)
                cost_flip += np.sum((n + n_fwd) ** 2, axis=-1)

                n_bwd = np.roll(n, 1, axis=axis)
                cost_curr += np.sum((n - n_bwd) ** 2, axis=-1)
                cost_flip += np.sum((n + n_bwd) ** 2, axis=-1)

            flip_mask = cost_flip < cost_curr
            n_flipped = int(np.sum(flip_mask))

            if verbose and (iteration < 5 or iteration % 10 == 0):
                energy = compute_gradient_energy(n)
                print(f"  Iter {iteration}: flipped {n_flipped} voxels, energy = {energy:.2f}")

            if n_flipped == 0:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break

            n[flip_mask] = -n[flip_mask]
            total_flips += n_flipped

        final_energy = compute_gradient_energy(n)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Total reduction: {initial_energy - final_energy:.2f} "
                  f"({100 * (initial_energy - final_energy) / initial_energy:.1f}%)")

        return OptimizationResult(
            director=DirectorField.from_array(
                n, metadata={'sign_method': 'layer_then_refine'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_flips=total_flips,
            method='layer_then_refine',
        )


def layer_propagation_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[LayerPropagationConfig] = None,
) -> OptimizationResult:
    """Functional interface for layer propagation optimization."""
    optimizer = LayerPropagationOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "LayerPropagationConfig",
    "LayerPropagationOptimizer",
    "layer_propagation_optimization",
]
