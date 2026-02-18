"""
Hierarchical Coarse-to-Fine Optimization for Sign Consistency.

Multi-scale approach that captures global structure at coarse levels
and refines locally at fine levels.

Information flow:
    Level 0 (coarse) -> Level 1 -> ... -> Level L (original resolution)

Coarsening strategy:
- 2x2x2 blocks -> single representative director
- Representative = dominant eigenvector of averaged Q-tensor
- Preserves orientation information, smooths noise

Complexity: O(N + N/8 + N/64 + ...) = O(N)

References:
-----------
[1] Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief
    propagation for early vision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ...core.director import DirectorField, DTYPE
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical optimization."""

    coarsen_factor: int = 2
    min_dimension: int = 4
    refine_iterations: int = 5
    use_qtensor_coarsening: bool = True
    use_graph_cuts_at_coarse: bool = False


class HierarchicalOptimizer(SignOptimizer):
    """
    Hierarchical coarse-to-fine optimizer.

    Good balance between speed and quality — captures global structure
    through coarsening and refines local details at full resolution.

    Example::

        optimizer = HierarchicalOptimizer()
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        self.config = config or HierarchicalConfig()

    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
    ) -> OptimizationResult:
        n = director.to_array().astype(DTYPE).copy()
        ny, nx, nz = n.shape[:3]

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Hierarchical Coarse-to-Fine Sign Optimization")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")

        pyramid = self._build_pyramid(n, verbose)
        num_levels = len(pyramid)

        if verbose:
            print(f"\nPyramid levels: {num_levels}")
            for i, level in enumerate(pyramid):
                print(f"  Level {i}: shape {level.shape[:3]}")

        # Optimize coarsest level
        if verbose:
            print(f"\nOptimizing coarsest level (Level 0)...")

        n_coarse = pyramid[0]

        if self.config.use_graph_cuts_at_coarse:
            n_coarse = self._optimize_with_graph_cuts(n_coarse)
        else:
            n_coarse = self._optimize_layer_propagation(n_coarse)
            n_coarse = self._iterative_refinement(n_coarse, max_iter=20)

        pyramid[0] = n_coarse

        if verbose:
            print(f"  Coarse level energy: {compute_gradient_energy(n_coarse):.2f}")

        # Refine up the pyramid
        if verbose:
            print(f"\nRefining through pyramid levels...")

        for level in range(1, num_levels):
            if verbose:
                print(f"\n  Level {level}:")

            n_fine = pyramid[level]
            n_coarse = pyramid[level - 1]

            n_fine = self._upsample_signs(n_coarse, n_fine, verbose)
            n_fine = self._iterative_refinement(
                n_fine, max_iter=self.config.refine_iterations)
            pyramid[level] = n_fine

            if verbose:
                print(f"    After refinement: energy = {compute_gradient_energy(n_fine):.2f}")

        n_optimized = pyramid[-1]
        final_energy = compute_gradient_energy(n_optimized)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")

        original = director.to_array()
        dot = np.sum(original * n_optimized, axis=-1)
        total_flips = int(np.sum(dot < 0))

        return OptimizationResult(
            director=DirectorField.from_array(
                n_optimized, metadata={'sign_method': 'hierarchical'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[compute_gradient_energy(p) for p in pyramid],
            total_flips=total_flips,
            method='hierarchical',
        )

    def _build_pyramid(self, n: np.ndarray, verbose: bool = False) -> List[np.ndarray]:
        pyramid = [n]
        current = n
        cf = self.config.coarsen_factor

        while min(current.shape[:3]) > self.config.min_dimension * cf:
            coarsened = self._coarsen(current)
            pyramid.insert(0, coarsened)
            current = coarsened

        return pyramid

    def _coarsen(self, n: np.ndarray) -> np.ndarray:
        cf = self.config.coarsen_factor
        ny, nx, nz = n.shape[:3]

        ny_c = ny // cf
        nx_c = nx // cf
        nz_c = nz // cf

        n_trimmed = n[:ny_c * cf, :nx_c * cf, :nz_c * cf]

        if self.config.use_qtensor_coarsening:
            n_coarse = np.zeros((ny_c, nx_c, nz_c, 3), dtype=DTYPE)

            for y in range(ny_c):
                for x in range(nx_c):
                    for z in range(nz_c):
                        block = n_trimmed[
                            y * cf:(y + 1) * cf,
                            x * cf:(x + 1) * cf,
                            z * cf:(z + 1) * cf
                        ]

                        Q_sum = np.zeros((3, 3), dtype=DTYPE)
                        for n_vec in block.reshape(-1, 3):
                            Q_sum += np.outer(n_vec, n_vec)
                        Q_avg = Q_sum / block.size

                        eigenvalues, eigenvectors = np.linalg.eigh(Q_avg)
                        n_rep = eigenvectors[:, -1]

                        norm = np.linalg.norm(n_rep)
                        if norm > 1e-6:
                            n_rep = n_rep / norm

                        n_coarse[y, x, z] = n_rep
        else:
            n_coarse = np.zeros((ny_c, nx_c, nz_c, 3), dtype=DTYPE)

            for y in range(ny_c):
                for x in range(nx_c):
                    for z in range(nz_c):
                        block = n_trimmed[
                            y * cf:(y + 1) * cf,
                            x * cf:(x + 1) * cf,
                            z * cf:(z + 1) * cf
                        ]

                        n_first = block[0, 0, 0]
                        block_aligned = block.copy()
                        for i in range(cf):
                            for j in range(cf):
                                for k in range(cf):
                                    if np.dot(block[i, j, k], n_first) < 0:
                                        block_aligned[i, j, k] = -block[i, j, k]

                        n_avg = np.mean(block_aligned.reshape(-1, 3), axis=0)
                        norm = np.linalg.norm(n_avg)
                        if norm > 1e-6:
                            n_avg = n_avg / norm

                        n_coarse[y, x, z] = n_avg

        return n_coarse

    def _upsample_signs(self, n_coarse: np.ndarray, n_fine: np.ndarray,
                        verbose: bool = False) -> np.ndarray:
        cf = self.config.coarsen_factor
        ny_f, nx_f, nz_f = n_fine.shape[:3]
        ny_c, nx_c, nz_c = n_coarse.shape[:3]

        n_result = n_fine.copy()
        flips = 0

        for y in range(ny_f):
            for x in range(nx_f):
                for z in range(nz_f):
                    y_c = min(y // cf, ny_c - 1)
                    x_c = min(x // cf, nx_c - 1)
                    z_c = min(z // cf, nz_c - 1)

                    if np.dot(n_fine[y, x, z], n_coarse[y_c, x_c, z_c]) < 0:
                        n_result[y, x, z] = -n_fine[y, x, z]
                        flips += 1

        if verbose:
            print(f"    Upsampled: {flips} flips")

        return n_result

    @staticmethod
    def _optimize_layer_propagation(n: np.ndarray) -> np.ndarray:
        nz = n.shape[2]
        for z in range(1, nz):
            dot = np.sum(n[:, :, z] * n[:, :, z - 1], axis=-1)
            flip_mask = dot < 0
            n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
        return n

    @staticmethod
    def _iterative_refinement(n: np.ndarray, max_iter: int = 10) -> np.ndarray:
        ny, nx, nz = n.shape[:3]

        for _ in range(max_iter):
            cost_curr = np.zeros((ny, nx, nz), dtype=DTYPE)
            cost_flip = np.zeros((ny, nx, nz), dtype=DTYPE)

            for axis in range(3):
                n_fwd = np.roll(n, -1, axis=axis)
                cost_curr += np.sum((n - n_fwd) ** 2, axis=-1)
                cost_flip += np.sum((n + n_fwd) ** 2, axis=-1)

                n_bwd = np.roll(n, 1, axis=axis)
                cost_curr += np.sum((n - n_bwd) ** 2, axis=-1)
                cost_flip += np.sum((n + n_bwd) ** 2, axis=-1)

            flip_mask = cost_flip < cost_curr
            if np.sum(flip_mask) == 0:
                break

            n[flip_mask] = -n[flip_mask]

        return n

    def _optimize_with_graph_cuts(self, n: np.ndarray) -> np.ndarray:
        try:
            from .graph_cuts import GraphCutsOptimizer
            director_tmp = DirectorField.from_array(n)
            optimizer = GraphCutsOptimizer()
            result = optimizer.optimize(director_tmp, verbose=False)
            return result.director.to_array()
        except (ImportError, Exception):
            return self._optimize_layer_propagation(n)


def hierarchical_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[HierarchicalConfig] = None,
) -> OptimizationResult:
    """Functional interface for hierarchical optimization."""
    optimizer = HierarchicalOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "HierarchicalConfig",
    "HierarchicalOptimizer",
    "hierarchical_optimization",
]
