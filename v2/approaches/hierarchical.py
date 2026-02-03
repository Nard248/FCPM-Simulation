"""
Hierarchical Coarse-to-Fine Optimization for Sign Consistency.

Multi-scale approach that captures global structure at coarse levels
and refines locally at fine levels.

Mathematical Basis:
------------------
Multi-scale energy minimization:

    E = Σ_l E^(l)  over pyramid levels l = 0 (coarsest) to L (finest)

At each level l, the optimization problem is:
    min E^(l)(S^(l)) subject to consistency with coarser level

Information flow:
    Level 0 (coarse) → Level 1 → ... → Level L (original resolution)

Coarsening strategy:
- 2×2×2 blocks → single representative director
- Representative = dominant eigenvector of averaged Q-tensor
- Preserves orientation information, smooths noise

Upsampling strategy:
- Interpolate signs from coarse to fine level
- Fine-tune with local optimization

Complexity: O(N + N/8 + N/64 + ...) = O(N)

Advantages:
- Global consistency from coarse level
- Fast (linear complexity)
- Robust to local noise

Disadvantages:
- May miss small-scale features like defects
- Coarsening assumes local smoothness

References:
-----------
[1] Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief
    propagation for early vision.
[2] Komodakis, N., & Tziritas, G. (2007). Approximate labeling via
    graph cuts based on linear programming.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fcpm.core.director import DirectorField, DTYPE

# Flexible imports for both package and standalone use
try:
    from v2.sign_optimization_v2 import OptimizationResult, compute_gradient_energy
except ImportError:
    from sign_optimization_v2 import OptimizationResult, compute_gradient_energy


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical optimization."""

    # Coarsening factor (2 = 2×2×2 blocks)
    coarsen_factor: int = 2

    # Minimum dimension at coarsest level
    min_dimension: int = 4

    # Refinement iterations at each level
    refine_iterations: int = 5

    # Use Q-tensor averaging for coarsening (vs simple director averaging)
    use_qtensor_coarsening: bool = True

    # Apply graph cuts at coarsest level (if available)
    use_graph_cuts_at_coarse: bool = False


class HierarchicalOptimizer:
    """
    Hierarchical coarse-to-fine optimizer.

    This optimizer provides a good balance between speed and quality:
    - Captures global structure through coarsening
    - Refines local details at full resolution
    - Linear time complexity

    Example:
    -------
    >>> optimizer = HierarchicalOptimizer()
    >>> result = optimizer.optimize(director, verbose=True)
    >>> print(f"Energy: {result.final_energy:.1f}")
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or HierarchicalConfig()

    def optimize(self, director: DirectorField,
                 verbose: bool = False) -> OptimizationResult:
        """
        Optimize director signs using hierarchical coarse-to-fine.

        Args:
            director: Input director field with arbitrary signs.
            verbose: Print progress information.

        Returns:
            OptimizationResult with optimized director and statistics.
        """
        n = director.to_array().astype(DTYPE).copy()
        ny, nx, nz = n.shape[:3]

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Hierarchical Coarse-to-Fine Sign Optimization")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print(f"Coarsen factor: {self.config.coarsen_factor}")
            print(f"Min dimension: {self.config.min_dimension}")

        # Build pyramid
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
            n_coarse = self._optimize_with_graph_cuts(n_coarse, verbose)
        else:
            n_coarse = self._optimize_layer_propagation(n_coarse)
            n_coarse = self._iterative_refinement(n_coarse, max_iter=20)

        pyramid[0] = n_coarse

        if verbose:
            energy_0 = compute_gradient_energy(n_coarse)
            print(f"  Coarse level energy: {energy_0:.2f}")

        # Refine up the pyramid
        if verbose:
            print(f"\nRefining through pyramid levels...")

        for level in range(1, num_levels):
            if verbose:
                print(f"\n  Level {level}:")

            # Upsample signs from coarser level
            n_fine = pyramid[level]
            n_coarse = pyramid[level - 1]

            n_fine = self._upsample_signs(n_coarse, n_fine, verbose)

            # Local refinement
            n_fine = self._iterative_refinement(
                n_fine,
                max_iter=self.config.refine_iterations
            )

            pyramid[level] = n_fine

            if verbose:
                energy_l = compute_gradient_energy(n_fine)
                print(f"    After refinement: energy = {energy_l:.2f}")

        # Final result is the finest level
        n_optimized = pyramid[-1]
        final_energy = compute_gradient_energy(n_optimized)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")

        # Count total flips (approximate)
        original = director.to_array()
        dot = np.sum(original * n_optimized, axis=-1)
        total_flips = np.sum(dot < 0)

        return OptimizationResult(
            director=DirectorField.from_array(n_optimized, metadata={'sign_method': 'hierarchical'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[compute_gradient_energy(p) for p in pyramid],
            flips_by_layer=[],
            total_flips=int(total_flips),
            method='hierarchical'
        )

    def _build_pyramid(self, n: np.ndarray, verbose: bool = False) -> List[np.ndarray]:
        """
        Build multi-scale pyramid by successive coarsening.

        Returns list from coarsest to finest.
        """
        pyramid = [n]

        current = n
        cf = self.config.coarsen_factor

        while min(current.shape[:3]) > self.config.min_dimension * cf:
            coarsened = self._coarsen(current)
            pyramid.insert(0, coarsened)  # Insert at beginning (coarsest first)
            current = coarsened

        return pyramid

    def _coarsen(self, n: np.ndarray) -> np.ndarray:
        """
        Coarsen director field by factor of 2 in each dimension.

        Uses Q-tensor averaging for proper orientation handling.
        """
        cf = self.config.coarsen_factor
        ny, nx, nz = n.shape[:3]

        # New dimensions (integer division)
        ny_c = ny // cf
        nx_c = nx // cf
        nz_c = nz // cf

        # Trim to exact multiple of cf
        n_trimmed = n[:ny_c*cf, :nx_c*cf, :nz_c*cf]

        if self.config.use_qtensor_coarsening:
            # Average Q-tensors in each block
            n_coarse = np.zeros((ny_c, nx_c, nz_c, 3), dtype=DTYPE)

            for y in range(ny_c):
                for x in range(nx_c):
                    for z in range(nz_c):
                        # Extract block
                        block = n_trimmed[
                            y*cf:(y+1)*cf,
                            x*cf:(x+1)*cf,
                            z*cf:(z+1)*cf
                        ]

                        # Compute average Q-tensor
                        # Q_ij = n_i * n_j - δ_ij/3
                        Q_sum = np.zeros((3, 3), dtype=DTYPE)
                        for n_vec in block.reshape(-1, 3):
                            Q_sum += np.outer(n_vec, n_vec)
                        Q_avg = Q_sum / block.size

                        # Dominant eigenvector
                        eigenvalues, eigenvectors = np.linalg.eigh(Q_avg)
                        n_rep = eigenvectors[:, -1]  # Largest eigenvalue

                        # Normalize
                        norm = np.linalg.norm(n_rep)
                        if norm > 1e-6:
                            n_rep = n_rep / norm

                        n_coarse[y, x, z] = n_rep
        else:
            # Simple averaging (may have cancellation issues)
            n_coarse = np.zeros((ny_c, nx_c, nz_c, 3), dtype=DTYPE)

            for y in range(ny_c):
                for x in range(nx_c):
                    for z in range(nz_c):
                        block = n_trimmed[
                            y*cf:(y+1)*cf,
                            x*cf:(x+1)*cf,
                            z*cf:(z+1)*cf
                        ]

                        # Align all vectors to first before averaging
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
        """
        Upsample signs from coarse to fine level.

        For each fine voxel, align with corresponding coarse voxel.
        """
        cf = self.config.coarsen_factor
        ny_f, nx_f, nz_f = n_fine.shape[:3]
        ny_c, nx_c, nz_c = n_coarse.shape[:3]

        n_result = n_fine.copy()
        flips = 0

        for y in range(ny_f):
            for x in range(nx_f):
                for z in range(nz_f):
                    # Corresponding coarse voxel
                    y_c = min(y // cf, ny_c - 1)
                    x_c = min(x // cf, nx_c - 1)
                    z_c = min(z // cf, nz_c - 1)

                    n_c = n_coarse[y_c, x_c, z_c]
                    n_f = n_fine[y, x, z]

                    # Align fine to coarse
                    if np.dot(n_f, n_c) < 0:
                        n_result[y, x, z] = -n_f
                        flips += 1

        if verbose:
            print(f"    Upsampled: {flips} flips")

        return n_result

    def _optimize_layer_propagation(self, n: np.ndarray) -> np.ndarray:
        """Simple layer propagation for coarsest level."""
        nz = n.shape[2]
        for z in range(1, nz):
            dot = np.sum(n[:, :, z] * n[:, :, z-1], axis=-1)
            flip_mask = dot < 0
            n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
        return n

    def _iterative_refinement(self, n: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """Local iterative refinement."""
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

    def _optimize_with_graph_cuts(self, n: np.ndarray, verbose: bool) -> np.ndarray:
        """Use graph cuts at coarsest level if available."""
        try:
            from .graph_cuts import GraphCutsOptimizer
            director_tmp = DirectorField.from_array(n)
            optimizer = GraphCutsOptimizer()
            result = optimizer.optimize(director_tmp, verbose=False)
            return result.director.to_array()
        except (ImportError, Exception):
            # Fallback to layer propagation
            return self._optimize_layer_propagation(n)


def hierarchical_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[HierarchicalConfig] = None
) -> OptimizationResult:
    """
    Functional interface for hierarchical optimization.

    Args:
        director: Input director field with arbitrary signs.
        verbose: Print progress information.
        config: Optional configuration.

    Returns:
        OptimizationResult with optimized director.

    Example:
    -------
    >>> result = hierarchical_optimization(director, verbose=True)
    >>> optimized = result.director
    """
    optimizer = HierarchicalOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


# Quick test
if __name__ == "__main__":
    from fcpm import create_cholesteric_director, simulate_fcpm
    from fcpm.reconstruction import reconstruct_via_qtensor

    print("Testing Hierarchical Optimization")
    print("=" * 60)

    # Create test data
    print("Creating cholesteric director...")
    director_gt = create_cholesteric_director(shape=(48, 48, 24), pitch=6.0)

    print("Simulating FCPM...")
    I_fcpm = simulate_fcpm(director_gt)

    print("Reconstructing (no sign fix)...")
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm)

    print("\nRunning Hierarchical optimization...")
    optimizer = HierarchicalOptimizer()
    result = optimizer.optimize(director_raw, verbose=True)

    # Compare with ground truth
    from fcpm import summary_metrics
    metrics = summary_metrics(result.director, director_gt)
    print(f"\nAngular error vs GT: {metrics['angular_error_mean_deg']:.2f} deg (mean)")
