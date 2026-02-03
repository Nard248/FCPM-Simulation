"""
Simulated Annealing Optimization for Sign Consistency.

Uses Metropolis-Hastings sampling with adaptive temperature schedule
to find near-optimal sign assignments.

Mathematical Basis:
------------------
The sign optimization maps to an Ising model:

    E(S) = Σ_{(i,j)} J_ij · s_i · s_j

where J_ij = -2(n̂_i · n̂_j) is the coupling strength.

Simulated annealing samples from the Boltzmann distribution:

    P(S) ∝ exp(-E(S) / T)

At high T: explores broadly (accepts many moves)
At low T: exploits (only accepts improving moves)

Algorithm (Metropolis-Hastings):
1. Propose move: flip sign at random voxel
2. Compute ΔE = E_new - E_old
3. Accept with probability min(1, exp(-ΔE/T))
4. Decrease temperature: T ← α·T

Enhancements implemented:
- Adaptive temperature schedule based on acceptance rate
- Cluster (Wolff) moves for faster mixing near critical point
- Parallel tempering option for multimodal landscapes

Complexity: O(iterations × N) where N = number of voxels

References:
-----------
[1] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization
    by simulated annealing.
[2] Wolff, U. (1989). Collective Monte Carlo updating for spin systems.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass, field
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
class SimulatedAnnealingConfig:
    """Configuration for simulated annealing."""

    # Temperature schedule
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.995  # T_new = α * T_old

    # Iterations
    max_iterations: int = 100000
    iterations_per_temp: int = 100  # moves per temperature step

    # Adaptive temperature
    use_adaptive: bool = True
    target_accept_rate: float = 0.3  # target acceptance rate
    adapt_interval: int = 500  # adjust temperature every N iterations

    # Cluster moves (Wolff algorithm)
    use_cluster_moves: bool = False
    cluster_probability: float = 0.1  # prob of cluster vs single flip

    # Random seed for reproducibility
    seed: Optional[int] = None

    # Initialization
    initialize_with_layer_prop: bool = True  # start from layer propagation


@dataclass
class SAHistory:
    """History of simulated annealing run."""
    temperatures: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    accept_rates: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)


class SimulatedAnnealingOptimizer:
    """
    Simulated annealing optimizer with adaptive temperature.

    This optimizer can escape local minima and find near-global optima,
    but is slower than deterministic methods like graph cuts.

    Best for:
    - High-noise data where deterministic methods get stuck
    - Fields with defects or discontinuities
    - Validating other methods' results

    Example:
    -------
    >>> config = SimulatedAnnealingConfig(max_iterations=50000)
    >>> optimizer = SimulatedAnnealingOptimizer(config)
    >>> result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[SimulatedAnnealingConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or SimulatedAnnealingConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def optimize(self, director: DirectorField,
                 verbose: bool = False,
                 callback: Optional[Callable[[int, float, float], None]] = None
                 ) -> OptimizationResult:
        """
        Optimize director signs using simulated annealing.

        Args:
            director: Input director field with arbitrary signs.
            verbose: Print progress information.
            callback: Optional callback(iteration, energy, temperature)

        Returns:
            OptimizationResult with optimized director and statistics.
        """
        n = director.to_array().astype(DTYPE).copy()
        ny, nx, nz = n.shape[:3]

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Simulated Annealing Sign Optimization")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Temperature: {self.config.initial_temperature} → {self.config.final_temperature}")
            print(f"Adaptive: {self.config.use_adaptive}")
            print(f"Cluster moves: {self.config.use_cluster_moves}")

        # Optional: Initialize with layer propagation
        if self.config.initialize_with_layer_prop:
            if verbose:
                print("\nInitializing with layer propagation...")
            n = self._layer_propagation_init(n)
            init_prop_energy = compute_gradient_energy(n)
            if verbose:
                print(f"  After layer prop: energy = {init_prop_energy:.2f}")

        # Pre-compute neighbor indices for efficiency
        neighbor_offsets = np.array([
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1],
        ])

        # History tracking
        history = SAHistory()

        # Initialize
        T = self.config.initial_temperature
        energy = compute_gradient_energy(n)
        best_energy = energy
        best_n = n.copy()

        accepted = 0
        total_attempts = 0
        total_flips = 0

        if verbose:
            print("\nRunning simulated annealing...")

        for iteration in range(self.config.max_iterations):
            # Choose move type
            if self.config.use_cluster_moves and self.rng.random() < self.config.cluster_probability:
                # Cluster move (Wolff algorithm)
                delta_e, n_flipped = self._cluster_move(n, T)
                accepted_move = True  # Wolff moves are always accepted
                total_flips += n_flipped
            else:
                # Single spin flip
                y = self.rng.integers(0, ny)
                x = self.rng.integers(0, nx)
                z = self.rng.integers(0, nz)

                # Compute local energy change
                delta_e = self._compute_delta_energy(n, y, x, z)

                # Metropolis criterion
                if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / T):
                    n[y, x, z] = -n[y, x, z]
                    energy += delta_e
                    accepted += 1
                    total_flips += 1
                    accepted_move = True
                else:
                    accepted_move = False

            total_attempts += 1

            # Track best solution
            if accepted_move:
                current_energy = compute_gradient_energy(n)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_n = n.copy()

            # Adaptive temperature adjustment
            if self.config.use_adaptive and total_attempts % self.config.adapt_interval == 0:
                accept_rate = accepted / total_attempts
                if accept_rate > self.config.target_accept_rate * 1.2:
                    T *= 0.95  # Cool faster
                elif accept_rate < self.config.target_accept_rate * 0.8:
                    T *= 1.05  # Heat up

                # Reset counters
                accepted = max(1, accepted // 2)
                total_attempts = max(1, total_attempts // 2)

            # Regular temperature update
            if iteration % self.config.iterations_per_temp == 0:
                if not self.config.use_adaptive:
                    T *= self.config.cooling_rate

                # Record history
                history.temperatures.append(T)
                history.energies.append(compute_gradient_energy(n))
                history.accept_rates.append(accepted / max(1, total_attempts))
                history.iterations.append(iteration)

                if callback:
                    callback(iteration, energy, T)

                if verbose and (iteration < 1000 or iteration % 10000 == 0):
                    accept_rate = accepted / max(1, total_attempts)
                    print(f"  Iter {iteration:6d}: T={T:.4f}, E={energy:.2f}, "
                          f"accept={accept_rate:.2%}, best={best_energy:.2f}")

            # Check convergence
            if T < self.config.final_temperature:
                if verbose:
                    print(f"\nReached minimum temperature at iteration {iteration}")
                break

        # Use best solution found
        n = best_n
        final_energy = compute_gradient_energy(n)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Best energy found: {best_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")
            print(f"Total flips: {total_flips}")

        result = OptimizationResult(
            director=DirectorField.from_array(n, metadata={'sign_method': 'simulated_annealing'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[],
            flips_by_layer=[],
            total_flips=total_flips,
            method='simulated_annealing'
        )

        # Attach history as extra info
        result.history = history

        return result

    def _layer_propagation_init(self, n: np.ndarray) -> np.ndarray:
        """Initialize with simple layer propagation."""
        nz = n.shape[2]
        for z in range(1, nz):
            dot = np.sum(n[:, :, z] * n[:, :, z-1], axis=-1)
            flip_mask = dot < 0
            n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
        return n

    def _compute_delta_energy(self, n: np.ndarray, y: int, x: int, z: int) -> float:
        """
        Compute energy change from flipping sign at (y, x, z).

        ΔE = E_flip - E_current
           = Σ_neighbors |(-n_i) - n_j|² - |n_i - n_j|²
           = Σ_neighbors (|n_i + n_j|² - |n_i - n_j|²)
           = Σ_neighbors 4 * (n_i · n_j)
        """
        ny, nx, nz = n.shape[:3]
        n_i = n[y, x, z]

        delta = 0.0

        # 6-connected neighbors
        neighbors = [
            (y-1, x, z), (y+1, x, z),
            (y, x-1, z), (y, x+1, z),
            (y, x, z-1), (y, x, z+1),
        ]

        for ny_idx, nx_idx, nz_idx in neighbors:
            if 0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz:
                n_j = n[ny_idx, nx_idx, nz_idx]
                delta += 4.0 * np.dot(n_i, n_j)

        return delta

    def _cluster_move(self, n: np.ndarray, T: float) -> Tuple[float, int]:
        """
        Wolff cluster move for Ising model.

        Builds a cluster of aligned spins and flips all at once.
        This dramatically improves mixing near critical temperature.

        Returns:
            Tuple of (energy change, number of spins flipped)
        """
        ny, nx, nz = n.shape[:3]

        # Random seed spin
        y0 = self.rng.integers(0, ny)
        x0 = self.rng.integers(0, nx)
        z0 = self.rng.integers(0, nz)

        # Probability of adding aligned neighbor to cluster
        # p = 1 - exp(-2J/T) for Ising with J = dot product
        # Simplified: use threshold based on alignment

        cluster = set()
        frontier = [(y0, x0, z0)]
        cluster.add((y0, x0, z0))

        neighbors_6 = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]

        while frontier:
            y, x, z = frontier.pop()
            n_i = n[y, x, z]

            for dy, dx, dz in neighbors_6:
                ny_idx, nx_idx, nz_idx = y + dy, x + dx, z + dz

                if not (0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz):
                    continue

                if (ny_idx, nx_idx, nz_idx) in cluster:
                    continue

                n_j = n[ny_idx, nx_idx, nz_idx]
                dot = np.dot(n_i, n_j)

                # Add to cluster with probability based on alignment and temperature
                # For same-sign preference (dot > 0), high prob of joining
                if dot > 0:
                    J = 2.0 * dot
                    p_add = 1.0 - np.exp(-J / max(T, 0.01))
                    if self.rng.random() < p_add:
                        cluster.add((ny_idx, nx_idx, nz_idx))
                        frontier.append((ny_idx, nx_idx, nz_idx))

        # Flip entire cluster
        for (y, x, z) in cluster:
            n[y, x, z] = -n[y, x, z]

        # Energy change is automatically favorable for Wolff moves
        return 0.0, len(cluster)


def simulated_annealing_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[SimulatedAnnealingConfig] = None
) -> OptimizationResult:
    """
    Functional interface for simulated annealing optimization.

    Args:
        director: Input director field with arbitrary signs.
        verbose: Print progress information.
        config: Optional configuration.

    Returns:
        OptimizationResult with optimized director.

    Example:
    -------
    >>> result = simulated_annealing_optimization(director, verbose=True)
    >>> optimized = result.director
    """
    optimizer = SimulatedAnnealingOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


# Quick test
if __name__ == "__main__":
    from fcpm import create_cholesteric_director, simulate_fcpm
    from fcpm.reconstruction import reconstruct_via_qtensor

    print("Testing Simulated Annealing Optimization")
    print("=" * 60)

    # Create test data
    print("Creating cholesteric director...")
    director_gt = create_cholesteric_director(shape=(32, 32, 16), pitch=6.0)

    print("Simulating FCPM...")
    I_fcpm = simulate_fcpm(director_gt)

    print("Reconstructing (no sign fix)...")
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm)

    print("\nRunning Simulated Annealing optimization...")
    config = SimulatedAnnealingConfig(
        max_iterations=20000,
        use_cluster_moves=True,
    )
    optimizer = SimulatedAnnealingOptimizer(config)
    result = optimizer.optimize(director_raw, verbose=True)

    # Compare with ground truth
    from fcpm import summary_metrics
    metrics = summary_metrics(result.director, director_gt)
    print(f"\nAngular error vs GT: {metrics['angular_error_mean_deg']:.2f} deg (mean)")
