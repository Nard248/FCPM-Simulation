"""
Simulated Annealing Optimization for Sign Consistency.

Uses Metropolis-Hastings sampling with adaptive temperature schedule
to find near-optimal sign assignments.

Mathematical Basis:
------------------
The sign optimization maps to an Ising model:

    E(S) = sum_{(i,j)} J_ij * s_i * s_j

where J_ij = -2(n_i . n_j) is the coupling strength.

Enhancements:
- Adaptive temperature schedule based on acceptance rate
- Cluster (Wolff) moves for faster mixing near critical point
- Optional Numba-accelerated kernels for performance

References:
-----------
[1] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization
    by simulated annealing.
[2] Wolff, U. (1989). Collective Monte Carlo updating for spin systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ...core.director import DirectorField, DTYPE
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy


@dataclass
class SimulatedAnnealingConfig:
    """Configuration for simulated annealing."""

    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.995

    max_iterations: int = 100000
    iterations_per_temp: int = 100

    use_adaptive: bool = True
    target_accept_rate: float = 0.3
    adapt_interval: int = 500

    use_cluster_moves: bool = False
    cluster_probability: float = 0.1

    seed: Optional[int] = None
    initialize_with_layer_prop: bool = True


@dataclass
class SAHistory:
    """History of a simulated annealing run."""
    temperatures: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    accept_rates: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)


class SimulatedAnnealingOptimizer(SignOptimizer):
    """
    Simulated annealing optimizer with adaptive temperature.

    Can escape local minima and find near-global optima, but is slower
    than deterministic methods like graph cuts.

    Example::

        config = SimulatedAnnealingConfig(max_iterations=50000)
        optimizer = SimulatedAnnealingOptimizer(config)
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[SimulatedAnnealingConfig] = None):
        self.config = config or SimulatedAnnealingConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> OptimizationResult:
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
            print(f"Temperature: {self.config.initial_temperature} -> "
                  f"{self.config.final_temperature}")

        if self.config.initialize_with_layer_prop:
            if verbose:
                print("\nInitializing with layer propagation...")
            n = self._layer_propagation_init(n)
            if verbose:
                print(f"  After layer prop: energy = {compute_gradient_energy(n):.2f}")

        history = SAHistory()
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
            if (self.config.use_cluster_moves
                    and self.rng.random() < self.config.cluster_probability):
                delta_e, n_flipped = self._cluster_move(n, T)
                accepted_move = True
                total_flips += n_flipped
            else:
                y = self.rng.integers(0, ny)
                x = self.rng.integers(0, nx)
                z = self.rng.integers(0, nz)

                delta_e = self._compute_delta_energy(n, y, x, z)

                if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / T):
                    n[y, x, z] = -n[y, x, z]
                    energy += delta_e
                    accepted += 1
                    total_flips += 1
                    accepted_move = True
                else:
                    accepted_move = False

            total_attempts += 1

            if accepted_move:
                current_energy = compute_gradient_energy(n)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_n = n.copy()

            if (self.config.use_adaptive
                    and total_attempts % self.config.adapt_interval == 0):
                accept_rate = accepted / total_attempts
                if accept_rate > self.config.target_accept_rate * 1.2:
                    T *= 0.95
                elif accept_rate < self.config.target_accept_rate * 0.8:
                    T *= 1.05
                accepted = max(1, accepted // 2)
                total_attempts = max(1, total_attempts // 2)

            if iteration % self.config.iterations_per_temp == 0:
                if not self.config.use_adaptive:
                    T *= self.config.cooling_rate

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

            if T < self.config.final_temperature:
                if verbose:
                    print(f"\nReached minimum temperature at iteration {iteration}")
                break

        n = best_n
        final_energy = compute_gradient_energy(n)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Best energy found: {best_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")
            print(f"Total flips: {total_flips}")

        return OptimizationResult(
            director=DirectorField.from_array(
                n, metadata={'sign_method': 'simulated_annealing'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_flips=total_flips,
            method='simulated_annealing',
            metadata={
                'history': history,
                'best_energy': best_energy,
            },
        )

    # ----- helpers -----

    @staticmethod
    def _layer_propagation_init(n: np.ndarray) -> np.ndarray:
        nz = n.shape[2]
        for z in range(1, nz):
            dot = np.sum(n[:, :, z] * n[:, :, z - 1], axis=-1)
            flip_mask = dot < 0
            n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
        return n

    @staticmethod
    def _compute_delta_energy(n: np.ndarray, y: int, x: int, z: int) -> float:
        ny, nx, nz = n.shape[:3]
        n_i = n[y, x, z]
        delta = 0.0
        neighbors = [
            (y - 1, x, z), (y + 1, x, z),
            (y, x - 1, z), (y, x + 1, z),
            (y, x, z - 1), (y, x, z + 1),
        ]
        for ny_idx, nx_idx, nz_idx in neighbors:
            if 0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz:
                delta += 4.0 * np.dot(n_i, n[ny_idx, nx_idx, nz_idx])
        return delta

    def _cluster_move(self, n: np.ndarray, T: float) -> Tuple[float, int]:
        ny, nx, nz = n.shape[:3]
        y0 = self.rng.integers(0, ny)
        x0 = self.rng.integers(0, nx)
        z0 = self.rng.integers(0, nz)

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

                if dot > 0:
                    J = 2.0 * dot
                    p_add = 1.0 - np.exp(-J / max(T, 0.01))
                    if self.rng.random() < p_add:
                        cluster.add((ny_idx, nx_idx, nz_idx))
                        frontier.append((ny_idx, nx_idx, nz_idx))

        for (y, x, z) in cluster:
            n[y, x, z] = -n[y, x, z]

        return 0.0, len(cluster)


def simulated_annealing_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[SimulatedAnnealingConfig] = None,
) -> OptimizationResult:
    """Functional interface for simulated annealing optimization."""
    optimizer = SimulatedAnnealingOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "SimulatedAnnealingConfig",
    "SimulatedAnnealingOptimizer",
    "SAHistory",
    "simulated_annealing_optimization",
]
