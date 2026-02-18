"""
Belief Propagation (Message Passing) for Sign Consistency.

Uses loopy belief propagation on the factor graph representation
to find the MAP (maximum a posteriori) sign assignment.

**Experimental**: convergence is not guaranteed for loopy graphs.
Damping stabilises oscillating messages in practice.

Mathematical Basis:
------------------
Factor graph with:
- Variable nodes: signs s_i in {+1, -1}
- Factor nodes: pairwise potentials psi_{ij}(s_i, s_j)

Message passing with damping:
    m_new <- alpha * m_old + (1 - alpha) * m_computed

References:
-----------
[1] Pearl, J. (1988). Probabilistic reasoning in intelligent systems.
[2] Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). Understanding
    belief propagation and its generalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.director import DirectorField, DTYPE
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy


@dataclass
class BeliefPropagationConfig:
    """Configuration for belief propagation."""

    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    damping: float = 0.5
    beta: float = 2.0
    schedule: str = 'parallel'
    uniform_init: bool = True


class BeliefPropagationOptimizer(SignOptimizer):
    """
    Belief propagation optimizer using message passing.

    Treats sign optimization as probabilistic inference and uses
    loopy BP to approximate the MAP solution.

    .. note::
        This optimizer is **experimental**. It may not converge for
        highly irregular fields.

    Example::

        config = BeliefPropagationConfig(max_iterations=50)
        optimizer = BeliefPropagationOptimizer(config)
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[BeliefPropagationConfig] = None):
        self.config = config or BeliefPropagationConfig()

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
            print("Belief Propagation Sign Optimization")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Damping: {self.config.damping}")
            print(f"Beta: {self.config.beta}")

        potentials = self._compute_potentials(n)
        messages = np.ones((ny, nx, nz, 6, 2), dtype=DTYPE) * 0.5

        if verbose:
            print("\nRunning message passing...")

        directions: List[Tuple[int, int, int, int]] = [
            (0, -1, 0, 0),
            (1, 1, 0, 0),
            (2, 0, -1, 0),
            (3, 0, 1, 0),
            (4, 0, 0, -1),
            (5, 0, 0, 1),
        ]
        opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

        converged = False
        iteration = 0

        for iteration in range(self.config.max_iterations):
            if self.config.schedule == 'parallel':
                messages_new = self._parallel_update(n, messages, potentials, directions, opposite)
            else:
                messages_new = self._sequential_update(n, messages, potentials, directions, opposite)

            damping = self.config.damping
            messages_damped = damping * messages + (1 - damping) * messages_new

            max_change = float(np.max(np.abs(messages_damped - messages)))
            messages = messages_damped

            if verbose and (iteration < 5 or iteration % 20 == 0):
                beliefs = self._compute_beliefs(messages, directions, opposite)
                signs = np.where(beliefs[:, :, :, 0] > beliefs[:, :, :, 1], 1.0, -1.0)
                n_temp = n * signs[:, :, :, np.newaxis]
                energy = compute_gradient_energy(n_temp)
                print(f"  Iter {iteration:3d}: max_change={max_change:.6f}, energy={energy:.2f}")

            if max_change < self.config.convergence_threshold:
                converged = True
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

        if verbose and not converged:
            print(f"\nDid not converge after {self.config.max_iterations} iterations")

        beliefs = self._compute_beliefs(messages, directions, opposite)
        signs = np.where(beliefs[:, :, :, 0] > beliefs[:, :, :, 1], 1.0, -1.0)
        n_optimized = n * signs[:, :, :, np.newaxis]

        final_energy = compute_gradient_energy(n_optimized)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")

        total_flips = int(np.sum(signs < 0))

        return OptimizationResult(
            director=DirectorField.from_array(
                n_optimized, metadata={'sign_method': 'belief_propagation'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_flips=total_flips,
            method='belief_propagation',
            metadata={
                'converged': converged,
                'iterations': iteration + 1,
            },
        )

    # ---------- internal helpers ----------

    def _compute_potentials(self, n: np.ndarray) -> Dict[int, np.ndarray]:
        ny, nx, nz = n.shape[:3]
        beta = self.config.beta
        potentials: Dict[int, np.ndarray] = {}

        deltas = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]

        for d, (dy, dx, dz) in enumerate(deltas):
            pot = np.zeros((ny, nx, nz, 2), dtype=DTYPE)
            n_neighbor = np.roll(np.roll(np.roll(n, -dy, axis=0), -dx, axis=1), -dz, axis=2)

            energy_same = np.sum((n - n_neighbor) ** 2, axis=-1)
            energy_diff = np.sum((n + n_neighbor) ** 2, axis=-1)

            pot[:, :, :, 0] = np.exp(-beta * energy_same)
            pot[:, :, :, 1] = np.exp(-beta * energy_diff)
            potentials[d] = pot

        return potentials

    def _parallel_update(self, n: np.ndarray, messages: np.ndarray,
                         potentials: Dict[int, np.ndarray],
                         directions: List[Tuple[int, int, int, int]],
                         opposite: Dict[int, int]) -> np.ndarray:
        ny, nx, nz = n.shape[:3]
        messages_new = np.zeros_like(messages)

        for d, dy, dx, dz in directions:
            incoming_product = np.ones((ny, nx, nz, 2), dtype=DTYPE)

            for d2, dy2, dx2, dz2 in directions:
                if d2 == d:
                    continue
                d2_opp = opposite[d2]
                m_incoming = np.roll(
                    np.roll(np.roll(messages[:, :, :, d2_opp, :], -dy2, axis=0),
                            -dx2, axis=1), -dz2, axis=2)
                incoming_product *= m_incoming

            pot = potentials[d]

            m_v0 = (pot[:, :, :, 0] * incoming_product[:, :, :, 0] +
                    pot[:, :, :, 1] * incoming_product[:, :, :, 1])
            m_v1 = (pot[:, :, :, 1] * incoming_product[:, :, :, 0] +
                    pot[:, :, :, 0] * incoming_product[:, :, :, 1])

            m_sum = m_v0 + m_v1 + 1e-10
            messages_new[:, :, :, d, 0] = m_v0 / m_sum
            messages_new[:, :, :, d, 1] = m_v1 / m_sum

        return messages_new

    def _sequential_update(self, n: np.ndarray, messages: np.ndarray,
                           potentials: Dict[int, np.ndarray],
                           directions: List[Tuple[int, int, int, int]],
                           opposite: Dict[int, int]) -> np.ndarray:
        messages_new = messages.copy()

        for d, dy, dx, dz in directions:
            ny, nx, nz = n.shape[:3]
            incoming_product = np.ones((ny, nx, nz, 2), dtype=DTYPE)

            for d2, dy2, dx2, dz2 in directions:
                if d2 == d:
                    continue
                d2_opp = opposite[d2]
                m_incoming = np.roll(
                    np.roll(np.roll(messages_new[:, :, :, d2_opp, :], -dy2, axis=0),
                            -dx2, axis=1), -dz2, axis=2)
                incoming_product *= m_incoming

            pot = potentials[d]
            m_v0 = (pot[:, :, :, 0] * incoming_product[:, :, :, 0] +
                    pot[:, :, :, 1] * incoming_product[:, :, :, 1])
            m_v1 = (pot[:, :, :, 1] * incoming_product[:, :, :, 0] +
                    pot[:, :, :, 0] * incoming_product[:, :, :, 1])

            m_sum = m_v0 + m_v1 + 1e-10
            messages_new[:, :, :, d, 0] = m_v0 / m_sum
            messages_new[:, :, :, d, 1] = m_v1 / m_sum

        return messages_new

    def _compute_beliefs(self, messages: np.ndarray,
                         directions: List[Tuple[int, int, int, int]],
                         opposite: Dict[int, int]) -> np.ndarray:
        ny, nx, nz = messages.shape[:3]
        beliefs = np.ones((ny, nx, nz, 2), dtype=DTYPE)

        for d, dy, dx, dz in directions:
            d_opp = opposite[d]
            m_incoming = np.roll(
                np.roll(np.roll(messages[:, :, :, d_opp, :], -dy, axis=0),
                        -dx, axis=1), -dz, axis=2)
            beliefs *= m_incoming

        beliefs_sum = beliefs.sum(axis=-1, keepdims=True) + 1e-10
        beliefs = beliefs / beliefs_sum
        return beliefs


def belief_propagation_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[BeliefPropagationConfig] = None,
) -> OptimizationResult:
    """Functional interface for belief propagation optimization."""
    optimizer = BeliefPropagationOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "BeliefPropagationConfig",
    "BeliefPropagationOptimizer",
    "belief_propagation_optimization",
]
