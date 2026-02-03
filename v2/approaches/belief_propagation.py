"""
Belief Propagation (Message Passing) for Sign Consistency.

Uses loopy belief propagation on the factor graph representation
to find the MAP (maximum a posteriori) sign assignment.

Mathematical Basis:
------------------
Factor graph representation:
- Variable nodes: signs s_i ∈ {+1, -1} for each voxel
- Factor nodes: pairwise potentials ψ_{ij}(s_i, s_j)

Pairwise potential (from Ising energy):
    ψ_{ij}(s_i, s_j) = exp(-β · |s_i·n̂_i - s_j·n̂_j|²)

where β is inverse temperature (higher = more confident).

Message passing:
    m_{u→v}(s_v) = Σ_{s_u} ψ(s_u, s_v) · Π_{w∈N(u)\\v} m_{w→u}(s_u)

Belief at node v:
    b_v(s_v) ∝ Π_{u∈N(v)} m_{u→v}(s_v)

MAP estimate:
    s*_v = argmax_{s_v} b_v(s_v)

Loopy BP and Damping:
Since 3D grids have loops, standard BP may not converge.
Damping stabilizes convergence:
    m_new ← α·m_computed + (1-α)·m_old

Convergence is not guaranteed but works well in practice for smooth fields.

Complexity: O(iterations × edges) = O(I × 6N)

References:
-----------
[1] Pearl, J. (1988). Probabilistic reasoning in intelligent systems.
[2] Murphy, K. P., Weiss, Y., & Jordan, M. I. (1999). Loopy belief
    propagation for approximate inference.
[3] Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). Understanding
    belief propagation and its generalizations.
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
class BeliefPropagationConfig:
    """Configuration for belief propagation."""

    # Maximum iterations
    max_iterations: int = 100

    # Convergence threshold (max change in messages)
    convergence_threshold: float = 1e-4

    # Damping factor (0 = no damping, 1 = full damping/no update)
    damping: float = 0.5

    # Inverse temperature (higher = more confident potentials)
    beta: float = 2.0

    # Message update schedule: 'parallel' or 'sequential'
    schedule: str = 'parallel'

    # Initialize messages uniformly (True) or from prior (False)
    uniform_init: bool = True


class BeliefPropagationOptimizer:
    """
    Belief propagation optimizer using message passing.

    This optimizer treats sign optimization as probabilistic inference
    and uses loopy BP to approximate the MAP solution.

    Characteristics:
    - Parallelizable message updates
    - Good approximation for smooth fields
    - May not converge for highly irregular fields

    Example:
    -------
    >>> optimizer = BeliefPropagationOptimizer()
    >>> result = optimizer.optimize(director, verbose=True)
    >>> print(f"Converged: {result.converged}")
    """

    def __init__(self, config: Optional[BeliefPropagationConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or BeliefPropagationConfig()

    def optimize(self, director: DirectorField,
                 verbose: bool = False) -> OptimizationResult:
        """
        Optimize director signs using belief propagation.

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
            print("Belief Propagation Sign Optimization")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Damping: {self.config.damping}")
            print(f"Beta: {self.config.beta}")

        # Precompute pairwise potentials
        # For each neighbor direction, compute potential for (same, different) signs
        potentials = self._compute_potentials(n)

        # Initialize messages
        # Messages are stored for each direction: 6 directions per voxel
        # m[y,x,z,d,s] = message from (y,x,z) to neighbor d about sign s ∈ {0,1} (0=+1, 1=-1)
        messages = np.ones((ny, nx, nz, 6, 2), dtype=DTYPE) * 0.5

        if verbose:
            print("\nRunning message passing...")

        # Neighbor directions
        directions = [
            (0, -1, 0, 0),  # y-1
            (1, 1, 0, 0),   # y+1
            (2, 0, -1, 0),  # x-1
            (3, 0, 1, 0),   # x+1
            (4, 0, 0, -1),  # z-1
            (5, 0, 0, 1),   # z+1
        ]

        # Opposite direction mapping
        opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

        converged = False
        iteration = 0

        for iteration in range(self.config.max_iterations):
            if self.config.schedule == 'parallel':
                messages_new = self._parallel_update(
                    n, messages, potentials, directions, opposite
                )
            else:
                messages_new = self._sequential_update(
                    n, messages, potentials, directions, opposite
                )

            # Apply damping
            damping = self.config.damping
            messages_damped = damping * messages + (1 - damping) * messages_new

            # Check convergence
            max_change = np.max(np.abs(messages_damped - messages))

            messages = messages_damped

            if verbose and (iteration < 5 or iteration % 20 == 0):
                # Compute current MAP and energy
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

        # Extract MAP solution
        beliefs = self._compute_beliefs(messages, directions, opposite)
        signs = np.where(beliefs[:, :, :, 0] > beliefs[:, :, :, 1], 1.0, -1.0)
        n_optimized = n * signs[:, :, :, np.newaxis]

        final_energy = compute_gradient_energy(n_optimized)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")

        # Count flips
        total_flips = np.sum(signs < 0)

        result = OptimizationResult(
            director=DirectorField.from_array(n_optimized, metadata={'sign_method': 'belief_propagation'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[],
            flips_by_layer=[],
            total_flips=int(total_flips),
            method='belief_propagation'
        )

        # Add convergence info
        result.converged = converged
        result.iterations = iteration + 1

        return result

    def _compute_potentials(self, n: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Precompute pairwise potentials for each direction.

        Returns dict mapping direction -> potential array.
        potential[y,x,z,0] = ψ(same signs) = exp(-β·|n_i - n_j|²)
        potential[y,x,z,1] = ψ(diff signs) = exp(-β·|n_i + n_j|²)
        """
        ny, nx, nz = n.shape[:3]
        beta = self.config.beta

        potentials = {}

        # Direction deltas
        deltas = [
            (-1, 0, 0),  # 0: y-1
            (1, 0, 0),   # 1: y+1
            (0, -1, 0),  # 2: x-1
            (0, 1, 0),   # 3: x+1
            (0, 0, -1),  # 4: z-1
            (0, 0, 1),   # 5: z+1
        ]

        for d, (dy, dx, dz) in enumerate(deltas):
            pot = np.zeros((ny, nx, nz, 2), dtype=DTYPE)

            # Shifted array for neighbor
            n_neighbor = np.roll(np.roll(np.roll(n, -dy, axis=0), -dx, axis=1), -dz, axis=2)

            # Energy for same signs: |n_i - n_j|²
            energy_same = np.sum((n - n_neighbor) ** 2, axis=-1)

            # Energy for different signs: |n_i + n_j|² = |-n_i - n_j|²
            energy_diff = np.sum((n + n_neighbor) ** 2, axis=-1)

            # Convert to potential (higher is better)
            pot[:, :, :, 0] = np.exp(-beta * energy_same)  # same signs
            pot[:, :, :, 1] = np.exp(-beta * energy_diff)  # different signs

            potentials[d] = pot

        return potentials

    def _parallel_update(self, n: np.ndarray, messages: np.ndarray,
                         potentials: Dict[int, np.ndarray],
                         directions: List[Tuple[int, int, int, int]],
                         opposite: Dict[int, int]) -> np.ndarray:
        """
        Parallel message update (all messages updated simultaneously).
        """
        ny, nx, nz = n.shape[:3]
        messages_new = np.zeros_like(messages)

        for d, dy, dx, dz in directions:
            # Message from (y,x,z) to neighbor in direction d
            # m_{u→v}(s_v) = Σ_{s_u} ψ(s_u, s_v) · Π_{w≠v} m_{w→u}(s_u)

            # Product of incoming messages from all directions except d
            # (the direction we're sending to)
            incoming_product = np.ones((ny, nx, nz, 2), dtype=DTYPE)

            for d2, dy2, dx2, dz2 in directions:
                if d2 == d:
                    continue

                # Message from neighbor in direction d2 to us
                # That neighbor is at (y+dy2, x+dx2, z+dz2)
                # The message it sends is in its opposite direction
                d2_opp = opposite[d2]
                m_incoming = np.roll(
                    np.roll(np.roll(messages[:, :, :, d2_opp, :], -dy2, axis=0),
                            -dx2, axis=1),
                    -dz2, axis=2
                )
                incoming_product *= m_incoming

            # Now compute outgoing message
            pot = potentials[d]

            # For each receiver sign s_v ∈ {0, 1}:
            # m(s_v) = Σ_{s_u} ψ(s_u, s_v) · product[s_u]

            # ψ indexing: pot[:,:,:,0] = same sign, pot[:,:,:,1] = different
            # If s_u = 0 (positive):
            #   s_v = 0 → same → pot[...,0]
            #   s_v = 1 → diff → pot[...,1]
            # If s_u = 1 (negative):
            #   s_v = 0 → diff → pot[...,1]
            #   s_v = 1 → same → pot[...,0]

            # Message for s_v = 0 (positive at receiver)
            m_v0 = (pot[:, :, :, 0] * incoming_product[:, :, :, 0] +  # s_u=0, same
                    pot[:, :, :, 1] * incoming_product[:, :, :, 1])   # s_u=1, diff

            # Message for s_v = 1 (negative at receiver)
            m_v1 = (pot[:, :, :, 1] * incoming_product[:, :, :, 0] +  # s_u=0, diff
                    pot[:, :, :, 0] * incoming_product[:, :, :, 1])   # s_u=1, same

            # Normalize
            m_sum = m_v0 + m_v1 + 1e-10
            messages_new[:, :, :, d, 0] = m_v0 / m_sum
            messages_new[:, :, :, d, 1] = m_v1 / m_sum

        return messages_new

    def _sequential_update(self, n: np.ndarray, messages: np.ndarray,
                           potentials: Dict[int, np.ndarray],
                           directions: List[Tuple[int, int, int, int]],
                           opposite: Dict[int, int]) -> np.ndarray:
        """
        Sequential message update (one direction at a time).

        Often converges faster than parallel for loopy graphs.
        """
        messages_new = messages.copy()

        for d, dy, dx, dz in directions:
            ny, nx, nz = n.shape[:3]

            # Compute incoming product with current messages
            incoming_product = np.ones((ny, nx, nz, 2), dtype=DTYPE)

            for d2, dy2, dx2, dz2 in directions:
                if d2 == d:
                    continue

                d2_opp = opposite[d2]
                m_incoming = np.roll(
                    np.roll(np.roll(messages_new[:, :, :, d2_opp, :], -dy2, axis=0),
                            -dx2, axis=1),
                    -dz2, axis=2
                )
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
        """
        Compute beliefs from converged messages.

        b_v(s_v) ∝ Π_{u∈N(v)} m_{u→v}(s_v)
        """
        ny, nx, nz = messages.shape[:3]
        beliefs = np.ones((ny, nx, nz, 2), dtype=DTYPE)

        for d, dy, dx, dz in directions:
            # Message from neighbor to us is stored at neighbor's position
            # in the opposite direction
            d_opp = opposite[d]
            m_incoming = np.roll(
                np.roll(np.roll(messages[:, :, :, d_opp, :], -dy, axis=0),
                        -dx, axis=1),
                -dz, axis=2
            )
            beliefs *= m_incoming

        # Normalize
        beliefs_sum = beliefs.sum(axis=-1, keepdims=True) + 1e-10
        beliefs = beliefs / beliefs_sum

        return beliefs


def belief_propagation_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[BeliefPropagationConfig] = None
) -> OptimizationResult:
    """
    Functional interface for belief propagation optimization.

    Args:
        director: Input director field with arbitrary signs.
        verbose: Print progress information.
        config: Optional configuration.

    Returns:
        OptimizationResult with optimized director.

    Example:
    -------
    >>> result = belief_propagation_optimization(director, verbose=True)
    >>> optimized = result.director
    """
    optimizer = BeliefPropagationOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


# Quick test
if __name__ == "__main__":
    from fcpm import create_cholesteric_director, simulate_fcpm
    from fcpm.reconstruction import reconstruct_via_qtensor

    print("Testing Belief Propagation Optimization")
    print("=" * 60)

    # Create test data
    print("Creating cholesteric director...")
    director_gt = create_cholesteric_director(shape=(32, 32, 16), pitch=6.0)

    print("Simulating FCPM...")
    I_fcpm = simulate_fcpm(director_gt)

    print("Reconstructing (no sign fix)...")
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm)

    print("\nRunning Belief Propagation optimization...")
    config = BeliefPropagationConfig(
        max_iterations=50,
        damping=0.5,
        beta=2.0,
    )
    optimizer = BeliefPropagationOptimizer(config)
    result = optimizer.optimize(director_raw, verbose=True)

    # Compare with ground truth
    from fcpm import summary_metrics
    metrics = summary_metrics(result.director, director_gt)
    print(f"\nAngular error vs GT: {metrics['angular_error_mean_deg']:.2f} deg (mean)")
