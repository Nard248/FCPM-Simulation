"""
Sign Optimization for Director Field Recovery.

After eigendecomposition of the Q-tensor, each voxel has a director ±n.
The goal is to choose consistent signs such that the field is smooth
(minimizes the gradient energy |∇n|²).

Algorithms:
1. Chain propagation (BFS from seed) - fast, works well for simple fields
2. Iterative local flipping - refines after chain propagation
3. Wavefront propagation - directional sweep
4. Combined approach - chain propagation + local refinement

Key insight: In nematics, n ≡ -n. Both choices are physically correct.
The goal is CONSISTENCY, not finding the "true" sign.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
from ..core.director import DirectorField, DTYPE


def chain_propagation(director: DirectorField,
                      start: Optional[Tuple[int, int, int]] = None) -> DirectorField:
    """
    Fix director signs via BFS chain propagation from a seed point.

    Algorithm:
    1. Start from seed point with fixed orientation
    2. BFS through all voxels
    3. For each new voxel, align with the already-fixed neighbor

    This ensures local consistency but may create domain walls if
    the field has topological defects.

    Args:
        director: Input director field (signs may be inconsistent).
        start: Starting point (y, x, z). If None, uses center.

    Returns:
        DirectorField with consistent signs.
    """
    shape = director.shape
    ny_dim, nx_dim, nz_dim = shape

    if start is None:
        start = (ny_dim // 2, nx_dim // 2, nz_dim // 2)

    # Work with a stacked array for efficiency
    n = director.to_array().copy()

    visited = np.zeros(shape, dtype=bool)
    queue = deque([start])
    visited[start] = True

    # 6-connected neighbors
    neighbor_offsets = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    while queue:
        cy, cx, cz = queue.popleft()
        n_curr = n[cy, cx, cz]

        for dy, dx, dz in neighbor_offsets:
            ny_idx = cy + dy
            nx_idx = cx + dx
            nz_idx = cz + dz

            # Boundary check
            if not (0 <= ny_idx < ny_dim and
                    0 <= nx_idx < nx_dim and
                    0 <= nz_idx < nz_dim):
                continue

            if visited[ny_idx, nx_idx, nz_idx]:
                continue

            # Neighbor's director
            n_neigh = n[ny_idx, nx_idx, nz_idx]

            # Flip if anti-aligned
            if np.dot(n_curr, n_neigh) < 0:
                n[ny_idx, nx_idx, nz_idx] = -n_neigh

            visited[ny_idx, nx_idx, nz_idx] = True
            queue.append((ny_idx, nx_idx, nz_idx))

    return DirectorField.from_array(n, metadata={'sign_method': 'chain_propagation'})


def iterative_local_flip(director: DirectorField,
                         max_iter: int = 100,
                         verbose: bool = False) -> Tuple[DirectorField, Dict]:
    """
    Iteratively flip signs to minimize gradient energy.

    At each iteration, for each voxel, compare the cost of keeping
    the current sign vs. flipping. Flip if it reduces energy.

    Energy: E = Σ |n_i - n_j|² over neighbor pairs

    Args:
        director: Input director field.
        max_iter: Maximum iterations.
        verbose: Print progress.

    Returns:
        Tuple of (optimized DirectorField, info dict).
    """
    n = director.to_array().astype(DTYPE)
    shape = n.shape[:3]

    history = []
    initial_energy = _compute_gradient_energy(n)

    if verbose:
        print(f"Initial gradient energy: {initial_energy:.2f}")

    for iteration in range(max_iter):
        # Compute cost for current and flipped orientation
        cost_curr = np.zeros(shape, dtype=DTYPE)
        cost_flip = np.zeros(shape, dtype=DTYPE)

        for axis in range(3):
            # Forward neighbor
            n_fwd = np.roll(n, -1, axis=axis)
            cost_curr += np.sum((n - n_fwd) ** 2, axis=-1)
            cost_flip += np.sum((n + n_fwd) ** 2, axis=-1)

            # Backward neighbor
            n_bwd = np.roll(n, 1, axis=axis)
            cost_curr += np.sum((n - n_bwd) ** 2, axis=-1)
            cost_flip += np.sum((n + n_bwd) ** 2, axis=-1)

        # Flip where beneficial
        flip_mask = cost_flip < cost_curr
        n_flipped = np.sum(flip_mask)

        energy = _compute_gradient_energy(n)
        history.append({
            'iteration': iteration,
            'flipped': int(n_flipped),
            'energy': float(energy)
        })

        if verbose and (iteration < 5 or iteration % 10 == 0):
            print(f"  Iter {iteration}: flipped {n_flipped} voxels, energy = {energy:.2f}")

        if n_flipped == 0:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

        n[flip_mask] = -n[flip_mask]

    final_energy = _compute_gradient_energy(n)

    info = {
        'iterations': iteration + 1,
        'converged': n_flipped == 0,
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_reduction': initial_energy - final_energy,
        'history': history,
    }

    return DirectorField.from_array(n, metadata={'sign_method': 'iterative_flip'}), info


def wavefront_propagation(director: DirectorField,
                          direction: str = 'z+') -> DirectorField:
    """
    Propagate sign fixing as a wavefront in a specific direction.

    This sweeps through the volume, aligning each plane with the previous.
    Useful when there's a known "good" starting plane.

    Args:
        director: Input director field.
        direction: Propagation direction ('x+', 'x-', 'y+', 'y-', 'z+', 'z-').

    Returns:
        DirectorField with fixed signs.
    """
    n = director.to_array().astype(DTYPE)
    shape = director.shape

    if direction == 'z+':
        for z in range(1, shape[2]):
            dot = np.sum(n[:, :, z-1] * n[:, :, z], axis=-1, keepdims=True)
            n[:, :, z] = np.where(dot < 0, -n[:, :, z], n[:, :, z])

    elif direction == 'z-':
        for z in range(shape[2] - 2, -1, -1):
            dot = np.sum(n[:, :, z+1] * n[:, :, z], axis=-1, keepdims=True)
            n[:, :, z] = np.where(dot < 0, -n[:, :, z], n[:, :, z])

    elif direction == 'y+':
        for y in range(1, shape[0]):
            dot = np.sum(n[y-1, :, :] * n[y, :, :], axis=-1, keepdims=True)
            n[y, :, :] = np.where(dot < 0, -n[y, :, :], n[y, :, :])

    elif direction == 'y-':
        for y in range(shape[0] - 2, -1, -1):
            dot = np.sum(n[y+1, :, :] * n[y, :, :], axis=-1, keepdims=True)
            n[y, :, :] = np.where(dot < 0, -n[y, :, :], n[y, :, :])

    elif direction == 'x+':
        for x in range(1, shape[1]):
            dot = np.sum(n[:, x-1, :] * n[:, x, :], axis=-1, keepdims=True)
            n[:, x, :] = np.where(dot < 0, -n[:, x, :], n[:, x, :])

    elif direction == 'x-':
        for x in range(shape[1] - 2, -1, -1):
            dot = np.sum(n[:, x+1, :] * n[:, x, :], axis=-1, keepdims=True)
            n[:, x, :] = np.where(dot < 0, -n[:, x, :], n[:, x, :])

    else:
        raise ValueError(f"Invalid direction: {direction}. "
                        f"Use 'x+', 'x-', 'y+', 'y-', 'z+', or 'z-'.")

    return DirectorField.from_array(n, metadata={'sign_method': f'wavefront_{direction}'})


def multi_axis_propagation(director: DirectorField,
                           axes_order: List[int] = [2, 0, 1]) -> DirectorField:
    """
    Apply vectorized chain propagation along multiple axes.

    Propagates from the center slice outward along each axis.

    Args:
        director: Input director field.
        axes_order: Order of axes to propagate (default: z, y, x).

    Returns:
        DirectorField with fixed signs.
    """
    n = director.to_array().astype(DTYPE)

    for axis in axes_order:
        dim = n.shape[axis]
        mid = dim // 2

        # Forward propagation from center
        for i in range(mid + 1, dim):
            slices_curr = [slice(None)] * 4
            slices_prev = [slice(None)] * 4
            slices_curr[axis] = i
            slices_prev[axis] = i - 1

            n_curr = n[tuple(slices_curr)]
            n_prev = n[tuple(slices_prev)]

            dot = np.sum(n_curr * n_prev, axis=-1, keepdims=True)
            n[tuple(slices_curr)] = np.where(dot < 0, -n_curr, n_curr)

        # Backward propagation from center
        for i in range(mid - 1, -1, -1):
            slices_curr = [slice(None)] * 4
            slices_prev = [slice(None)] * 4
            slices_curr[axis] = i
            slices_prev[axis] = i + 1

            n_curr = n[tuple(slices_curr)]
            n_prev = n[tuple(slices_prev)]

            dot = np.sum(n_curr * n_prev, axis=-1, keepdims=True)
            n[tuple(slices_curr)] = np.where(dot < 0, -n_curr, n_curr)

    return DirectorField.from_array(n, metadata={'sign_method': 'multi_axis_propagation'})


def combined_optimization(director: DirectorField,
                          verbose: bool = False) -> Tuple[DirectorField, Dict]:
    """
    Combined approach: chain propagation + local refinement.

    This is the recommended method for most cases:
    1. Chain propagation establishes global consistency
    2. Local flipping removes remaining errors

    Args:
        director: Input director field.
        verbose: Print progress.

    Returns:
        Tuple of (optimized DirectorField, info dict).
    """
    if verbose:
        print("=" * 60)
        print("Combined Sign Optimization")
        print("=" * 60)
        print("\nPhase 1: Chain propagation...")

    director_chain = chain_propagation(director)

    if verbose:
        n = director_chain.to_array()
        energy_chain = _compute_gradient_energy(n)
        print(f"  After chain propagation: energy = {energy_chain:.2f}")
        print("\nPhase 2: Local flip refinement...")

    director_opt, info = iterative_local_flip(director_chain, max_iter=50, verbose=verbose)

    info['method'] = 'combined'
    director_opt.metadata['sign_method'] = 'combined_optimization'

    return director_opt, info


def _compute_gradient_energy(n: np.ndarray) -> float:
    """Compute total gradient energy: Σ |n_i - n_j|² over neighbors."""
    energy = 0.0
    for axis in range(3):
        n_shifted = np.roll(n, -1, axis=axis)
        energy += np.sum((n - n_shifted) ** 2)
    return float(energy)


def gradient_energy(director: DirectorField) -> float:
    """
    Compute the gradient energy of a director field.

    E = Σ |n_i - n_j|² over all neighbor pairs

    Lower energy indicates more consistent orientations.

    Args:
        director: Director field to evaluate.

    Returns:
        Total gradient energy.
    """
    return _compute_gradient_energy(director.to_array())
