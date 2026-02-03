"""
Sign Optimization V2 - Layer-by-Layer Energy Minimization.

Physics-informed approach based on Frank elastic energy minimization.

Key Innovation:
- Instead of global BFS or iterative flipping, we process layer-by-layer
- For each voxel, we choose the sign that minimizes local gradient energy
- Considers ALL already-determined neighbors (previous layer + in-plane)

The Frank elastic energy (one-constant approximation):
    E = (K/2) * ∫|∇n|² dV ≈ (K/2) * Σ|n_i - n_j|²

Algorithm:
1. Start at z=0, assume all positive signs (arbitrary reference)
2. For z=1,2,...,nz-1:
   a. For each voxel (y,x) in layer z:
      - Compute energy E+ with current sign (+)
      - Compute energy E- with flipped sign (-)
      - Energy considers neighbors in z-1 and already-processed (y,x) in current z
   b. Choose sign that minimizes energy
3. Optional: Refine with in-plane passes within each layer

This is a greedy layer-wise optimization that respects the physical
smoothness constraint while avoiding global oscillation issues.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fcpm.core.director import DirectorField, DTYPE


@dataclass
class OptimizationResult:
    """Results from V2 sign optimization."""
    director: DirectorField
    initial_energy: float
    final_energy: float
    energy_by_layer: List[float]
    flips_by_layer: List[int]
    total_flips: int
    method: str


def compute_gradient_energy(n: np.ndarray) -> float:
    """
    Compute total Frank elastic energy (squared gradient approximation).

    E = Σ |n_i - n_j|² over all neighbor pairs

    Lower energy = smoother, more physically realistic field.
    """
    energy = 0.0
    for axis in range(3):
        n_shifted = np.roll(n, -1, axis=axis)
        energy += np.sum((n - n_shifted) ** 2)
    return float(energy)


def compute_local_energy(n: np.ndarray, y: int, x: int, z: int,
                         sign: int = 1) -> float:
    """
    Compute local energy contribution for a single voxel.

    Considers 6 neighbors (if they exist within bounds).

    Args:
        n: Director array (ny, nx, nz, 3)
        y, x, z: Voxel coordinates
        sign: +1 or -1 for the voxel's orientation

    Returns:
        Local energy contribution
    """
    ny_dim, nx_dim, nz_dim = n.shape[:3]
    n_voxel = sign * n[y, x, z]

    energy = 0.0

    # 6-connected neighbors
    neighbors = [
        (y-1, x, z), (y+1, x, z),  # y neighbors
        (y, x-1, z), (y, x+1, z),  # x neighbors
        (y, x, z-1), (y, x, z+1),  # z neighbors
    ]

    for ny_idx, nx_idx, nz_idx in neighbors:
        # Skip out-of-bounds
        if not (0 <= ny_idx < ny_dim and
                0 <= nx_idx < nx_dim and
                0 <= nz_idx < nz_dim):
            continue

        n_neighbor = n[ny_idx, nx_idx, nz_idx]
        energy += np.sum((n_voxel - n_neighbor) ** 2)

    return energy


def layer_by_layer_optimization(director: DirectorField,
                                 start_layer: int = 0,
                                 direction: str = 'forward',
                                 verbose: bool = False) -> OptimizationResult:
    """
    Layer-by-layer sign optimization using local energy minimization.

    Algorithm:
    1. Fix first layer with positive signs (reference)
    2. For each subsequent layer:
       - Process voxels in raster order (row by row)
       - For each voxel, compute energy with + and - signs
       - Choose sign that minimizes energy considering:
         * Neighbors in previous layer (already determined)
         * Already-processed neighbors in current layer

    Args:
        director: Input director field with arbitrary signs
        start_layer: Which z-layer to start from (default: 0)
        direction: 'forward' (z+) or 'backward' (z-)
        verbose: Print progress

    Returns:
        OptimizationResult with optimized director and statistics
    """
    n = director.to_array().astype(DTYPE).copy()
    ny_dim, nx_dim, nz_dim = n.shape[:3]

    # Track statistics
    energy_by_layer = []
    flips_by_layer = []
    total_flips = 0

    initial_energy = compute_gradient_energy(n)

    if verbose:
        print("=" * 60)
        print("V2 Layer-by-Layer Sign Optimization")
        print("=" * 60)
        print(f"Shape: {director.shape}")
        print(f"Initial energy: {initial_energy:.2f}")
        print(f"Direction: {direction}")
        print()

    # Determine layer order
    if direction == 'forward':
        layer_range = range(start_layer + 1, nz_dim)
    else:
        layer_range = range(start_layer - 1, -1, -1)

    # Process layer by layer
    for z in layer_range:
        layer_flips = 0

        # Process each voxel in the layer
        for y in range(ny_dim):
            for x in range(nx_dim):
                # Compute energy with current sign (+1)
                energy_plus = compute_local_energy(n, y, x, z, sign=+1)

                # Compute energy with flipped sign (-1)
                energy_minus = compute_local_energy(n, y, x, z, sign=-1)

                # Choose sign that minimizes energy
                if energy_minus < energy_plus:
                    n[y, x, z] = -n[y, x, z]
                    layer_flips += 1

        # Record statistics
        layer_energy = compute_gradient_energy(n)
        energy_by_layer.append(layer_energy)
        flips_by_layer.append(layer_flips)
        total_flips += layer_flips

        if verbose and (z < 5 or z % 10 == 0 or z == nz_dim - 1):
            print(f"  Layer {z:3d}: flipped {layer_flips:5d} voxels, "
                  f"energy = {layer_energy:.2f}")

    final_energy = compute_gradient_energy(n)

    if verbose:
        print()
        print(f"Final energy: {final_energy:.2f}")
        print(f"Energy reduction: {initial_energy - final_energy:.2f} "
              f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")
        print(f"Total flips: {total_flips}")

    result = OptimizationResult(
        director=DirectorField.from_array(n, metadata={'sign_method': 'layer_by_layer_v2'}),
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_by_layer=energy_by_layer,
        flips_by_layer=flips_by_layer,
        total_flips=total_flips,
        method='layer_by_layer'
    )

    return result


def bidirectional_layer_optimization(director: DirectorField,
                                      verbose: bool = False) -> OptimizationResult:
    """
    Bidirectional layer optimization: forward then backward pass.

    This helps resolve inconsistencies by propagating information
    in both directions through the volume.

    Args:
        director: Input director field
        verbose: Print progress

    Returns:
        OptimizationResult with optimized director
    """
    if verbose:
        print("=" * 60)
        print("V2 Bidirectional Layer Optimization")
        print("=" * 60)
        print("\nPass 1: Forward (z=0 → z=max)")

    # Forward pass
    result1 = layer_by_layer_optimization(
        director, start_layer=0, direction='forward', verbose=verbose
    )

    if verbose:
        print("\nPass 2: Backward (z=max → z=0)")

    # Backward pass starting from last layer
    nz = director.shape[2]
    result2 = layer_by_layer_optimization(
        result1.director, start_layer=nz-1, direction='backward', verbose=verbose
    )

    # Combine statistics
    result2.method = 'bidirectional'
    result2.initial_energy = result1.initial_energy
    result2.total_flips = result1.total_flips + result2.total_flips

    return result2


def in_plane_refinement(director: DirectorField,
                        max_iter: int = 10,
                        verbose: bool = False) -> OptimizationResult:
    """
    Refine signs within each layer independently.

    After layer-by-layer optimization, this does additional passes
    within each xy-plane to catch any remaining inconsistencies.

    Args:
        director: Input director field
        max_iter: Max iterations per layer
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    n = director.to_array().astype(DTYPE).copy()
    ny_dim, nx_dim, nz_dim = n.shape[:3]

    initial_energy = compute_gradient_energy(n)
    total_flips = 0

    if verbose:
        print("=" * 60)
        print("V2 In-Plane Refinement")
        print("=" * 60)
        print(f"Initial energy: {initial_energy:.2f}")
        print()

    # Process each layer
    for z in range(nz_dim):
        for iteration in range(max_iter):
            layer_flips = 0

            for y in range(ny_dim):
                for x in range(nx_dim):
                    # Only consider in-plane + z neighbors
                    energy_plus = compute_local_energy(n, y, x, z, sign=+1)
                    energy_minus = compute_local_energy(n, y, x, z, sign=-1)

                    if energy_minus < energy_plus:
                        n[y, x, z] = -n[y, x, z]
                        layer_flips += 1

            if layer_flips == 0:
                break

            total_flips += layer_flips

        if verbose and (z < 3 or z % 10 == 0 or z == nz_dim - 1):
            energy = compute_gradient_energy(n)
            print(f"  Layer {z:3d}: energy = {energy:.2f}")

    final_energy = compute_gradient_energy(n)

    if verbose:
        print(f"\nFinal energy: {final_energy:.2f}")
        print(f"Total flips: {total_flips}")

    return OptimizationResult(
        director=DirectorField.from_array(n, metadata={'sign_method': 'in_plane_v2'}),
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_by_layer=[],
        flips_by_layer=[],
        total_flips=total_flips,
        method='in_plane_refinement'
    )


def combined_v2_optimization(director: DirectorField,
                              verbose: bool = False) -> OptimizationResult:
    """
    Full V2 optimization pipeline - NOW USES layer_then_refine.

    This method is conceptually cleaner and achieves comparable results to V1:
    1. Layer-by-layer propagation (establishes z-continuity)
    2. Iterative refinement (global smoothing)

    Args:
        director: Input director field
        verbose: Print progress

    Returns:
        OptimizationResult with fully optimized director
    """
    # Use the new layer_then_refine which is cleaner and effective
    return layer_then_refine(director, max_refine_iter=50, verbose=verbose)


# Vectorized version for better performance
def layer_by_layer_vectorized(director: DirectorField,
                               verbose: bool = False) -> OptimizationResult:
    """
    Vectorized layer-by-layer optimization.

    Key insight: Only use the PREVIOUS layer (z-1) as reference when
    determining signs for layer z. This is the correct interpretation
    of layer-by-layer propagation.

    Args:
        director: Input director field
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    n = director.to_array().astype(DTYPE).copy()
    ny_dim, nx_dim, nz_dim = n.shape[:3]

    initial_energy = compute_gradient_energy(n)
    flips_by_layer = []
    energy_by_layer = []
    total_flips = 0

    if verbose:
        print("=" * 60)
        print("V2 Vectorized Layer-by-Layer (z-1 reference only)")
        print("=" * 60)
        print(f"Initial energy: {initial_energy:.2f}")
        print()

    # Process layer by layer - ONLY using previous layer as reference
    for z in range(1, nz_dim):
        # Get current layer and previous layer
        n_curr = n[:, :, z]      # Shape: (ny, nx, 3)
        n_prev = n[:, :, z-1]    # Shape: (ny, nx, 3) - FIXED reference

        # Energy with current sign: |n_curr - n_prev|²
        energy_plus = np.sum((n_curr - n_prev) ** 2, axis=-1)

        # Energy with flipped sign: |-n_curr - n_prev|² = |n_curr + n_prev|²
        energy_minus = np.sum((-n_curr - n_prev) ** 2, axis=-1)

        # Flip where minus sign has lower energy
        flip_mask = energy_minus < energy_plus
        n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]

        layer_flips = np.sum(flip_mask)
        flips_by_layer.append(int(layer_flips))
        total_flips += layer_flips

        layer_energy = compute_gradient_energy(n)
        energy_by_layer.append(layer_energy)

        if verbose and (z < 5 or z % 10 == 0 or z == nz_dim - 1):
            print(f"  Layer {z:3d}: flipped {layer_flips:5d} voxels, "
                  f"energy = {layer_energy:.2f}")

    final_energy = compute_gradient_energy(n)

    if verbose:
        print()
        print(f"Final energy: {final_energy:.2f}")
        print(f"Energy reduction: {initial_energy - final_energy:.2f}")

    return OptimizationResult(
        director=DirectorField.from_array(n, metadata={'sign_method': 'layer_vectorized_v2'}),
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_by_layer=energy_by_layer,
        flips_by_layer=flips_by_layer,
        total_flips=int(total_flips),
        method='layer_vectorized'
    )


def layer_then_refine(director: DirectorField,
                      max_refine_iter: int = 20,
                      verbose: bool = False) -> OptimizationResult:
    """
    V2 RECOMMENDED: Layer propagation followed by iterative refinement.

    This is the key innovation combining:
    1. Layer-by-layer propagation (fast, establishes z-continuity)
    2. Iterative local refinement (catches in-plane inconsistencies)

    This should achieve similar or better results than V1 with a cleaner
    conceptual framework.

    Args:
        director: Input director field
        max_refine_iter: Max refinement iterations
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    n = director.to_array().astype(DTYPE).copy()
    ny_dim, nx_dim, nz_dim = n.shape[:3]

    initial_energy = compute_gradient_energy(n)
    total_flips = 0

    if verbose:
        print("=" * 60)
        print("V2 Layer-Then-Refine Optimization")
        print("=" * 60)
        print(f"Initial energy: {initial_energy:.2f}")

    # Phase 1: Layer-by-layer propagation (z-1 reference only)
    if verbose:
        print("\nPhase 1: Layer propagation (z-1 reference)...")

    for z in range(1, nz_dim):
        n_curr = n[:, :, z]
        n_prev = n[:, :, z-1]

        # Flip to align with previous layer
        dot = np.sum(n_curr * n_prev, axis=-1)
        flip_mask = dot < 0
        n[:, :, z][flip_mask] = -n[:, :, z][flip_mask]
        total_flips += np.sum(flip_mask)

    energy_after_layer = compute_gradient_energy(n)
    if verbose:
        print(f"  After layer propagation: energy = {energy_after_layer:.2f}")

    # Phase 2: Iterative refinement (same as V1's iterative_local_flip)
    if verbose:
        print("\nPhase 2: Iterative refinement...")

    for iteration in range(max_refine_iter):
        # Compute cost for current and flipped orientation (all 6 neighbors)
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
        n_flipped = np.sum(flip_mask)

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
        print()
        print(f"Final energy: {final_energy:.2f}")
        print(f"Total reduction: {initial_energy - final_energy:.2f} "
              f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")

    return OptimizationResult(
        director=DirectorField.from_array(n, metadata={'sign_method': 'layer_then_refine_v2'}),
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_by_layer=[],
        flips_by_layer=[],
        total_flips=int(total_flips),
        method='layer_then_refine'
    )


if __name__ == "__main__":
    # Quick test
    from fcpm import create_cholesteric_director, simulate_fcpm, reconstruct

    print("Creating test director field...")
    director_gt = create_cholesteric_director(shape=(32, 32, 16), pitch=6.0)

    print("Simulating FCPM...")
    I_fcpm = simulate_fcpm(director_gt)

    print("Reconstructing (Q-tensor only, no sign fix)...")
    from fcpm.reconstruction import reconstruct_via_qtensor
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm)

    print("\n" + "=" * 70)
    print("Testing V2 Sign Optimization")
    print("=" * 70)

    # Test V2 optimization
    result = combined_v2_optimization(director_raw, verbose=True)

    print("\n" + "=" * 70)
    print("Comparison with V1")
    print("=" * 70)

    from fcpm.reconstruction import combined_optimization
    director_v1, info_v1 = combined_optimization(director_raw, verbose=True)

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"V1 Final Energy: {info_v1['final_energy']:.2f}")
    print(f"V2 Final Energy: {result.final_energy:.2f}")
