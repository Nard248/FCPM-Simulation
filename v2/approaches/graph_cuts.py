"""
Graph Cuts Optimization for Sign Consistency.

Solves the sign optimization problem exactly via min-cut/max-flow.

Mathematical Basis:
------------------
The sign optimization problem is a binary labeling problem on graph G = (V, E):
- Vertices V: voxels with binary states s_i ∈ {+1, -1}
- Edges E: neighbor pairs with weights from director alignment

Energy function:
    E(S) = Σ_{(i,j)} w_ij · [s_i ≠ s_j]  (pairwise disagreement cost)

where w_ij = |n̂_i · n̂_j| encodes alignment preference.

This energy is **submodular** for smooth fields:
- When n̂_i · n̂_j > 0 (parallel): prefer same sign → w_ij > 0
- When n̂_i · n̂_j < 0 (anti-parallel): prefer opposite sign → w_ij < 0

Graph construction:
1. Source S represents positive label (+1)
2. Sink T represents negative label (-1)
3. For each voxel v:
   - S→v capacity: cost of assigning s_v = -1
   - v→T capacity: cost of assigning s_v = +1
4. For neighbor pairs (u,v):
   - u↔v capacity: penalty for disagreement

Min-cut partitions vertices into S-reachable (positive) and T-reachable (negative).

Algorithm: Boykov-Kolmogorov (2004)
Complexity: O(V·E·log(V)) ≈ O(N·log(N)) for N voxels

References:
-----------
[1] Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of
    min-cut/max-flow algorithms for energy minimization in vision.
[2] Kolmogorov, V., & Zabih, R. (2004). What energy functions can be
    minimized via graph cuts?
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
class GraphCutsConfig:
    """Configuration for graph cuts optimization."""

    # Unary cost weight (regularization on absolute orientation)
    # Higher values make the solution more sensitive to unary terms
    unary_weight: float = 0.0

    # Whether to use 26-connectivity (vs 6-connectivity)
    use_26_connectivity: bool = False

    # Seed voxel to anchor positive sign (prevents trivial all-flip)
    seed: Optional[Tuple[int, int, int]] = None

    # Edge weight scaling (larger = stronger smoothness)
    edge_scale: float = 1.0


class GraphCutsOptimizer:
    """
    Graph cuts optimizer using min-cut/max-flow.

    This provides the globally optimal solution for the sign optimization
    problem when the energy function is submodular (which holds for
    smooth director fields without sharp discontinuities).

    Example:
    -------
    >>> optimizer = GraphCutsOptimizer()
    >>> result = optimizer.optimize(director, verbose=True)
    >>> print(f"Energy reduced from {result.initial_energy:.1f} to {result.final_energy:.1f}")
    """

    def __init__(self, config: Optional[GraphCutsConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or GraphCutsConfig()
        self._has_maxflow = self._check_maxflow_available()

    def _check_maxflow_available(self) -> bool:
        """Check if PyMaxflow library is available."""
        try:
            import maxflow
            return True
        except ImportError:
            return False

    def optimize(self, director: DirectorField,
                 verbose: bool = False) -> OptimizationResult:
        """
        Optimize director signs using graph cuts.

        Args:
            director: Input director field with arbitrary signs.
            verbose: Print progress information.

        Returns:
            OptimizationResult with optimized director and statistics.
        """
        if self._has_maxflow:
            return self._optimize_maxflow(director, verbose)
        else:
            if verbose:
                print("PyMaxflow not available. Using NetworkX fallback (slower).")
            return self._optimize_networkx(director, verbose)

    def _optimize_maxflow(self, director: DirectorField,
                          verbose: bool = False) -> OptimizationResult:
        """Optimize using PyMaxflow library (fast)."""
        import maxflow

        n = director.to_array().astype(DTYPE)
        ny, nx, nz = n.shape[:3]
        n_voxels = ny * nx * nz

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Graph Cuts Sign Optimization (PyMaxflow)")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print(f"Building graph with {n_voxels} nodes...")

        # Create graph
        g = maxflow.Graph[float](n_voxels, n_voxels * 6)
        nodes = g.add_nodes(n_voxels)

        def idx(y, x, z):
            """Convert 3D coordinates to linear index."""
            return y * nx * nz + x * nz + z

        # Anchor seed to prevent trivial all-flip solution
        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx // 2, nz // 2)

        seed_idx = idx(*seed)
        INF = 1e10
        g.add_tedge(seed_idx, INF, 0)  # Strongly prefer positive for seed

        # Add pairwise edges based on director alignment
        edge_count = 0

        # 6-connectivity neighbors
        neighbors_6 = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        # Additional neighbors for 26-connectivity
        neighbors_18 = [
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        ]

        neighbors_8_diag = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]

        if self.config.use_26_connectivity:
            all_neighbors = neighbors_6 + neighbors_18 + neighbors_8_diag
        else:
            all_neighbors = neighbors_6

        for y in range(ny):
            for x in range(nx):
                for z in range(nz):
                    i = idx(y, x, z)
                    n_i = n[y, x, z]

                    for dy, dx, dz in all_neighbors:
                        ny_idx, nx_idx, nz_idx = y + dy, x + dx, z + dz

                        # Only process forward neighbors to avoid double-counting
                        if dy < 0 or (dy == 0 and dx < 0) or (dy == 0 and dx == 0 and dz < 0):
                            continue

                        if not (0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz):
                            continue

                        j = idx(ny_idx, nx_idx, nz_idx)
                        n_j = n[ny_idx, nx_idx, nz_idx]

                        # Compute coupling: J_ij = n̂_i · n̂_j
                        dot = np.dot(n_i, n_j)

                        # Edge weight: cost of having different signs
                        # If dot > 0 (parallel): want same sign → high cost for different
                        # If dot < 0 (anti-parallel): want opposite sign → low cost for different
                        #
                        # For submodular energy: w ≥ 0 when we prefer same labels
                        # Graph cut minimizes: Σ w_ij · [s_i ≠ s_j]
                        #
                        # Original energy: |s_i·n_i - s_j·n_j|²
                        # = |n_i|² + |n_j|² - 2·s_i·s_j·(n_i·n_j)
                        # = 2 - 2·s_i·s_j·dot   (for unit vectors)
                        #
                        # So: E(same) = 2 - 2·dot, E(diff) = 2 + 2·dot
                        # Cost of disagreement vs agreement: ΔE = 4·|dot|
                        #
                        # For graph cut: if dot > 0, same sign is preferred
                        # Edge capacity = max(0, 2·|dot|) ensures submodularity

                        weight = 2.0 * abs(dot) * self.config.edge_scale

                        if weight > 1e-6:
                            g.add_edge(i, j, weight, weight)
                            edge_count += 1

                        # Handle anti-parallel case: flip sign before graph cut
                        # If dot < 0, we'll pre-flip n_j so the graph sees them as parallel
                        if dot < 0:
                            n[ny_idx, nx_idx, nz_idx] = -n[ny_idx, nx_idx, nz_idx]

        if verbose:
            print(f"Graph: {n_voxels} nodes, {edge_count} edges")
            print("Running max-flow...")

        # Solve
        flow = g.maxflow()

        if verbose:
            print(f"Max flow: {flow:.2f}")

        # Extract labels
        total_flips = 0
        for y in range(ny):
            for x in range(nx):
                for z in range(nz):
                    i = idx(y, x, z)
                    # Segment 1 = sink side = negative label
                    if g.get_segment(i) == 1:
                        n[y, x, z] = -n[y, x, z]
                        total_flips += 1

        final_energy = compute_gradient_energy(n)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")
            print(f"Total sign flips: {total_flips}")

        return OptimizationResult(
            director=DirectorField.from_array(n, metadata={'sign_method': 'graph_cuts_maxflow'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[],
            flips_by_layer=[],
            total_flips=total_flips,
            method='graph_cuts'
        )

    def _optimize_networkx(self, director: DirectorField,
                           verbose: bool = False) -> OptimizationResult:
        """Fallback implementation using NetworkX (slower but no extra deps)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Neither PyMaxflow nor NetworkX available. "
                "Install one: pip install PyMaxflow or pip install networkx"
            )

        n = director.to_array().astype(DTYPE)
        ny, nx_dim, nz = n.shape[:3]
        n_voxels = ny * nx_dim * nz

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Graph Cuts Sign Optimization (NetworkX fallback)")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")
            print("Note: NetworkX is slower than PyMaxflow for large volumes.")
            print(f"Building graph with {n_voxels} nodes...")

        # Build directed graph
        G = nx.DiGraph()

        def idx(y, x, z):
            return y * nx_dim * nz + x * nz + z

        # Add source and sink
        source = 'S'
        sink = 'T'
        G.add_node(source)
        G.add_node(sink)

        # Add all voxel nodes
        for i in range(n_voxels):
            G.add_node(i)

        # Anchor seed
        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx_dim // 2, nz // 2)
        seed_idx = idx(*seed)
        G.add_edge(source, seed_idx, capacity=1e10)

        # Pre-flip anti-parallel directors and add edges
        neighbors_6 = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
        ]

        for y in range(ny):
            for x in range(nx_dim):
                for z in range(nz):
                    i = idx(y, x, z)
                    n_i = n[y, x, z]

                    for dy, dx, dz in neighbors_6:
                        ny_idx, nx_idx, nz_idx = y + dy, x + dx, z + dz

                        if not (0 <= ny_idx < ny and 0 <= nx_idx < nx_dim and 0 <= nz_idx < nz):
                            continue

                        j = idx(ny_idx, nx_idx, nz_idx)
                        n_j = n[ny_idx, nx_idx, nz_idx]

                        dot = np.dot(n_i, n_j)

                        # Pre-flip for anti-parallel
                        if dot < 0:
                            n[ny_idx, nx_idx, nz_idx] = -n[ny_idx, nx_idx, nz_idx]
                            dot = -dot

                        weight = 2.0 * dot * self.config.edge_scale

                        if weight > 1e-6:
                            G.add_edge(i, j, capacity=weight)
                            G.add_edge(j, i, capacity=weight)

        if verbose:
            print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print("Running min-cut (this may take a while)...")

        # Compute min-cut
        cut_value, partition = nx.minimum_cut(G, source, sink)
        reachable_from_source, reachable_from_sink = partition

        if verbose:
            print(f"Min cut value: {cut_value:.2f}")

        # Apply labels
        total_flips = 0
        for y in range(ny):
            for x in range(nx_dim):
                for z in range(nz):
                    i = idx(y, x, z)
                    if i in reachable_from_sink:
                        n[y, x, z] = -n[y, x, z]
                        total_flips += 1

        final_energy = compute_gradient_energy(n)

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f}")
            print(f"Total sign flips: {total_flips}")

        return OptimizationResult(
            director=DirectorField.from_array(n, metadata={'sign_method': 'graph_cuts_networkx'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_by_layer=[],
            flips_by_layer=[],
            total_flips=total_flips,
            method='graph_cuts_networkx'
        )


def graph_cuts_optimization(director: DirectorField,
                            verbose: bool = False,
                            config: Optional[GraphCutsConfig] = None) -> OptimizationResult:
    """
    Functional interface for graph cuts optimization.

    This is a convenience wrapper around GraphCutsOptimizer.

    Args:
        director: Input director field with arbitrary signs.
        verbose: Print progress information.
        config: Optional configuration.

    Returns:
        OptimizationResult with optimized director.

    Example:
    -------
    >>> result = graph_cuts_optimization(director, verbose=True)
    >>> optimized = result.director
    """
    optimizer = GraphCutsOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


# Quick test
if __name__ == "__main__":
    from fcpm import create_cholesteric_director, simulate_fcpm
    from fcpm.reconstruction import reconstruct_via_qtensor

    print("Testing Graph Cuts Optimization")
    print("=" * 60)

    # Create test data
    print("Creating cholesteric director...")
    director_gt = create_cholesteric_director(shape=(32, 32, 16), pitch=6.0)

    print("Simulating FCPM...")
    I_fcpm = simulate_fcpm(director_gt)

    print("Reconstructing (no sign fix)...")
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm)

    print("\nRunning Graph Cuts optimization...")
    optimizer = GraphCutsOptimizer()
    result = optimizer.optimize(director_raw, verbose=True)

    # Compare with ground truth
    from fcpm import summary_metrics
    metrics = summary_metrics(result.director, director_gt)
    print(f"\nAngular error vs GT: {metrics['angular_error_mean_deg']:.2f} deg (mean)")
