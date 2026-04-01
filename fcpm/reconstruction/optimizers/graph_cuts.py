"""
Graph Cuts Optimization for Sign Consistency.

Solves the sign optimization problem via min-cut/max-flow on a graph
where edge weights encode director alignment.

Mathematical Basis:
------------------
The sign optimization problem is a binary labeling problem on graph G = (V, E):
- Vertices V: voxels with binary states s_i in {+1, -1}
- Edges E: neighbor pairs with weights from director alignment

For the graph-cut energy to be submodular (required for exact s-t min-cut),
all neighbor pairs must have positive alignment (n_i . n_j > 0).  A BFS
pre-alignment pass from the seed voxel ensures this by flipping signs to
establish local consistency before the graph is built.

The graph-cut energy:
    E_GC(S) = sum_{(i,j)} 2|n_i . n_j| * [s_i != s_j]

For unit vectors with positive couplings, minimising E_GC over sign
assignments is equivalent to minimising the gradient energy
sum |s_i n_i - s_j n_j|^2.

The solution is optimal subject to the seed-voxel constraint.  A post-
optimisation global-flip check accounts for the nematic n -> -n symmetry.

Algorithm: Boykov-Kolmogorov (2004)
Complexity: O(V * E * log(V)) ~ O(N * log(N)) for N voxels

References:
-----------
[1] Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of
    min-cut/max-flow algorithms for energy minimization in vision.
[2] Kolmogorov, V., & Zabih, R. (2004). What energy functions can be
    minimized via graph cuts?
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ...core.director import DirectorField, DTYPE
from ..base import OptimizationResult, SignOptimizer, compute_gradient_energy


@dataclass
class GraphCutsConfig:
    """Configuration for graph cuts optimization."""

    unary_weight: float = 0.0
    use_26_connectivity: bool = False
    seed: Optional[Tuple[int, int, int]] = None
    edge_scale: float = 1.0


def _bfs_pre_align(n: np.ndarray, seed: Tuple[int, int, int]) -> np.ndarray:
    """Pre-align director signs via BFS from the seed voxel.

    Ensures that for most neighbor pairs, n_i . n_j > 0, which is
    required for the graph-cut submodularity condition.

    Args:
        n: Director array of shape (ny, nx, nz, 3), modified in-place.
        seed: Starting voxel (y, x, z).

    Returns:
        The pre-aligned array (same object as input).
    """
    ny, nx, nz = n.shape[:3]
    visited = np.zeros((ny, nx, nz), dtype=bool)
    visited[seed] = True

    queue = deque([seed])
    neighbors_6 = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    while queue:
        y, x, z = queue.popleft()
        n_i = n[y, x, z]

        for dy, dx, dz in neighbors_6:
            ny_idx, nx_idx, nz_idx = y + dy, x + dx, z + dz
            if not (0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz):
                continue
            if visited[ny_idx, nx_idx, nz_idx]:
                continue

            visited[ny_idx, nx_idx, nz_idx] = True

            if np.dot(n_i, n[ny_idx, nx_idx, nz_idx]) < 0:
                n[ny_idx, nx_idx, nz_idx] = -n[ny_idx, nx_idx, nz_idx]

            queue.append((ny_idx, nx_idx, nz_idx))

    return n


class GraphCutsOptimizer(SignOptimizer):
    """
    Graph cuts optimizer using min-cut/max-flow.

    Finds the sign assignment that minimises the gradient energy
    ``sum |n_i - n_j|^2`` by:

    1. Pre-aligning signs via BFS from the seed (ensures submodularity)
    2. Building a graph where edge weights = 2|n_i . n_j|
    3. Finding the min-cut via max-flow
    4. A global-flip check for nematic n -> -n symmetry

    Example::

        optimizer = GraphCutsOptimizer()
        result = optimizer.optimize(director, verbose=True)
    """

    def __init__(self, config: Optional[GraphCutsConfig] = None):
        self.config = config or GraphCutsConfig()
        self._has_maxflow = self._check_maxflow_available()

    @staticmethod
    def _check_maxflow_available() -> bool:
        try:
            import maxflow  # noqa: F401
            return True
        except ImportError:
            return False

    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
    ) -> OptimizationResult:
        if self._has_maxflow:
            return self._optimize_maxflow(director, verbose)
        else:
            if verbose:
                print("PyMaxflow not available. Using NetworkX fallback (slower).")
            return self._optimize_networkx(director, verbose)

    # -----------------------------------------------------------------
    # PyMaxflow implementation
    # -----------------------------------------------------------------
    def _optimize_maxflow(self, director: DirectorField, verbose: bool) -> OptimizationResult:
        import maxflow

        n = director.to_array().astype(DTYPE).copy()
        ny, nx, nz = n.shape[:3]
        n_voxels = ny * nx * nz

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Graph Cuts Sign Optimization (PyMaxflow)")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")

        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx // 2, nz // 2)

        # Step 1: BFS pre-alignment to ensure submodularity
        if verbose:
            print("Pre-aligning via BFS...")
        _bfs_pre_align(n, seed)

        energy_after_bfs = compute_gradient_energy(n)
        if verbose:
            print(f"  After BFS pre-align: energy = {energy_after_bfs:.2f}")
            print(f"Building graph with {n_voxels} nodes...")

        # Step 2: Build graph on the pre-aligned field
        g = maxflow.Graph[float](n_voxels, n_voxels * 6)
        nodes = g.add_nodes(n_voxels)

        def idx(y, x, z):
            return y * nx * nz + x * nz + z

        seed_idx = idx(*seed)
        INF = 1e10
        g.add_tedge(seed_idx, INF, 0)

        edge_count = 0

        neighbors_6 = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

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

                        if dy < 0 or (dy == 0 and dx < 0) or (dy == 0 and dx == 0 and dz < 0):
                            continue

                        if not (0 <= ny_idx < ny and 0 <= nx_idx < nx and 0 <= nz_idx < nz):
                            continue

                        j = idx(ny_idx, nx_idx, nz_idx)
                        n_j = n[ny_idx, nx_idx, nz_idx]

                        dot = np.dot(n_i, n_j)
                        # After BFS pre-alignment, most dots are positive.
                        # Use abs(dot) to handle any remaining negative edges.
                        weight = 2.0 * abs(dot) * self.config.edge_scale

                        if weight > 1e-6:
                            g.add_edge(i, j, weight, weight)
                            edge_count += 1

        if verbose:
            print(f"Graph: {n_voxels} nodes, {edge_count} edges")
            print("Running max-flow...")

        flow = g.maxflow()

        if verbose:
            print(f"Max flow: {flow:.2f}")

        # Step 3: Apply sign assignments from the partition
        total_flips = 0
        for y in range(ny):
            for x in range(nx):
                for z in range(nz):
                    i = idx(y, x, z)
                    if g.get_segment(i) == 1:
                        n[y, x, z] = -n[y, x, z]
                        total_flips += 1

        final_energy = compute_gradient_energy(n)

        # Step 4: Global sign check (nematic symmetry n -> -n)
        flipped_energy = compute_gradient_energy(-n)
        if flipped_energy < final_energy:
            n = -n
            final_energy = flipped_energy
            total_flips = n_voxels - total_flips

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f} "
                  f"({100*(initial_energy - final_energy)/initial_energy:.1f}%)")
            print(f"Total sign flips: {total_flips}")

        return OptimizationResult(
            director=DirectorField.from_array(n, metadata={'sign_method': 'graph_cuts_maxflow'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_flips=total_flips,
            method='graph_cuts',
        )

    # -----------------------------------------------------------------
    # NetworkX fallback
    # -----------------------------------------------------------------
    def _optimize_networkx(self, director: DirectorField, verbose: bool) -> OptimizationResult:
        try:
            import networkx as nx_graph
        except ImportError:
            raise ImportError(
                "Neither PyMaxflow nor NetworkX available. "
                "Install one: pip install PyMaxflow or pip install networkx"
            )

        n = director.to_array().astype(DTYPE).copy()
        ny, nx_dim, nz = n.shape[:3]
        n_voxels = ny * nx_dim * nz

        initial_energy = compute_gradient_energy(n)

        if verbose:
            print("=" * 60)
            print("Graph Cuts Sign Optimization (NetworkX fallback)")
            print("=" * 60)
            print(f"Shape: {director.shape}")
            print(f"Initial energy: {initial_energy:.2f}")

        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx_dim // 2, nz // 2)

        # Step 1: BFS pre-alignment
        if verbose:
            print("Pre-aligning via BFS...")
        _bfs_pre_align(n, seed)

        if verbose:
            print(f"Building graph with {n_voxels} nodes...")

        G = nx_graph.DiGraph()

        def idx(y, x, z):
            return y * nx_dim * nz + x * nz + z

        source = 'S'
        sink = 'T'
        G.add_node(source)
        G.add_node(sink)

        for i in range(n_voxels):
            G.add_node(i)

        seed_idx = idx(*seed)

        # Source -> seed with infinite capacity (fixes seed's sign)
        G.add_edge(source, seed_idx, capacity=1e10)

        # All non-seed voxels -> sink with small capacity
        for i in range(n_voxels):
            if i != seed_idx:
                G.add_edge(i, sink, capacity=1e-6)

        neighbors_6 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

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
                        weight = 2.0 * abs(dot) * self.config.edge_scale

                        if weight > 1e-6:
                            G.add_edge(i, j, capacity=weight)
                            G.add_edge(j, i, capacity=weight)

        if verbose:
            print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print("Running min-cut...")

        cut_value, partition = nx_graph.minimum_cut(G, source, sink)
        reachable_from_source, reachable_from_sink = partition

        if verbose:
            print(f"Min cut value: {cut_value:.2f}")

        total_flips = 0
        for y in range(ny):
            for x in range(nx_dim):
                for z in range(nz):
                    i = idx(y, x, z)
                    if i in reachable_from_sink:
                        n[y, x, z] = -n[y, x, z]
                        total_flips += 1

        final_energy = compute_gradient_energy(n)

        # Global sign check (nematic symmetry n -> -n)
        flipped_energy = compute_gradient_energy(-n)
        if flipped_energy < final_energy:
            n = -n
            final_energy = flipped_energy
            total_flips = n_voxels - total_flips

        if verbose:
            print(f"\nFinal energy: {final_energy:.2f}")
            print(f"Energy reduction: {initial_energy - final_energy:.2f}")
            print(f"Total sign flips: {total_flips}")

        return OptimizationResult(
            director=DirectorField.from_array(n, metadata={'sign_method': 'graph_cuts_networkx'}),
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_flips=total_flips,
            method='graph_cuts_networkx',
        )


def graph_cuts_optimization(
    director: DirectorField,
    verbose: bool = False,
    config: Optional[GraphCutsConfig] = None,
) -> OptimizationResult:
    """Functional interface for graph cuts optimization."""
    optimizer = GraphCutsOptimizer(config)
    return optimizer.optimize(director, verbose=verbose)


__all__ = [
    "GraphCutsConfig",
    "GraphCutsOptimizer",
    "graph_cuts_optimization",
]
