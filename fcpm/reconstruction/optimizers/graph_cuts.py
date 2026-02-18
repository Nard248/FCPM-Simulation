"""
Graph Cuts Optimization for Sign Consistency.

Solves the sign optimization problem exactly via min-cut/max-flow.

Mathematical Basis:
------------------
The sign optimization problem is a binary labeling problem on graph G = (V, E):
- Vertices V: voxels with binary states s_i in {+1, -1}
- Edges E: neighbor pairs with weights from director alignment

Energy function:
    E(S) = sum_{(i,j)} w_ij * [s_i != s_j]  (pairwise disagreement cost)

where w_ij = |n_i . n_j| encodes alignment preference.

This energy is **submodular** for smooth fields, so the graph-cut solution
is globally optimal.

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


class GraphCutsOptimizer(SignOptimizer):
    """
    Graph cuts optimizer using min-cut/max-flow.

    Provides the globally optimal solution for the sign optimization
    problem when the energy function is submodular (smooth director fields).

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

        g = maxflow.Graph[float](n_voxels, n_voxels * 6)
        nodes = g.add_nodes(n_voxels)

        def idx(y, x, z):
            return y * nx * nz + x * nz + z

        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx // 2, nz // 2)

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
                        weight = 2.0 * abs(dot) * self.config.edge_scale

                        if weight > 1e-6:
                            g.add_edge(i, j, weight, weight)
                            edge_count += 1

                        if dot < 0:
                            n[ny_idx, nx_idx, nz_idx] = -n[ny_idx, nx_idx, nz_idx]

        if verbose:
            print(f"Graph: {n_voxels} nodes, {edge_count} edges")
            print("Running max-flow...")

        flow = g.maxflow()

        if verbose:
            print(f"Max flow: {flow:.2f}")

        total_flips = 0
        for y in range(ny):
            for x in range(nx):
                for z in range(nz):
                    i = idx(y, x, z)
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

        seed = self.config.seed
        if seed is None:
            seed = (ny // 2, nx_dim // 2, nz // 2)
        seed_idx = idx(*seed)
        G.add_edge(source, seed_idx, capacity=1e10)

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

                        if dot < 0:
                            n[ny_idx, nx_idx, nz_idx] = -n[ny_idx, nx_idx, nz_idx]
                            dot = -dot

                        weight = 2.0 * dot * self.config.edge_scale

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
