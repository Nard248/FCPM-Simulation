# Advanced Sign Optimization Approaches

This document describes the mathematical foundations and implementation details of the advanced sign optimization methods for FCPM director reconstruction.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Approach 1: Graph Cuts](#approach-1-graph-cuts)
4. [Approach 2: Simulated Annealing](#approach-2-simulated-annealing)
5. [Approach 3: Hierarchical Optimization](#approach-3-hierarchical-optimization)
6. [Approach 4: Belief Propagation](#approach-4-belief-propagation)
7. [Comparison Summary](#comparison-summary)
8. [Usage Guide](#usage-guide)

---

## Problem Statement

After Q-tensor eigendecomposition, each voxel has a director **n̂** with an arbitrary sign. Due to the nematic symmetry (**n ≡ -n**), both orientations are physically equivalent. However, for a smooth, physically realistic field, we need consistent sign assignments.

**Goal**: Find the optimal sign assignment **S*** that minimizes the gradient energy:

```
S* = argmin E(S) = argmin Σ_{(i,j)∈neighbors} |s_i·n̂_i - s_j·n̂_j|²
```

where `s_i ∈ {+1, -1}` for all voxels `i`.

---

## Mathematical Foundation

### Ising Model Equivalence

The sign optimization problem is equivalent to an **Ising model** on a 3D lattice:

```
E(S) = Σ_{(i,j)} J_ij · s_i · s_j + const
```

where the coupling strength is:

```
J_ij = -2(n̂_i · n̂_j)
```

**Key insight**:
- When `J_ij > 0` (directors are parallel): prefer same signs
- When `J_ij < 0` (directors are anti-parallel): prefer opposite signs

### Energy Expansion

For unit vectors, the energy can be expanded as:

```
|s_i·n̂_i - s_j·n̂_j|² = 2 - 2·s_i·s_j·(n̂_i · n̂_j)
```

This shows the direct connection to the Ising model with:
- Ferromagnetic coupling when directors are aligned
- Antiferromagnetic coupling when anti-aligned

---

## Approach 1: Graph Cuts

**File**: `approaches/graph_cuts.py`

### Theory

Graph cuts solve binary labeling problems exactly when the energy function is **submodular**. Our sign optimization energy is submodular for smooth fields.

The algorithm constructs a graph G = (V, E):
- **Vertices V**: All voxels plus source (S) and sink (T)
- **Source S**: Represents positive sign (+1)
- **Sink T**: Represents negative sign (-1)
- **Edges**: Weighted by director alignment

### Graph Construction

For each voxel v:
- Edge S→v: Cost of assigning `s_v = -1`
- Edge v→T: Cost of assigning `s_v = +1`

For each neighbor pair (u, v):
- Edge u↔v: Penalty for disagreement, weight = `2|n̂_u · n̂_v|`

### Algorithm

1. Build graph from director field
2. Run Boykov-Kolmogorov max-flow algorithm
3. Find min-cut (separates S-reachable from T-reachable)
4. Assign signs based on partition

### Complexity

**O(V·E·log(V)) ≈ O(N·log(N))** for N voxels

### Pros/Cons

| Pros | Cons |
|------|------|
| Globally optimal | Requires submodularity |
| Fast in practice | May fail at sharp discontinuities |
| Deterministic | Needs PyMaxflow or NetworkX |

### Usage

```python
from v2.approaches import GraphCutsOptimizer

optimizer = GraphCutsOptimizer()
result = optimizer.optimize(director, verbose=True)
optimized_director = result.director
```

---

## Approach 2: Simulated Annealing

**File**: `approaches/simulated_annealing.py`

### Theory

Simulated annealing samples from the Boltzmann distribution:

```
P(S) ∝ exp(-E(S) / T)
```

At high temperature T: explores broadly (accepts many moves)
At low temperature T: exploits (only accepts improving moves)

### Metropolis-Hastings Algorithm

1. Initialize with layer propagation
2. For each iteration:
   - Randomly select voxel
   - Propose sign flip
   - Compute ΔE = E_new - E_old
   - Accept with probability `min(1, exp(-ΔE/T))`
3. Decrease temperature: `T ← α·T` (typical α = 0.995)
4. Continue until frozen (`T < T_min`)

### Enhancements

1. **Cluster Moves (Wolff Algorithm)**:
   - Build clusters of aligned spins
   - Flip entire cluster at once
   - Dramatically improves mixing near critical temperature

2. **Adaptive Temperature**:
   - Monitor acceptance rate
   - Speed up cooling if accepting too many moves
   - Slow down if rejecting too many

### Energy Change Calculation

Local energy change for flipping spin at voxel i:

```
ΔE = 4 · Σ_{j∈neighbors(i)} (n̂_i · n̂_j)
```

This is O(1) per flip (only 6 neighbors).

### Complexity

**O(iterations × N)** ≈ O(10⁵ × N) for typical settings

### Pros/Cons

| Pros | Cons |
|------|------|
| Can escape local minima | Slow |
| Asymptotically optimal | Stochastic (varies between runs) |
| Works with any energy | Requires tuning |

### Usage

```python
from v2.approaches import SimulatedAnnealingOptimizer, SimulatedAnnealingConfig

config = SimulatedAnnealingConfig(
    max_iterations=50000,
    use_cluster_moves=True,
    use_adaptive=True
)
optimizer = SimulatedAnnealingOptimizer(config)
result = optimizer.optimize(director, verbose=True)
```

---

## Approach 3: Hierarchical Optimization

**File**: `approaches/hierarchical.py`

### Theory

Multi-scale optimization captures global structure at coarse levels and refines locally at fine levels.

### Pyramid Construction

Build a pyramid of director fields:
- Level 0: Coarsest (e.g., 6×6×3)
- Level L: Original resolution

Coarsening uses 2×2×2 blocks → single representative director

### Coarsening Strategy

For each 2×2×2 block:
1. Compute average Q-tensor: `Q_avg = (1/8) Σ Q_i`
2. Find dominant eigenvector of Q_avg
3. Normalize to get representative director

This preserves orientation information while smoothing noise.

### Algorithm

1. Build pyramid (coarsest to finest)
2. Optimize coarsest level (small problem, exact or near-exact)
3. For each finer level:
   - Upsample signs from coarser level
   - Refine with local iterative optimization
4. Return finest level result

### Complexity

**O(N + N/8 + N/64 + ...) = O(N)** - linear!

### Pros/Cons

| Pros | Cons |
|------|------|
| Very fast | May miss small-scale features |
| Global consistency | Coarsening assumes smoothness |
| Linear complexity | Less accurate at defects |

### Usage

```python
from v2.approaches import HierarchicalOptimizer, HierarchicalConfig

config = HierarchicalConfig(
    coarsen_factor=2,
    min_dimension=4,
    refine_iterations=5
)
optimizer = HierarchicalOptimizer(config)
result = optimizer.optimize(director, verbose=True)
```

---

## Approach 4: Belief Propagation

**File**: `approaches/belief_propagation.py`

### Theory

Treats sign optimization as probabilistic inference on a factor graph:
- **Variable nodes**: Signs `s_i ∈ {+1, -1}`
- **Factor nodes**: Pairwise potentials `ψ_{ij}(s_i, s_j)`

### Pairwise Potential

```
ψ_{ij}(s_i, s_j) = exp(-β · |s_i·n̂_i - s_j·n̂_j|²)
```

where β is inverse temperature (higher = more confident).

### Message Passing

Messages encode beliefs about neighbor states:

```
m_{u→v}(s_v) = Σ_{s_u} ψ(s_u, s_v) · Π_{w∈N(u)\v} m_{w→u}(s_u)
```

### Belief Computation

Belief at node v:

```
b_v(s_v) ∝ Π_{u∈N(v)} m_{u→v}(s_v)
```

### MAP Estimate

```
s*_v = argmax_{s_v} b_v(s_v)
```

### Damping for Convergence

Loopy BP on 3D grids may oscillate. Damping stabilizes convergence:

```
m_new ← α·m_old + (1-α)·m_computed
```

Typical α = 0.5 works well.

### Complexity

**O(iterations × edges)** = O(I × 6N)

### Pros/Cons

| Pros | Cons |
|------|------|
| Good approximation | May not converge |
| Parallelizable | Approximate (not exact) |
| Works on loopy graphs | Sensitive to β parameter |

### Usage

```python
from v2.approaches import BeliefPropagationOptimizer, BeliefPropagationConfig

config = BeliefPropagationConfig(
    max_iterations=100,
    damping=0.5,
    beta=2.0
)
optimizer = BeliefPropagationOptimizer(config)
result = optimizer.optimize(director, verbose=True)
```

---

## Comparison Summary

### Expected Performance

| Approach | Quality | Speed | Complexity |
|----------|---------|-------|------------|
| Graph Cuts | Best (exact) | Fast | O(N·log N) |
| Simulated Annealing | Very Good | Slow | O(iterations × N) |
| Hierarchical | Good | Very Fast | O(N) |
| Belief Propagation | Good | Medium | O(iterations × N) |
| V2 Layer+Refine | Good | Fast | O(N) |

### Recommendation by Use Case

| Scenario | Recommended Approach |
|----------|---------------------|
| General use, smooth fields | Graph Cuts |
| Large volumes (>128³) | Hierarchical |
| High noise (>10%) | Simulated Annealing |
| Fields with defects | Simulated Annealing or V2 |
| Quick preview | Hierarchical or V2 |
| Validation | Compare multiple methods |

---

## Usage Guide

### Quick Start

```python
import sys
sys.path.insert(0, 'path/to/v2')

from approaches import (
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer
)

# Load your director field
# director_raw = ...

# Choose an optimizer
optimizer = GraphCutsOptimizer()

# Run optimization
result = optimizer.optimize(director_raw, verbose=True)

# Get optimized director
optimized_director = result.director
print(f"Energy reduced from {result.initial_energy:.1f} to {result.final_energy:.1f}")
```

### Running Benchmarks

```bash
# Run benchmark script
python v2/benchmark_approaches.py

# Or use the Jupyter notebook
jupyter notebook v2/benchmark_notebook.ipynb
```

### Dependencies

- **Required**: numpy, scipy
- **Optional**:
  - PyMaxflow (for fast graph cuts): `pip install PyMaxflow`
  - NetworkX (fallback for graph cuts): `pip install networkx`
  - Matplotlib (for visualization): `pip install matplotlib`

---

## References

1. Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision.

2. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing.

3. Wolff, U. (1989). Collective Monte Carlo updating for spin systems.

4. Pearl, J. (1988). Probabilistic reasoning in intelligent systems.

5. Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). Understanding belief propagation and its generalizations.

6. Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief propagation for early vision.
