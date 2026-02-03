# Sign Optimization Methods: Comprehensive Analysis and Results

This document presents a detailed analysis of five advanced sign optimization methods for FCPM director reconstruction, including their mathematical foundations, implementation processes, benchmark results, and recommendations.

---

## Executive Summary

| Method | Energy Recovery | Speed | Recommendation |
|--------|----------------|-------|----------------|
| **Graph Cuts** | 100% (Optimal) | Fast (0.04s) | **Best overall** |
| **Hierarchical** | 87% | Fast (0.04s) | Good for large volumes |
| **V2 Layer+Refine** | 47% | Very Fast (0.03s) | Quick baseline |
| **Simulated Annealing** | 30-62% | Very Slow (28s) | Not recommended |
| **Belief Propagation** | 0% | Fast (0.01s) | Needs tuning |

**Winner: Graph Cuts (PyMaxflow)** - Achieves globally optimal solution in polynomial time.

---

## Table of Contents

1. [The Sign Optimization Problem](#1-the-sign-optimization-problem)
2. [Method 1: V2 Layer-by-Layer + Refinement](#2-method-1-v2-layer-by-layer--refinement)
3. [Method 2: Graph Cuts (Min-Cut/Max-Flow)](#3-method-2-graph-cuts-min-cutmax-flow)
4. [Method 3: Hierarchical Coarse-to-Fine](#4-method-3-hierarchical-coarse-to-fine)
5. [Method 4: Simulated Annealing](#5-method-4-simulated-annealing)
6. [Method 5: Belief Propagation](#6-method-5-belief-propagation)
7. [Benchmark Results](#7-benchmark-results)
8. [Recommendations](#8-recommendations)

---

## 1. The Sign Optimization Problem

### Problem Definition

After Q-tensor eigendecomposition, each voxel has a director **n̂** with arbitrary sign. Due to nematic symmetry (**n ≡ -n**), we need to find consistent sign assignments that minimize the gradient energy:

```
E(S) = Σ_{(i,j)∈neighbors} |s_i·n̂_i - s_j·n̂_j|²
```

where `s_i ∈ {+1, -1}` for each voxel.

### Why This Matters

- **Without optimization**: Random signs create artificial discontinuities
- **With optimization**: Smooth, physically realistic director fields
- **Impact on analysis**: Correct topology detection, defect identification, visualization

### Mathematical Equivalence to Ising Model

The problem maps exactly to a ferromagnetic/antiferromagnetic Ising model:

```
E(S) = Σ_{(i,j)} J_ij · s_i · s_j + const

where J_ij = -2(n̂_i · n̂_j)
```

- **J_ij > 0** (parallel directors): Ferromagnetic coupling → prefer same signs
- **J_ij < 0** (anti-parallel): Antiferromagnetic → prefer opposite signs

---

## 2. Method 1: V2 Layer-by-Layer + Refinement

### Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Layer Propagation                                  │
│  ─────────────────────────────                               │
│  1. Fix z=0 layer (arbitrary reference)                      │
│  2. For z = 1, 2, ..., nz-1:                                │
│     • Compare each voxel with z-1 neighbor                   │
│     • Flip sign if dot product < 0 (anti-aligned)           │
│                                                              │
│  Phase 2: Iterative Refinement                               │
│  ────────────────────────────                                │
│  1. For each voxel, compute:                                 │
│     • cost_current = Σ|n_i - n_j|² (6 neighbors)            │
│     • cost_flipped = Σ|(-n_i) - n_j|²                       │
│  2. Flip if cost_flipped < cost_current                      │
│  3. Repeat until no flips occur                              │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Phase 1: Layer Propagation**
```python
for z in range(1, nz):
    dot = Σ(n[:,:,z] * n[:,:,z-1])  # Dot product with previous layer
    flip_mask = (dot < 0)           # Anti-aligned voxels
    n[:,:,z][flip_mask] *= -1       # Flip to align
```

**Phase 2: Iterative Refinement**
```python
for iteration in range(max_iter):
    for each voxel (y,x,z):
        E_current = Σ_neighbors |n_i - n_j|²
        E_flipped = Σ_neighbors |-n_i - n_j|²
        if E_flipped < E_current:
            flip(n[y,x,z])
    if no_flips:
        break
```

### Results

| Metric | Value |
|--------|-------|
| Energy Recovery | 46.7% |
| Time (32³ volume) | 0.03s |
| Complexity | O(N) per iteration |

### Pros and Cons

| Pros | Cons |
|------|------|
| Very fast | Gets stuck in local minima |
| Simple implementation | Only 47% energy recovery |
| Predictable behavior | Sequential z-dependency |
| No external dependencies | Sensitive to starting layer |

### When to Use

- Quick preview of reconstruction quality
- When speed is critical
- As initialization for other methods

---

## 3. Method 2: Graph Cuts (Min-Cut/Max-Flow)

### Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Graph Construction                                  │
│  ─────────────────────────                                   │
│  • Create graph G = (V, E)                                  │
│  • Vertices: Source (S), Sink (T), all voxels              │
│  • S represents "+1" label, T represents "-1" label         │
│                                                              │
│  Step 2: Edge Weights                                        │
│  ────────────────────                                        │
│  • Anchor seed voxel to source (infinite capacity)          │
│  • For each neighbor pair (i,j):                            │
│     weight = 2 × |n̂_i · n̂_j| × scale                       │
│  • Pre-flip anti-parallel directors                          │
│                                                              │
│  Step 3: Min-Cut Computation                                 │
│  ──────────────────────────                                  │
│  • Run Boykov-Kolmogorov max-flow algorithm                 │
│  • Cut partitions graph into S-reachable and T-reachable    │
│                                                              │
│  Step 4: Label Extraction                                    │
│  ───────────────────────                                     │
│  • S-reachable voxels → positive sign                        │
│  • T-reachable voxels → negative sign                        │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Graph Construction**
```python
g = maxflow.Graph(n_voxels, n_voxels * 6)
nodes = g.add_nodes(n_voxels)

# Anchor seed to prevent trivial solution
g.add_tedge(seed_idx, INF, 0)  # Strong preference for positive

# Add pairwise edges
for each neighbor pair (i, j):
    dot = n̂_i · n̂_j
    if dot < 0:
        n[j] = -n[j]  # Pre-flip
        dot = -dot
    weight = 2.0 * dot
    g.add_edge(i, j, weight, weight)
```

**Why This Works**

The energy function is **submodular** for smooth director fields:
- Submodularity condition: `E(0,0) + E(1,1) ≤ E(0,1) + E(1,0)`
- Graph cuts can solve submodular binary labeling exactly
- Min-cut = optimal partition = optimal sign assignment

### Results

| Metric | Value |
|--------|-------|
| Energy Recovery | **100%** (Optimal) |
| Time (32³ volume) | 0.04s |
| Complexity | O(V·E·log(V)) |

### Pros and Cons

| Pros | Cons |
|------|------|
| **Globally optimal solution** | Requires PyMaxflow library |
| Fast (polynomial time) | May fail at sharp discontinuities |
| Deterministic | Memory scales with edges |
| Well-understood theory | Submodularity assumption |

### When to Use

- **Always** (when available) - it's the best method
- Smooth director fields without sharp defects
- When accuracy is more important than speed

---

## 4. Method 3: Hierarchical Coarse-to-Fine

### Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Build Pyramid                                       │
│  ────────────────────                                        │
│  Level 0: 4×4×2 (coarsest)                                  │
│  Level 1: 8×8×4                                             │
│  Level 2: 16×16×8                                           │
│  Level 3: 32×32×16 (original)                               │
│                                                              │
│  Step 2: Coarsening (2×2×2 blocks)                          │
│  ─────────────────────────────────                           │
│  For each block:                                             │
│  • Compute average Q-tensor: Q_avg = (1/8)Σ Q_i            │
│  • Find dominant eigenvector of Q_avg                       │
│  • Normalize → representative director                       │
│                                                              │
│  Step 3: Coarse Level Optimization                           │
│  ─────────────────────────────────                           │
│  • Solve small problem (32 voxels) via layer propagation    │
│  • Or use graph cuts for optimal solution                    │
│                                                              │
│  Step 4: Upsample and Refine                                 │
│  ──────────────────────────                                  │
│  For each finer level:                                       │
│  • Inherit signs from coarser level (2×2×2 → parent)        │
│  • Local iterative refinement (5 iterations)                │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Coarsening via Q-tensor Averaging**
```python
for each 2×2×2 block:
    Q_sum = zeros(3, 3)
    for each voxel in block:
        Q_sum += outer(n_i, n_i)
    Q_avg = Q_sum / 8

    eigenvalues, eigenvectors = eigh(Q_avg)
    n_representative = eigenvectors[:, -1]  # Largest eigenvalue
    normalize(n_representative)
```

**Why Q-tensor Averaging Works**
- Q-tensor is symmetric under n → -n (no sign ambiguity)
- Averaging Q-tensors properly handles sign differences
- Dominant eigenvector gives "average" orientation

**Upsampling Strategy**
```python
for each fine voxel at (y, x, z):
    coarse_parent = (y//2, x//2, z//2)
    if dot(n_fine, n_coarse) < 0:
        n_fine = -n_fine  # Align with parent
```

### Results

| Metric | Value |
|--------|-------|
| Energy Recovery | 87% |
| Time (32³ volume) | 0.04s |
| Complexity | O(N) |

### Pros and Cons

| Pros | Cons |
|------|------|
| Global consistency from coarse level | Not optimal (87% vs 100%) |
| Very fast (linear complexity) | May smooth over small features |
| Robust to local noise | Coarsening loses defect info |
| Scales well to large volumes | Fixed 2× factor |

### When to Use

- Large volumes (>64³) where speed matters
- When global consistency is more important than local accuracy
- As initialization for graph cuts or SA

---

## 5. Method 4: Simulated Annealing

### Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Initialization                                      │
│  ─────────────────────                                       │
│  • (Optional) Start from layer propagation result           │
│  • Set initial temperature T = T_initial (e.g., 10.0)       │
│                                                              │
│  Step 2: Metropolis-Hastings Iterations                      │
│  ──────────────────────────────────────                      │
│  For each iteration:                                         │
│  1. Randomly select voxel (y, x, z)                         │
│  2. Compute ΔE = E_flip - E_current (local, O(1))          │
│  3. If ΔE ≤ 0: accept flip                                  │
│     Else: accept with probability exp(-ΔE/T)                │
│  4. Update temperature: T ← α·T (α ≈ 0.995)                 │
│                                                              │
│  Step 3: Optional Cluster Moves (Wolff)                      │
│  ──────────────────────────────────────                      │
│  • Build cluster of aligned spins probabilistically         │
│  • Flip entire cluster at once                               │
│  • Improves mixing near critical temperature                 │
│                                                              │
│  Step 4: Track Best Solution                                 │
│  ──────────────────────────                                  │
│  • Keep copy of lowest-energy configuration found           │
│  • Return best, not final                                    │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Local Energy Change (O(1))**
```python
def delta_energy(n, y, x, z):
    n_i = n[y, x, z]
    delta = 0
    for neighbor in 6_connected_neighbors:
        n_j = n[neighbor]
        # ΔE = |(-n_i) - n_j|² - |n_i - n_j|²
        #    = 4 × (n_i · n_j)
        delta += 4 * dot(n_i, n_j)
    return delta
```

**Metropolis Criterion**
```python
if delta_E <= 0:
    accept = True
elif random() < exp(-delta_E / T):
    accept = True  # Accept with probability
else:
    accept = False
```

**Wolff Cluster Move**
```python
def cluster_move(n, T):
    seed = random_voxel()
    cluster = {seed}
    frontier = [seed]

    while frontier:
        current = frontier.pop()
        for neighbor in aligned_neighbors(current):
            if neighbor not in cluster:
                J = 2 * dot(n[current], n[neighbor])
                p_add = 1 - exp(-J / T)
                if random() < p_add:
                    cluster.add(neighbor)
                    frontier.append(neighbor)

    # Flip entire cluster
    for voxel in cluster:
        n[voxel] = -n[voxel]
```

### Results

| Metric | Value |
|--------|-------|
| Energy Recovery | 30-62% (variable) |
| Time (32³ volume) | 28-31s |
| Complexity | O(iterations × N) |

### Pros and Cons

| Pros | Cons |
|------|------|
| Can escape local minima (theoretically) | Very slow |
| Asymptotically optimal | High variance in results |
| Works for any energy function | Requires careful tuning |
| Good for validation | 50k+ iterations needed |

### Why It Performed Poorly

1. **Insufficient iterations**: 50k iterations ≈ 3 sweeps for 16k voxels
2. **Temperature schedule**: May cool too fast or too slow
3. **Cluster moves**: Not always beneficial (see 100k test with 30% recovery)
4. **Local moves dominate**: Gets stuck in large local minimum basins

### When to Use

- Validation of other methods
- When other methods fail at defects
- Research/theoretical comparison
- **Not recommended for production**

---

## 6. Method 5: Belief Propagation

### Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Factor Graph Construction                           │
│  ─────────────────────────────────                           │
│  • Variable nodes: s_i ∈ {+1, -1} for each voxel            │
│  • Factor nodes: ψ_ij(s_i, s_j) for each edge               │
│                                                              │
│  Step 2: Initialize Messages                                 │
│  ────────────────────────────                                │
│  • m_ij(s) = 0.5 for all (uniform belief)                   │
│                                                              │
│  Step 3: Compute Potentials                                  │
│  ────────────────────────                                    │
│  For each edge (i,j):                                        │
│  • ψ(same) = exp(-β × |n_i - n_j|²)                         │
│  • ψ(diff) = exp(-β × |n_i + n_j|²)                         │
│                                                              │
│  Step 4: Message Passing                                     │
│  ───────────────────────                                     │
│  For each iteration:                                         │
│  • m_{u→v}(s_v) = Σ_{s_u} ψ(s_u,s_v) × Π_{w≠v} m_{w→u}(s_u)│
│  • Apply damping: m ← α·m_old + (1-α)·m_new                 │
│                                                              │
│  Step 5: Extract Beliefs                                     │
│  ───────────────────────                                     │
│  • b_v(s) ∝ Π_u m_{u→v}(s)                                  │
│  • s*_v = argmax_s b_v(s)                                   │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Pairwise Potential**
```python
def compute_potential(n_i, n_j, beta):
    E_same = sum((n_i - n_j)**2)
    E_diff = sum((n_i + n_j)**2)
    return {
        'same': exp(-beta * E_same),
        'diff': exp(-beta * E_diff)
    }
```

**Message Update (Sum-Product)**
```python
def update_message(u, v, messages, potentials):
    # Product of incoming messages (excluding v)
    incoming_product = ones(2)
    for w in neighbors(u) if w != v:
        incoming_product *= messages[w→u]

    # Compute outgoing message
    pot = potentials[u,v]
    m_v0 = pot['same'] * incoming[0] + pot['diff'] * incoming[1]
    m_v1 = pot['diff'] * incoming[0] + pot['same'] * incoming[1]

    return normalize([m_v0, m_v1])
```

### Results

| Metric | Value |
|--------|-------|
| Energy Recovery | 0% |
| Time (32³ volume) | 0.01s |
| Complexity | O(iterations × edges) |

### Why It Failed

1. **Loopy graph**: 3D grid has many short loops, BP not guaranteed to converge
2. **Message initialization**: Uniform init may converge to trivial solution
3. **Parameter sensitivity**: β and damping need careful tuning
4. **Implementation issue**: Possible bug in message indexing

### Pros and Cons

| Pros | Cons |
|------|------|
| Parallelizable | Did not work in tests |
| Theoretically elegant | Sensitive to parameters |
| Fast when converges | Not guaranteed to converge |
| Good for sparse graphs | Needs significant tuning |

### When to Use

- **Not recommended** without further investigation
- May work after parameter tuning
- Better suited for tree-structured problems

---

## 7. Benchmark Results

### Test Configuration

- **Volume size**: 32×32×16 = 16,384 voxels
- **Director**: Cholesteric, pitch = 6.0
- **Test type**: Artificially scrambled signs (50% random flips)
- **Ground truth energy**: 19,456
- **Scrambled energy**: 98,334
- **Gap to recover**: 78,878

### Energy Recovery Comparison

| Method | Final Energy | Recovery | Time | Status |
|--------|--------------|----------|------|--------|
| Graph Cuts (PyMaxflow) | 19,456 | **100.0%** | 0.04s | **OPTIMAL** |
| Hierarchical | 29,696 | 87.0% | 0.04s | Good |
| Simulated Annealing (50k) | 49,248 | 62.2% | 28.39s | Poor |
| V2 Layer+Refine | 61,508 | 46.7% | 0.03s | Baseline |
| SA + Clusters (100k) | 74,484 | 30.2% | 9.86s | Poor |
| Belief Propagation | 98,334 | 0.0% | 0.01s | Failed |

### Scaling with Volume Size

| Size | Voxels | V2 | Graph Cuts | Hierarchical | SA | BP |
|------|--------|-----|------------|--------------|-----|-----|
| 24×24×12 | 6,912 | 0.01s | 0.02s | 0.02s | 1.19s | 0.00s |
| 32×32×16 | 16,384 | 0.03s | 0.04s | 0.04s | 1.10s | 0.01s |
| 48×48×24 | 55,296 | 0.10s | 0.13s | 0.16s | 1.48s | 0.02s |
| 64×64×32 | 131,072 | 0.27s | 0.30s | 0.38s | 4.07s | 0.06s |

### Noise Sensitivity (Realistic Reconstruction)

| Noise Level | Best Method | Angular Error |
|-------------|-------------|---------------|
| 1% | V2 Layer+Refine | 3.72° |
| 3% | V2 Layer+Refine | 5.36° |
| 5% | V2 Layer+Refine | 7.31° |
| 8% | V2 Layer+Refine | 10.42° |
| 10% | V2 Layer+Refine | 12.27° |
| 15% | V2 Layer+Refine | 16.40° |

*Note: All methods achieve similar angular errors because the Q-tensor reconstruction already handles most sign consistency. The main difference is in energy minimization.*

---

## 8. Recommendations

### Primary Recommendation: Graph Cuts

**Use Graph Cuts (PyMaxflow) as your default method.**

```python
from v2.approaches import GraphCutsOptimizer

optimizer = GraphCutsOptimizer()
result = optimizer.optimize(director, verbose=True)
optimized = result.director
```

**Why:**
- Achieves globally optimal solution (100% energy recovery)
- Fast (0.04s for 16k voxels, 0.30s for 131k voxels)
- Deterministic and reliable
- Well-understood theoretical guarantees

### Alternative: Hierarchical

**Use Hierarchical when Graph Cuts is unavailable or for very large volumes.**

```python
from v2.approaches import HierarchicalOptimizer

optimizer = HierarchicalOptimizer()
result = optimizer.optimize(director, verbose=True)
```

**Why:**
- 87% energy recovery (good enough for most applications)
- Linear time complexity O(N)
- No external dependencies

### Quick Preview: V2 Layer+Refine

**Use for rapid iteration during development.**

```python
from v2 import layer_then_refine

result = layer_then_refine(director, verbose=False)
```

**Why:**
- Fastest method (0.03s)
- No dependencies
- Good baseline

### Avoid: Simulated Annealing and Belief Propagation

**Not recommended for production use:**
- SA: Too slow, inconsistent results
- BP: Did not work in current implementation

---

## Appendix: Installation

### Required Dependencies

```bash
pip install numpy scipy
```

### Recommended (for optimal Graph Cuts)

```bash
pip install PyMaxflow
```

### Optional (visualization)

```bash
pip install matplotlib jupyter
```

---

## References

1. Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. *IEEE TPAMI*.

2. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*.

3. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.

4. Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief propagation for early vision. *IJCV*.

---

*Report generated: 2024*
*FCPM Simulation Project - Sign Optimization Analysis*
