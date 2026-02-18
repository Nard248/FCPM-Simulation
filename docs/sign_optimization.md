# Sign Optimization

## The Sign Ambiguity Problem

In nematic liquid crystals, the director **n** and **-n** are physically equivalent.
When reconstructing from FCPM data, each voxel's sign is determined independently,
leading to inconsistent sign assignments that create artificial discontinuities.

Sign optimization resolves this by minimizing the elastic (gradient) energy:

$$E = \frac{K}{2} \int |\nabla \mathbf{n}|^2 \, dV$$

## Using the Optimizers

All optimizers follow the same interface:

```python
import fcpm

director = fcpm.load_director('reconstructed.npz')

# Pick an optimizer
optimizer = fcpm.GraphCutsOptimizer()
result = optimizer.optimize(director, verbose=True)

# Result contains everything
print(f"Energy: {result.initial_energy:.1f} -> {result.final_energy:.1f}")
print(f"Reduction: {result.energy_reduction_pct:.1f}%")
director_fixed = result.director
```

## Available Methods

### Combined (V1)

The original approach: chain propagation followed by iterative local flips.
Fast and reliable baseline.

```python
optimizer = fcpm.CombinedOptimizer()
```

### Layer Propagation

Propagates sign consistency layer-by-layer along z, then refines iteratively.
Good for structures with strong z-axis ordering.

```python
optimizer = fcpm.LayerPropagationOptimizer()
```

### Graph Cuts

Formulates sign optimization as a binary labeling problem and solves via
min-cut/max-flow. Achieves the global optimum for the pairwise energy.

```python
optimizer = fcpm.GraphCutsOptimizer()
# Requires pymaxflow (pip install pymaxflow), falls back to NetworkX
```

### Simulated Annealing

Metropolis-Hastings with adaptive temperature schedule and optional Wolff
cluster moves. Can escape local minima but requires more time.

```python
optimizer = fcpm.SimulatedAnnealingOptimizer(
    fcpm.SimulatedAnnealingConfig(
        max_iterations=10000,
        initial_temperature=5.0,
        seed=42,
    )
)
```

### Hierarchical

Multi-scale coarse-to-fine approach. Coarsens the volume using Q-tensor
averaging, optimizes at coarse scale, then refines progressively.

```python
optimizer = fcpm.HierarchicalOptimizer()
```

### Belief Propagation

Message passing on a factor graph. Experimental — may not converge on all
structures.

```python
optimizer = fcpm.BeliefPropagationOptimizer(
    fcpm.BeliefPropagationConfig(max_iterations=30)
)
```

## Comparison

| Method | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| Combined | Good | Fast | Reliable baseline |
| LayerPropagation | Good | Fast | Best for layered structures |
| GraphCuts | Best | Medium | Global optimum for binary energy |
| SimulatedAnnealing | Very Good | Slow | Can escape local minima |
| Hierarchical | Very Good | Medium | Good speed/accuracy balance |
| BeliefPropagation | Variable | Medium | Experimental |

## Frank Energy Analysis

For detailed energy analysis with anisotropic elastic constants:

```python
from fcpm import FrankConstants, compute_frank_energy_anisotropic

frank = FrankConstants(K1=10.3, K2=7.4, K3=16.48)  # 5CB constants (pN)
energy = compute_frank_energy_anisotropic(director.to_array(), frank)

print(f"Splay (K1): {energy['splay_integrated']:.2f}")
print(f"Twist (K2): {energy['twist_integrated']:.2f}")
print(f"Bend  (K3): {energy['bend_integrated']:.2f}")
print(f"Total:      {energy['total_integrated']:.2f}")
```
