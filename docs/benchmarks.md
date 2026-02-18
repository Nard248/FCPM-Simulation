# Benchmarks

## Sign Optimization Comparison

The validation script `examples/03_soliton_validation.py` benchmarks all six
sign-optimization methods on synthetic cholesteric fields with controlled sign
scrambling (50% random flips).

### Running the Benchmark

```bash
# Synthetic data (no input file needed)
python examples/03_soliton_validation.py

# With real LCSim data
python examples/03_soliton_validation.py --input data/CF1.npz --output results/

# With verbose output
python examples/03_soliton_validation.py -v
```

### Typical Results (64x64x32 Cholesteric, pitch=8)

| Method | Energy Reduction | Sign Accuracy | Energy Recovery | Time |
|--------|-----------------|---------------|-----------------|------|
| Combined (V1) | ~85-95% | ~0.85-0.95 | ~85-95% | <0.5s |
| LayerPropagation | ~85-95% | ~0.85-0.95 | ~85-95% | <0.5s |
| GraphCuts | ~95-100% | ~0.95-1.00 | ~95-100% | <1s |
| SimulatedAnnealing | ~90-98% | ~0.90-0.98 | ~90-98% | 1-10s |
| Hierarchical | ~90-98% | ~0.90-0.98 | ~90-98% | <1s |
| BeliefPropagation | ~80-95% | ~0.80-0.95 | ~80-95% | <1s |

!!! note
    Results depend on the specific structure, size, and random seed.
    Graph Cuts achieves the global optimum for the pairwise energy formulation.

### Metrics Explained

- **Energy Reduction**: Percentage decrease in gradient energy after optimization
- **Sign Accuracy**: Fraction of voxels whose sign matches the ground truth
- **Energy Recovery**: How much of the scrambling-induced energy increase is recovered
  (100% = perfect recovery to ground-truth energy)
- **Time**: Wall-clock time on a single CPU core

### Noise Sensitivity

The validation notebook (`examples/03_soliton_validation.ipynb`) includes a noise
sensitivity study that tests all methods at different noise levels (1%, 3%, 5%, 10%).
Graph Cuts and Hierarchical methods maintain the highest accuracy under noise.

## Numba Acceleration

When `numba` is installed (`pip install numba` or `uv sync --extra perf`),
the Simulated Annealing optimizer uses JIT-compiled kernels for the Metropolis
sweep, providing significant speedup:

```python
from fcpm.reconstruction.optimizers._numba_kernels import NUMBA_AVAILABLE
print(f"Numba available: {NUMBA_AVAILABLE}")
```

The first call incurs a compilation overhead; subsequent calls use cached machine code.
