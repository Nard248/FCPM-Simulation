# What the Library Can and Cannot Do

Honest assessment of the library's current capabilities, limitations, and reliability by field type.

## What the Library Can Do

| Capability | Status | Confidence |
|------------|:------:|:----------:|
| Simulate FCPM intensities from a known director | Done | High |
| Reconstruct Q-tensor from 4-angle FCPM | Done | High (for Q_xx, Q_yy, Q_xy) |
| Extract director via eigendecomposition | Done | High |
| Resolve sign ambiguity (6 algorithms) | Done | Medium-High (see below) |
| Compute gradient and Frank elastic energies | Done | High |
| Compute per-voxel confidence and ambiguity maps | Done | Medium |
| Detect branch cuts in the director field | Done | Medium |
| Generate topology test fields (disclinations, skyrmions, torons) | Done | Medium |
| Add realistic noise (Poisson + Gaussian + systematics) | Done | Medium |
| Read/write NPZ, NPY, MAT, HDF5, TIFF, VTK | Done | High |
| Preprocessing (crop, filter, normalize, background removal) | Done | High |
| Sensitivity analysis framework | Done | Medium |

## What the Library Cannot Do (Yet)

| Limitation | Impact | Planned |
|------------|--------|:-------:|
| No likelihood-based / MAP reconstruction | Uses heuristic energy minimization, not statistically optimal | Priority 3.2 |
| No noise propagation through reconstruction | Confidence map is geometry-based, not noise-aware | Priority 3.2 |
| No automated benchmark runner (CLI) | Benchmarks require running notebooks manually | Priority 5.2 |
| No CI pipeline for regression tests | Tests run locally only | Priority 5 |
| Limited 3D visualization | Requires optional `topovec` or external tools | -- |

## Reliability by Field Type

| Field Type | Reconstruction | Sign Optimization | Notes |
|------------|:-:|:-:|-----|
| Uniform / planar tilt | Excellent | Excellent | Trivial case |
| Cholesteric twist | Excellent | Excellent | All optimizers perform well |
| Radial (hedgehog) | Good | Good | Singularity at center has low confidence |
| Integer disclinations (+1, -1) | Good | Good | Sign optimization works; branch cuts are algorithmic artifacts only |
| Half-integer disclinations (+1/2, -1/2) | Partial | **Caution** | No smooth director exists. Use Q-tensor comparison. `ambiguity_mask` should be checked |
| Z-solitons | Good | Good | Tested on LCSim data (19 structures) |
| Torons | Good | Medium | Complex 3D topology, may need Hierarchical optimizer |
| Near-vertical director (nz ~ 1) | **Poor** | N/A | In-plane angle is numerically unstable. `confidence_map` drops to ~0 |
| High noise (SNR < 10 dB) | Poor | Degrades | Energy minimization may introduce bias |
| Model mismatch (wrong K, wrong pitch) | Unknown | Unknown | Sensitivity study framework available but not yet systematically validated |

## Sign Optimizer Reliability

Based on Priority 0 correctness testing (89 tests, brute-force verification on small fields):

| Optimizer | Energy Monotonicity | Correctness | Speed | Recommendation |
|-----------|:---:|:---:|:---:|-----|
| Combined (V1) | Guaranteed | Verified (brute-force) | Fast | **Default choice** |
| GraphCuts | Guaranteed | Verified (brute-force) | Medium | Best for clean data |
| Hierarchical | Guaranteed (with energy guard) | Good | Medium | Good for large volumes |
| LayerPropagation | Guaranteed (with energy guard) | Good | Fast | Good for layered structures |
| SimulatedAnnealing | Stochastic | Good (approaches optimum) | Slow | When trapped in local minima |
| BeliefPropagation | Not guaranteed | **Experimental** | Medium | Not recommended for production |

"Guaranteed" means the optimizer never increases the gradient energy, verified on 300 random fields.
