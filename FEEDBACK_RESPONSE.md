# Response to Reviewer Feedback — FCPM v2.0.0

This document maps each of the 10 reviewer feedback points to the specific code
changes made in version 2.0.0.

---

## Feedback Point 1: Modern Packaging (UV / pyproject.toml)

**Reviewer request:** Replace legacy setup.py with modern Python packaging.

**Changes made:**

| File | Action | Details |
|------|--------|---------|
| `pyproject.toml` | Created | Hatchling build backend, requires-python >=3.10, structured optional dependencies |
| `setup.py` | Modified | Added deprecation comment; kept temporarily for pip backward compatibility |
| `README.md` | Modified | Installation section rewritten with UV-first instructions |

**Dependency structure in pyproject.toml:**
- **Core:** numpy>=1.24, scipy>=1.10, matplotlib>=3.7
- **`[full]`:** pymaxflow (graph cuts), tifffile (TIFF I/O), h5py (HDF5 I/O)
- **`[dev]`:** pytest, pytest-cov, ruff
- **`[perf]`:** numba (JIT-compiled kernels)
- **`[docs]`:** mkdocs, mkdocs-material, mkdocstrings

**Verification:** `uv sync && uv run python -c "import fcpm; print(fcpm.__version__)"` prints `2.0.0`.

---

## Feedback Point 2: README with Visual Results

**Reviewer request:** Add figures and visual results to the README.

**Changes made:**

| File | Section Added | Content |
|------|---------------|---------|
| `README.md` | "Sign Optimization (V2)" | Code example, optimizer comparison table (6 methods with speed/accuracy), Frank energy example |
| `README.md` | "V2 Sign Optimizers" (API table) | All new classes: SignOptimizer, OptimizationResult, 6 optimizer classes |
| `README.md` | "Examples" table | Added entries for validation notebook/script and v2 demo notebook |
| `README.md` | "References" | Short citations with link to REFERENCES.md |

**Note:** The notebook `examples/04_v2_demo.ipynb` contains embedded plot outputs (bar charts, director slices, error maps) that serve as the visual results. These are viewable directly on GitHub.

---

## Feedback Point 3: Scientific References

**Reviewer request:** Cite the algorithms and physics properly.

**Changes made:**

| File | Action | Details |
|------|--------|---------|
| `REFERENCES.md` | Created | 15 full citations with DOIs |
| `README.md` | Modified | Added "References" section linking to REFERENCES.md |
| `docs/references.md` | Created | Key papers section for documentation site |

**Citations cover:**
1. **LC Physics:** de Gennes & Prost (1993), Frank (1958)
2. **FCPM Technique:** Smalyukh et al. (2001), Smalyukh (2022)
3. **Solitons:** Ackerman & Smalyukh (2017), Tai & Smalyukh (2020)
4. **Graph Cuts:** Boykov & Kolmogorov (2004)
5. **Simulated Annealing:** Kirkpatrick et al. (1983), Wolff (1989)
6. **Belief Propagation:** Pearl (1988), Yedidia et al. (2003)
7. **Q-tensors:** Schiele & Trimper (1983)
8. **Numerical Methods:** Briggs et al. (2000), Lam & Suen (1997)

---

## Feedback Point 4: Architecture Refactor

**Reviewer request:** Unify the fragmented v2/ code into the main fcpm/ package with a clean class hierarchy.

**This is the largest change.** The v2/ directory had `sys.path.insert` hacks, duplicated utilities, and inconsistent return formats. All of this is now unified.

**New files created:**

| File | Purpose |
|------|---------|
| `fcpm/reconstruction/base.py` | `SignOptimizer` ABC + `OptimizationResult` dataclass |
| `fcpm/reconstruction/energy.py` | Canonical `compute_gradient_energy()` (replaces 3 duplicates) + `FrankConstants` + `compute_frank_energy_anisotropic()` |
| `fcpm/reconstruction/optimizers/__init__.py` | Subpackage exports |
| `fcpm/reconstruction/optimizers/combined.py` | `CombinedOptimizer` — wraps V1 `combined_optimization()` |
| `fcpm/reconstruction/optimizers/layer_propagation.py` | `LayerPropagationOptimizer` — extracted from `v2/sign_optimization_v2.py` |
| `fcpm/reconstruction/optimizers/graph_cuts.py` | `GraphCutsOptimizer` — moved from `v2/approaches/graph_cuts.py` |
| `fcpm/reconstruction/optimizers/simulated_annealing.py` | `SimulatedAnnealingOptimizer` — moved from `v2/approaches/simulated_annealing.py` |
| `fcpm/reconstruction/optimizers/hierarchical.py` | `HierarchicalOptimizer` — moved from `v2/approaches/hierarchical.py` |
| `fcpm/reconstruction/optimizers/belief_propagation.py` | `BeliefPropagationOptimizer` — moved from `v2/approaches/belief_propagation.py` |

**Key design decisions:**

1. **`SignOptimizer` ABC** — single method `optimize(director, verbose) -> OptimizationResult`. All 6 optimizers inherit from this.

2. **`OptimizationResult` dataclass** — uniform output with fields: `director`, `initial_energy`, `final_energy`, `energy_by_layer`, `flips_by_layer`, `total_flips`, `method`, `metadata` (extensible Dict for optimizer-specific data like convergence history).

3. **Config dataclasses** — each optimizer has its own config (e.g., `SimulatedAnnealingConfig(max_iterations=10000, initial_temperature=5.0, seed=42)`).

4. **Backward compatibility** — `v2/__init__.py` and `v2/approaches/__init__.py` converted to deprecation shims that import from `fcpm.reconstruction.optimizers`. Old imports still work with a DeprecationWarning.

5. **V1 functions preserved** — `from fcpm import combined_optimization`, `chain_propagation`, etc. all still work unchanged.

**Modified files:**

| File | Changes |
|------|---------|
| `fcpm/__init__.py` | Version 1.1.0 → 2.0.0, ~30 new exports added |
| `fcpm/reconstruction/__init__.py` | Added imports for all optimizer classes, configs, functional interfaces |
| `v2/__init__.py` | Converted to deprecation shim |
| `v2/approaches/__init__.py` | Converted to deprecation shim |

**Tests:** Created `tests/test_optimizers.py` with 35 tests covering inheritance, result types, energy non-increase, shape preservation, imports, backward compatibility, and Frank energy.

---

## Feedback Point 5: HDF5 Storage

**Reviewer request:** Support HDF5 for large datasets.

**Changes made:**

| File | Function Added | Details |
|------|----------------|---------|
| `fcpm/io/exporters.py` | `save_director_hdf5()` | Stores nx/ny/nz arrays with gzip compression; metadata as HDF5 attributes |
| `fcpm/io/loaders.py` | `load_director_hdf5()` | Reads nx/ny/nz from HDF5 file |
| `fcpm/io/loaders.py` | `load_director()` | Updated auto-detection for `.h5`/`.hdf5` extensions |
| `fcpm/io/__init__.py` | — | Added exports |
| `fcpm/__init__.py` | — | Added exports |

**Guarded import:** `h5py` is imported inside the functions with try/except, so the core library works without it.

---

## Feedback Point 6: Documentation Site

**Reviewer request:** Add proper documentation beyond the README.

**Changes made:**

| File | Purpose |
|------|---------|
| `mkdocs.yml` | Site config: Material theme, mkdocstrings plugin, navigation structure |
| `docs/index.md` | Landing page with feature list |
| `docs/installation.md` | UV + pip installation guide |
| `docs/quickstart.md` | Two-workflow quick start |
| `docs/workflows.md` | Detailed pipeline documentation |
| `docs/sign_optimization.md` | All 6 methods with code examples and comparison table |
| `docs/preprocessing.md` | Cropping, filtering, noise |
| `docs/visualization.md` | All plot functions |
| `docs/benchmarks.md` | Performance data and Numba notes |
| `docs/references.md` | Key papers |
| `docs/api/core.md` | Core API (DirectorField, QTensor, FCPMSimulator) |
| `docs/api/reconstruction.md` | Reconstruction + energy API |
| `docs/api/optimizers.md` | All 6 optimizer classes + configs |
| `docs/api/io.md` | Loaders and exporters |
| `docs/api/visualization.md` | Plot functions |
| `docs/api/utils.md` | Metrics and noise |
| `.github/workflows/docs.yml` | Auto-deploy on push to main |

**API docs** use `:::` directives from mkdocstrings, so they auto-generate from docstrings and stay in sync with code changes.

---

## Feedback Point 7: Anisotropic Frank Energy

**Reviewer request:** Implement the full Frank elastic energy, not just the single-constant approximation.

**Changes made:**

| File | What | Details |
|------|------|---------|
| `fcpm/reconstruction/energy.py` | `FrankConstants` | Dataclass: K1=10.3, K2=7.4, K3=16.48 pN (5CB defaults), optional pitch |
| `fcpm/reconstruction/energy.py` | `compute_frank_energy_anisotropic()` | Full decomposition using central finite differences |
| `fcpm/__init__.py` | — | Exports `FrankConstants`, `compute_frank_energy_anisotropic` |

**Physics implemented:**

```
f_splay = (K1/2)(∇·n)²
f_twist = (K2/2)(n·∇×n + q₀)²     where q₀ = 2π/pitch
f_bend  = (K3/2)|n×∇×n|²
```

Spatial derivatives computed via central finite differences:
`∂nᵢ/∂xⱼ ≈ (nᵢ[j+1] - nᵢ[j-1]) / 2h`

**Returns:** Per-voxel arrays (splay, twist, bend, total) AND integrated scalars (splay_integrated, twist_integrated, bend_integrated, total_integrated).

**Validation:** For a perfect cholesteric with matching pitch, the twist term is near-zero because q₀ cancels the natural helical twist.

---

## Feedback Point 8: Pointwise Metrics

**Reviewer request:** Add sign accuracy and spatial error distribution metrics.

**Changes made:**

| File | Function | Details |
|------|----------|---------|
| `fcpm/utils/metrics.py` | `sign_accuracy(recon, gt)` | Fraction of voxels where dot(n_recon, n_gt) > 0. Range [0, 1]. |
| `fcpm/utils/metrics.py` | `spatial_error_distribution(recon, gt)` | Returns dict with `error_map` (full 3D array), `layer_mean`, `layer_median`, `layer_max` (per-z-layer statistics) |
| `fcpm/utils/metrics.py` | `summary_metrics()` | Extended to include `sign_accuracy` when ground truth is provided |
| `fcpm/utils/__init__.py` | — | Added exports |
| `fcpm/__init__.py` | — | Added exports |

**Usage:**
```python
acc = fcpm.sign_accuracy(optimized_director, ground_truth)  # e.g. 0.95
dist = fcpm.spatial_error_distribution(optimized_director, ground_truth)
# dist['layer_mean'] is a (nz,) array of mean angular error per z-layer
```

---

## Feedback Point 9: Numba Acceleration

**Reviewer request:** JIT-compile the performance-critical simulated annealing inner loop.

**Changes made:**

| File | Function | Details |
|------|----------|---------|
| `fcpm/reconstruction/optimizers/_numba_kernels.py` | `sa_compute_delta_energy()` | JIT-compiled O(1) local energy delta (6-neighbor sum) |
| `fcpm/reconstruction/optimizers/_numba_kernels.py` | `sa_metropolis_sweep()` | JIT-compiled full raster-order Metropolis sweep |
| `fcpm/reconstruction/optimizers/_numba_kernels.py` | `NUMBA_AVAILABLE` | Runtime flag for detection |

**Design:**
- If numba is installed: functions are decorated with `@njit(cache=True)`
- If numba is NOT installed: identical pure-Python fallback functions are defined
- The SA optimizer imports from `_numba_kernels` unconditionally — it works either way
- First call incurs JIT compilation overhead; subsequent calls use cached machine code

---

## Feedback Point 10: Real Data Validation

**Reviewer request:** Validate on real soliton structures from article.lcpen / LCSim data.

**Changes made:**

| File | Purpose |
|------|---------|
| `fcpm/io/loaders.py` → `load_lcsim_npz()` | Reads LCSim NPZ format: PATH key (shape 1,sx,sy,sz,1,3), settings JSON |
| `examples/03_soliton_validation.py` | CLI benchmark: `python examples/03_soliton_validation.py --input data/CF1.npz` |
| `examples/03_soliton_validation.ipynb` | Interactive validation notebook (28 cells) |
| `examples/04_v2_demo.ipynb` | Comprehensive v2 demo — fully executed with all outputs (37 cells) |

**04_v2_demo.ipynb contains:**
1. Ground truth cholesteric visualization (3 z-slices)
2. Frank energy decomposition (splay/twist/bend fractions)
3. Controlled sign-scramble experiment
4. Benchmark of all 6 optimizers (table + bar charts)
5. Director slice comparison for every method
6. Angular error maps per method
7. Frank energy before/after comparison
8. Spatial error distribution by depth
9. Full FCPM pipeline (simulate → noise → reconstruct → optimize → evaluate)
10. Noise sensitivity study across 4 noise levels

**Key result:** GraphCuts achieves the highest sign accuracy; Hierarchical provides the best speed/accuracy trade-off.

---

## Summary of All Files Changed

### New files: 33
### Modified files: 11
### Total lines added: ~6,200
### Tests: 69 passing (35 new + 34 existing)
### Version: 1.1.0 → 2.0.0
### Backward compatibility: Full (all v1 imports preserved)
