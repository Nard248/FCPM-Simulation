# FCPM v2.0.0 — Changelog and Technical Description

## Summary

Version 2.0.0 is a major release that addresses 10 expert reviewer feedback points.
It unifies the fragmented `v2/` experimental code into a clean, tested, and documented
architecture under the main `fcpm/` package.

---

## Phase 1: Architecture Refactor

### Problem
Sign optimization code lived in a separate `v2/` directory with `sys.path` hacks
and duplicated utilities. Each optimizer had its own way of returning results.

### Changes

**New file: `fcpm/reconstruction/base.py`**
- `SignOptimizer` — Abstract base class (ABC) with a single method: `optimize(director, verbose) -> OptimizationResult`
- `OptimizationResult` — Dataclass with fields: `director`, `initial_energy`, `final_energy`, `energy_by_layer`, `flips_by_layer`, `total_flips`, `method`, `metadata` (extensible dict)
- Added computed properties: `energy_reduction`, `energy_reduction_pct`

**New file: `fcpm/reconstruction/energy.py`**
- `compute_gradient_energy()` — Canonical implementation replacing 3 duplicates across v1/v2
- `FrankConstants` — Dataclass for K1 (splay), K2 (twist), K3 (bend), pitch
- `compute_frank_energy_anisotropic()` — Full Frank elastic energy decomposition using central finite differences for div(n), curl(n), splay/twist/bend separation

**New subpackage: `fcpm/reconstruction/optimizers/`**

| File | Source | Optimizer |
|------|--------|-----------|
| `combined.py` | Wraps V1 `combined_optimization()` | `CombinedOptimizer` |
| `layer_propagation.py` | Extracted from `v2/sign_optimization_v2.py` | `LayerPropagationOptimizer` |
| `graph_cuts.py` | Moved from `v2/approaches/graph_cuts.py` | `GraphCutsOptimizer` |
| `simulated_annealing.py` | Moved from `v2/approaches/simulated_annealing.py` | `SimulatedAnnealingOptimizer` |
| `hierarchical.py` | Moved from `v2/approaches/hierarchical.py` | `HierarchicalOptimizer` |
| `belief_propagation.py` | Moved from `v2/approaches/belief_propagation.py` | `BeliefPropagationOptimizer` |

Each optimizer removed its `sys.path.insert` hack and imports from `..base` instead.
History/convergence data stored in `metadata` dict instead of ad-hoc attributes.

**Modified: `fcpm/reconstruction/__init__.py`**
- Added imports for all new classes, configs, and functional interfaces
- V1 function exports preserved for backward compatibility

**Modified: `fcpm/__init__.py`**
- Version bumped from `1.1.0` to `2.0.0`
- Added ~30 new exports

**Modified: `v2/__init__.py` and `v2/approaches/__init__.py`**
- Converted to deprecation shims that import from `fcpm.reconstruction.optimizers`
- Emit `DeprecationWarning` on import

**New: `tests/test_optimizers.py`**
- 35 tests covering: inheritance, result type, energy non-increase, shape preservation, method field, imports, backward compatibility, energy module, Frank energy, OptimizationResult properties

---

## Phase 2: UV Packaging

### Problem
No `pyproject.toml`; only a legacy `setup.py`.

### Changes

**New: `pyproject.toml`**
- Build backend: hatchling
- Core deps: numpy>=1.24, scipy>=1.10, matplotlib>=3.7
- Optional extras: `full` (pymaxflow, tifffile, h5py), `dev` (pytest, ruff), `perf` (numba), `docs` (mkdocs)
- Ruff config: target Python 3.10, line-length 100

**Modified: `setup.py`**
- Kept temporarily with deprecation comment

**Modified: `README.md`**
- Installation section rewritten with UV-first instructions

---

## Phase 3: Physics, Metrics, HDF5, Numba

### 3.1 Anisotropic Frank Energy

**File: `fcpm/reconstruction/energy.py`** (created in Phase 1, extended here)
- `FrankConstants(K1=10.3, K2=7.4, K3=16.48, pitch=None)` — 5CB defaults in pN
- `compute_frank_energy_anisotropic()` computes:
  - `div(n) = dn_x/dx + dn_y/dy + dn_z/dz` (splay)
  - `n . curl(n) + q0` (twist, with cholesteric pitch)
  - `|n x curl(n)|^2` (bend)
- Returns per-voxel arrays and integrated scalars for each component

### 3.2 Pointwise Metrics

**Modified: `fcpm/utils/metrics.py`**
- `sign_accuracy(recon, gt)` — Fraction of voxels with `dot(n_recon, n_gt) > 0`
- `spatial_error_distribution(recon, gt)` — Per-z-layer mean/median/max angular error + full error map
- `summary_metrics()` extended to include `sign_accuracy`

### 3.3 HDF5 Storage

**Modified: `fcpm/io/exporters.py`**
- `save_director_hdf5(director, filepath, metadata)` — Stores nx/ny/nz with gzip compression, metadata as HDF5 attributes

**Modified: `fcpm/io/loaders.py`**
- `load_director_hdf5(filepath)` — Reads nx/ny/nz from HDF5
- `load_lcsim_npz(filepath)` — Reads article.lcpen format: PATH key with shape `(1,sx,sy,sz,1,3)`, parses settings JSON
- `load_director()` auto-detection updated for `.h5`/`.hdf5`

### 3.4 Numba Acceleration

**New: `fcpm/reconstruction/optimizers/_numba_kernels.py`**
- `sa_compute_delta_energy(n, y, x, z)` — JIT-compiled local energy delta in O(1) per voxel
- `sa_metropolis_sweep(n, temperature, rng_state)` — JIT-compiled full Metropolis sweep
- Pure-Python fallbacks when numba is not installed
- `NUMBA_AVAILABLE` flag for runtime detection

---

## Phase 4: Real Data Validation

**Modified: `fcpm/io/loaders.py`**
- `load_lcsim_npz()` parses LCSim/article.lcpen NPZ files with `PATH` key

**New: `examples/03_soliton_validation.py`**
- CLI script: loads real or synthetic data, scrambles 50% of signs, runs all 6 optimizers
- Reports: energy reduction, sign accuracy, energy recovery, timing
- Saves JSON results and matplotlib comparison bar charts to output directory
- Usage: `python examples/03_soliton_validation.py [--input FILE] [--output DIR]`

**New: `examples/03_soliton_validation.ipynb`**
- Interactive notebook with 28 cells
- Sections: load data, scramble signs, run all optimizers, summary table, bar charts, director slice comparison, error maps, Frank energy decomposition, noise sensitivity study, spatial error distribution

---

## Phase 5: README and References

**New: `REFERENCES.md`**
- 15 scientific citations covering: LC physics (de Gennes, Frank), FCPM technique (Smalyukh), solitons (Ackerman), graph cuts (Boykov), simulated annealing (Kirkpatrick), Wolff clusters, belief propagation (Pearl, Yedidia), Q-tensors (Schiele), multigrid (Briggs)

**Modified: `README.md`**
- Added "Sign Optimization (V2)" section with usage examples and optimizer comparison table
- Added Frank energy analysis example
- Added V2 optimizer classes to API reference table
- Added "References" section linking to REFERENCES.md
- Added validation notebook/script to examples table

---

## Phase 6: MkDocs Documentation Site

**New: `mkdocs.yml`**
- Material theme with dark/light toggle
- mkdocstrings plugin for auto-generated Python API docs
- Navigation: Home, Getting Started, User Guide, API Reference, Benchmarks, References

**New: `docs/` directory** (10 markdown pages)
- `index.md` — Landing page
- `installation.md` — UV + pip instructions
- `quickstart.md` — Two workflow examples
- `workflows.md` — Detailed pipeline documentation
- `sign_optimization.md` — All 6 methods with code examples and comparison table
- `preprocessing.md` — Cropping, filtering, noise
- `visualization.md` — All plot functions
- `benchmarks.md` — Performance data and Numba acceleration
- `references.md` — Key papers

**New: `docs/api/`** (6 API reference pages)
- `core.md`, `reconstruction.md`, `optimizers.md`, `io.md`, `visualization.md`, `utils.md`
- Each uses `:::` directives for mkdocstrings auto-generation

**New: `.github/workflows/docs.yml`**
- Deploys documentation on push to main via `mkdocs gh-deploy`

---

## New Files Created (Total: 25)

| File | Purpose |
|------|---------|
| `fcpm/reconstruction/base.py` | SignOptimizer ABC + OptimizationResult |
| `fcpm/reconstruction/energy.py` | Gradient energy + Frank energy |
| `fcpm/reconstruction/optimizers/__init__.py` | Subpackage exports |
| `fcpm/reconstruction/optimizers/combined.py` | CombinedOptimizer |
| `fcpm/reconstruction/optimizers/layer_propagation.py` | LayerPropagationOptimizer |
| `fcpm/reconstruction/optimizers/graph_cuts.py` | GraphCutsOptimizer |
| `fcpm/reconstruction/optimizers/simulated_annealing.py` | SimulatedAnnealingOptimizer |
| `fcpm/reconstruction/optimizers/hierarchical.py` | HierarchicalOptimizer |
| `fcpm/reconstruction/optimizers/belief_propagation.py` | BeliefPropagationOptimizer |
| `fcpm/reconstruction/optimizers/_numba_kernels.py` | Numba JIT kernels |
| `tests/test_optimizers.py` | 35 optimizer tests |
| `pyproject.toml` | Modern packaging |
| `examples/03_soliton_validation.py` | Validation CLI script |
| `examples/03_soliton_validation.ipynb` | Validation notebook |
| `REFERENCES.md` | Scientific citations |
| `CHANGELOG_V2.md` | This document |
| `mkdocs.yml` | Documentation site config |
| `docs/index.md` | Docs landing page |
| `docs/installation.md` | Installation guide |
| `docs/quickstart.md` | Quick start guide |
| `docs/workflows.md` | Workflow documentation |
| `docs/sign_optimization.md` | Sign optimization guide |
| `docs/preprocessing.md` | Preprocessing docs |
| `docs/visualization.md` | Visualization docs |
| `docs/benchmarks.md` | Benchmark results |
| `docs/references.md` | References page |
| `docs/api/core.md` | Core API reference |
| `docs/api/reconstruction.md` | Reconstruction API reference |
| `docs/api/optimizers.md` | Optimizers API reference |
| `docs/api/io.md` | I/O API reference |
| `docs/api/visualization.md` | Visualization API reference |
| `docs/api/utils.md` | Utilities API reference |
| `.github/workflows/docs.yml` | Documentation auto-deploy |

## Modified Files (Total: 11)

| File | Changes |
|------|---------|
| `fcpm/__init__.py` | Version 2.0.0, ~30 new exports |
| `fcpm/reconstruction/__init__.py` | Optimizer class exports |
| `fcpm/utils/metrics.py` | sign_accuracy, spatial_error_distribution |
| `fcpm/utils/__init__.py` | New metric exports |
| `fcpm/io/loaders.py` | load_director_hdf5, load_lcsim_npz |
| `fcpm/io/exporters.py` | save_director_hdf5 |
| `fcpm/io/__init__.py` | New I/O exports |
| `v2/__init__.py` | Deprecation shim |
| `v2/approaches/__init__.py` | Deprecation shim |
| `setup.py` | Deprecation comment |
| `README.md` | V2 API, references, examples |

---

## Backward Compatibility

All V1 imports continue to work unchanged:
- `from fcpm import combined_optimization`
- `from fcpm import gradient_energy`
- `from v2.approaches import GraphCutsOptimizer` (with deprecation warning)

All 69 existing tests pass.
