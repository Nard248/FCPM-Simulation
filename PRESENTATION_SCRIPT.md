# Presentation Script — FCPM v2.0 Updates

## How to Present

**Open two things side by side:**
1. This script (printed or on a secondary screen)
2. `examples/04_v2_demo.ipynb` in Jupyter — the notebook is pre-executed, just scroll through it. No need to re-run.

**Total time:** ~15-20 minutes + Q&A

---

## Opening (1 minute)

> "We've addressed all 10 feedback points from the review. The result is FCPM v2.0 — a major release that unifies the codebase, adds proper physics, modern packaging, and validated benchmarks. Let me walk you through each change."

---

## Feedback Point 1: Modern Packaging

**What reviewers said:** Use UV / pyproject.toml instead of setup.py.

**What we did:**
- Created `pyproject.toml` with hatchling build backend
- Core dependencies: numpy, scipy, matplotlib
- Optional extras: `full` (pymaxflow, tifffile, h5py), `dev` (pytest, ruff), `perf` (numba), `docs` (mkdocs)
- Installation is now `uv sync` or `pip install -e .`

**Show:** Open `pyproject.toml` and point out the clean dependency structure.

> "Installation is now one command. Optional dependencies are grouped by use case — you only install what you need."

---

## Feedback Point 2: README with Visuals

**What reviewers said:** Add figures and visual results to README.

**What we did:**
- Added "Sign Optimization (V2)" section with optimizer comparison table
- Added Frank energy analysis code example
- Added V2 API reference table listing all new classes
- Added examples table showing the new validation notebooks
- Added references section

**Show:** Scroll through `README.md` on GitHub — point out the optimizer table, the code examples, the API reference.

> "The README now serves as both a quick-start guide and an API overview. Someone can understand all 6 optimization methods just from the README."

---

## Feedback Point 3: Scientific References

**What reviewers said:** Cite the underlying algorithms and physics properly.

**What we did:**
- Created `REFERENCES.md` with 15 full citations
- Covers: LC physics (de Gennes, Frank), FCPM technique (Smalyukh), solitons (Ackerman), graph cuts (Boykov), simulated annealing (Kirkpatrick), Wolff clusters, belief propagation (Pearl), Q-tensors, multigrid methods
- README links to REFERENCES.md

**Show:** Open `REFERENCES.md` briefly.

> "Every algorithm we implement is now properly attributed. Graph cuts cites Boykov & Kolmogorov 2004, simulated annealing cites Kirkpatrick 1983, and so on."

---

## Feedback Point 4: Architecture Refactor — THIS IS THE BIG ONE

**What reviewers said:** Unify the fragmented v2/ experimental code into the main package.

**What we did:**
- Created `SignOptimizer` abstract base class in `fcpm/reconstruction/base.py`
- Created `OptimizationResult` dataclass with energy, flips, metadata
- Moved all 6 optimizers into `fcpm/reconstruction/optimizers/` subpackage
- Removed all `sys.path.insert` hacks from v2/
- Converted v2/ to deprecation shims (old code still works, emits warnings)
- Every optimizer now has the same interface: `optimizer.optimize(director) -> OptimizationResult`

**Show in notebook:** Scroll to **Section 4** ("Unified Sign Optimizer Framework"). Point out:
- Cell showing all 6 classes inherit from `SignOptimizer`
- The uniform benchmark loop — same 3 lines of code for every optimizer
- The results table with energy reduction, sign accuracy, flips, and timing

> "Before this refactor, each optimizer had its own return format and import path. Now they all share one interface. You can swap optimizers with a single line change."

**Key talking point:**

> "The architecture follows the Strategy Pattern. `SignOptimizer` is the ABC, each concrete optimizer implements `optimize()`, and `OptimizationResult` is the uniform output. This makes it trivial to add new methods — just subclass and implement."

---

## Feedback Point 5: HDF5 Storage

**What reviewers said:** Support HDF5 for large datasets.

**What we did:**
- `save_director_hdf5()` — stores nx/ny/nz with gzip compression, metadata as HDF5 attributes
- `load_director_hdf5()` — reads back with auto-detection for .h5/.hdf5 extensions
- `load_director()` auto-detection updated

> "For large experimental datasets, HDF5 with gzip compression is significantly more efficient than NPZ. The API is the same — load_director() auto-detects the format."

---

## Feedback Point 6: Documentation Site

**What reviewers said:** Add proper documentation beyond the README.

**What we did:**
- Created MkDocs site with Material theme (`mkdocs.yml`)
- 10 user guide pages: installation, quick start, workflows, sign optimization, preprocessing, visualization, benchmarks, references
- 6 API reference pages with auto-generated docstrings via mkdocstrings
- GitHub Actions workflow for auto-deploy on push to main

**Show:** Open `mkdocs.yml` briefly, show the navigation structure.

> "The docs site auto-deploys from main. API docs are generated from docstrings, so they stay in sync with the code."

---

## Feedback Point 7: Anisotropic Frank Energy — IMPORTANT PHYSICS

**What reviewers said:** Implement the full Frank elastic energy, not just the single-constant gradient approximation.

**What we did:**
- `FrankConstants` dataclass: K1=10.3 (splay), K2=7.4 (twist), K3=16.48 (bend) pN — 5CB defaults
- `compute_frank_energy_anisotropic()` decomposes into:
  - Splay: (K1/2)(div n)^2
  - Twist: (K2/2)(n · curl n + q0)^2  (with cholesteric pitch)
  - Bend: (K3/2)|n × curl n|^2
- Uses central finite differences for spatial derivatives
- Returns per-voxel arrays AND integrated scalars

**Show in notebook:** Scroll to **Section 2** ("Anisotropic Frank Energy"). Point out:
- The Frank constants printout
- The energy decomposition table (splay/twist/bend fractions)
- Then scroll to **Section 8** for the bar chart comparing GT vs scrambled vs optimized

> "For a cholesteric with matching pitch, the twist energy is near-zero because the equilibrium twist q0 cancels the natural helical twist. After scrambling, all three components spike. After optimization, they recover close to ground truth."

**Key talking point:**

> "The gradient energy is a good cost function for optimization, but for physics analysis you need the anisotropic decomposition. Now we have both."

---

## Feedback Point 8: Pointwise Metrics

**What reviewers said:** Add per-voxel sign accuracy and spatial error distribution.

**What we did:**
- `sign_accuracy(recon, gt)` — fraction of voxels with correct sign (dot > 0)
- `spatial_error_distribution(recon, gt)` — per-z-layer mean/median/max angular error + full error map
- Extended `summary_metrics()` to include sign accuracy

**Show in notebook:** Scroll to **Section 9** ("Spatial Error Distribution"). Point out the depth-profile plot showing mean/median/max angular error per z-layer.

> "Sign accuracy tells you what fraction of voxels the optimizer got right. The spatial distribution shows whether errors concentrate at the boundaries or are uniform — critical for understanding where reconstruction fails."

---

## Feedback Point 9: Numba Acceleration

**What reviewers said:** JIT-compile the performance-critical inner loop.

**What we did:**
- Created `_numba_kernels.py` with JIT-compiled versions of:
  - `sa_compute_delta_energy()` — O(1) per voxel energy delta
  - `sa_metropolis_sweep()` — full Metropolis sweep
- Graceful fallback: if numba isn't installed, pure-Python versions are used transparently
- `NUMBA_AVAILABLE` flag for runtime detection

> "The Simulated Annealing inner loop visits every voxel and computes a 6-neighbor energy delta. With Numba, this is compiled to machine code. The API is unchanged — it just runs faster."

---

## Feedback Point 10: Real Data Validation

**What reviewers said:** Validate on real soliton structures, not just synthetic fields.

**What we did:**
- `load_lcsim_npz()` — reads article.lcpen/LCSim format (PATH key, settings JSON)
- `examples/03_soliton_validation.py` — CLI benchmark script
- `examples/03_soliton_validation.ipynb` — interactive validation notebook
- `examples/04_v2_demo.ipynb` — the comprehensive demo notebook (what we're showing now)

**Show in notebook:** Scroll to **Section 5** ("Results Summary Table") and **Section 11** ("Noise Sensitivity").

Point out:
- The summary table with all 6 methods compared
- The noise sensitivity plot showing how accuracy degrades with noise
- GraphCuts typically achieves the highest accuracy
- Hierarchical provides the best speed/accuracy trade-off

> "We ran a controlled experiment: take a known cholesteric, scramble 50% of signs, run all 6 optimizers, measure energy recovery and sign accuracy. Then we repeated this through the full FCPM pipeline with noise levels from 1% to 10%."

---

## Closing Summary (2 minutes)

**Show in notebook:** Scroll to **Section 12** (the summary table at the bottom).

> "To summarize the 10 feedback points:"

| # | Feedback | Solution | Status |
|---|----------|----------|--------|
| 1 | Modern packaging | `pyproject.toml` + UV | Done |
| 2 | README visuals | Optimizer tables, code examples, API reference | Done |
| 3 | References | `REFERENCES.md` with 15 citations | Done |
| 4 | Architecture | `SignOptimizer` ABC, unified subpackage, deprecation shims | Done |
| 5 | HDF5 | `save/load_director_hdf5` with gzip | Done |
| 6 | Documentation | MkDocs site with 16 pages + auto-deploy | Done |
| 7 | Frank energy | Splay/twist/bend decomposition with 5CB constants | Done |
| 8 | Pointwise metrics | `sign_accuracy`, `spatial_error_distribution` | Done |
| 9 | Numba | JIT-compiled SA kernels with pure-Python fallback | Done |
| 10 | Validation | Benchmark notebook, CLI script, noise sensitivity | Done |

> "69 tests pass. All v1 imports are backward compatible. The version is 2.0.0."

---

## Anticipated Questions

**Q: Why is GraphCuts the best?**
> It solves the exact binary labeling problem via min-cut/max-flow. The sign choice is binary (+/-), and the pairwise energy is submodular, so graph cuts finds the global optimum. Other methods find local optima.

**Q: When would you NOT use GraphCuts?**
> For very large volumes where memory is a concern (it builds a full graph), or when the energy has higher-order terms that aren't captured by the pairwise formulation. In those cases, Hierarchical is a good alternative.

**Q: How much does Numba help?**
> The SA inner loop goes from ~10s to ~1s for a 64^3 volume. The improvement is proportional to volume size. For small volumes the JIT compilation overhead dominates.

**Q: Are the v2/ files deleted?**
> No, they're converted to deprecation shims. `from v2.approaches import GraphCutsOptimizer` still works but prints a DeprecationWarning. This preserves backward compatibility for any existing notebooks or scripts.

**Q: What about the existing notebooks?**
> Notebook 01 and 02 are unchanged and still work. We added 03 (validation benchmark) and 04 (comprehensive v2 demo).
