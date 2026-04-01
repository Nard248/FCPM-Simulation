# Repository Structure

Complete map of every file and directory in the FCPM-Simulation repository.

---

## Root Directory

| File | Purpose |
|------|---------|
| `README.md` | User-facing documentation: installation, quick start, workflows, API reference |
| `CHANGELOG.md` | Version history with audit remediation fixes |
| `AUDIT_REMEDIATION_PLAN.md` | Detailed remediation plan (6 priorities, 35 items) from March 2026 external audit |
| `REFERENCES.md` | 15 scientific citations covering FCPM, LC physics, and algorithms used |
| `REPOSITORY_STRUCTURE.md` | This file |
| `pyproject.toml` | **Canonical** build config (hatchling). Core deps, extras, ruff config, pytest config |
| `setup.py` | **Deprecated** legacy build file kept for backward compatibility. Points to `pyproject.toml` |
| `mkdocs.yml` | MkDocs Material documentation site configuration |
| `.gitignore` | Excludes `.idea/`, `.claude/`, `archive/`, `output/`, `gitlab-codes/`, `*.pdf`, `*.xlsx` |

---

## `fcpm/` — The Library

The installable Python package. All public API is re-exported through `fcpm/__init__.py` (116 symbols in `__all__`).

### Package Root

| File | Purpose |
|------|---------|
| `__init__.py` | Public API surface. `__version__ = '2.0.0'`. Re-exports all classes, functions, and constants from subpackages. Users only need `import fcpm` |
| `__main__.py` | Entry point for `python -m fcpm`. Delegates to `cli.main()` |
| `cli.py` | Command-line interface with 4 subcommands: `reconstruct`, `simulate`, `info`, `convert` |
| `pipeline.py` | `FCPMPipeline` class and `PipelineConfig` dataclass for stateful processing. Also `quick_reconstruct()` one-liner |
| `workflows.py` | `WorkflowConfig`, `WorkflowResults`, `run_simulation_reconstruction()`, `run_reconstruction()` — two-workflow convenience functions |
| `README.md` | Developer-facing package README |

### `fcpm/core/` — Core Data Structures and Physics

The physics kernel. All other subpackages import from here.

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports from `director`, `qtensor`, `simulation` |
| `director.py` | `DirectorField` dataclass (nx, ny, nz arrays + metadata). Factory functions: `create_uniform_director()`, `create_cholesteric_director()`, `create_radial_director()`. Also defines `DTYPE = np.float64` |
| `qtensor.py` | `QTensor` dataclass (5 independent components Q_xx through Q_yz). `director_to_qtensor()`, `qtensor_difference()`. Methods for eigendecomposition to extract directors |
| `simulation.py` | `simulate_fcpm()` — generates FCPM intensities via `I(a) = [nx*cos(a) + ny*sin(a)]^4`. Also `FCPMSimulator` class, `add_gaussian_noise()`, `add_poisson_noise()`, `compute_fcpm_observables()` |

### `fcpm/reconstruction/` — Reconstruction Algorithms

Three-step pipeline: Q-tensor reconstruction -> sign optimization -> evaluation.

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports all reconstruction functions and optimizer classes |
| `base.py` | `SignOptimizer` ABC and `OptimizationResult` dataclass. The contract all 6 optimizers implement |
| `energy.py` | `compute_gradient_energy()` — the canonical cost function (`sum \|n_i - n_j\|^2`). `FrankConstants` dataclass and `compute_frank_energy_anisotropic()` for splay/twist/bend decomposition |
| `direct.py` | Direct inversion methods: `extract_magnitudes()`, `extract_cross_term()`, `reconstruct_director_direct()` |
| `qtensor_method.py` | Q-tensor reconstruction: `qtensor_from_fcpm()`, `qtensor_from_fcpm_exact()`, `reconstruct_via_qtensor()`. Determines Q_xx, Q_yy, Q_xy exactly; Q_xz, Q_yz have sign ambiguity |
| `sign_optimization.py` | **V1** sign optimization functions (backward compatible): `chain_propagation()`, `iterative_local_flip()`, `wavefront_propagation()`, `combined_optimization()`, `gradient_energy()` |

### `fcpm/reconstruction/optimizers/` — V2 Sign Optimizers

Six concrete `SignOptimizer` subclasses, each with a config dataclass and functional interface.

| File | Class | Algorithm |
|------|-------|-----------|
| `__init__.py` | — | Re-exports all optimizer classes, configs, and functional interfaces |
| `combined.py` | `CombinedOptimizer` | V1 wrapper: BFS chain propagation + iterative local flip. Extracts `total_flips` from V1 history |
| `layer_propagation.py` | `LayerPropagationOptimizer` | Phase 1: align each z-layer with previous (O(N)). Phase 2: iterative refinement with energy guard |
| `graph_cuts.py` | `GraphCutsOptimizer` | BFS pre-alignment for submodularity, then min-cut/max-flow via PyMaxflow (or NetworkX fallback). Global-flip check for nematic symmetry |
| `simulated_annealing.py` | `SimulatedAnnealingOptimizer` | Metropolis-Hastings with adaptive temperature. Optional Wolff cluster moves with correct delta_e tracking |
| `hierarchical.py` | `HierarchicalOptimizer` | Gaussian pyramid via Q-tensor coarsening (dominant eigenvector). Optimize coarse, propagate signs up, refine at each level |
| `belief_propagation.py` | `BeliefPropagationOptimizer` | **Experimental.** Loopy BP with sign-invariant potentials based on `\|n_i . n_j\|`. Damped parallel/sequential message updates |
| `_numba_kernels.py` | — | Optional JIT-compiled kernels for SA: `sa_compute_delta_energy()`, `sa_metropolis_sweep()`. Pure-Python fallback when numba unavailable |

### `fcpm/visualization/` — Plotting

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports from `director_plot`, `topovec`, `analysis` |
| `director_plot.py` | `plot_director_slice()` (quiver), `plot_director_streamlines()`, `plot_director_rgb()`, `plot_fcpm_intensities()`, `compare_directors()`, `plot_error_map()` |
| `topovec.py` | Optional 3D visualization via TopoVec library. `visualize_3d_matplotlib()` fallback. `export_for_paraview()` for VTK export |
| `analysis.py` | `plot_error_histogram()`, `plot_error_by_depth()`, `plot_intensity_reconstruction()`, `plot_convergence()`, `plot_qtensor_components()`, `plot_order_parameter()`, `summary_statistics()` |

### `fcpm/io/` — Input/Output

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports all loaders and exporters |
| `loaders.py` | Auto-detection: `load_director()`, `load_fcpm()`. Format-specific: `load_director_npz/npy/mat/tiff/hdf5()`, `load_fcpm_npz/tiff_stack/mat()`, `load_qtensor_npz()`, `load_lcsim_npz()` (LCSim experimental format with `PATH` key), `load_simulation_results()` |
| `exporters.py` | `save_director_npz/npy/hdf5()`, `save_fcpm_npz/tiff()`, `save_qtensor_npz()`, `save_simulation_results()`, `export_for_matlab()`, `export_for_vtk()` |

### `fcpm/preprocessing/` — Data Preprocessing

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports from `cropping` and `filtering` |
| `cropping.py` | `crop_director()`, `crop_director_center()`, `crop_fcpm()`, `crop_fcpm_center()`, `crop_qtensor()`, `pad_director()`, `extract_slice()`, `subsample_director()`, `subsample_fcpm()` |
| `filtering.py` | `gaussian_filter_fcpm()`, `median_filter_fcpm()`, `bilateral_filter_fcpm()`, `remove_background_fcpm()`, `normalize_fcpm()`, `clip_fcpm()`, `smooth_director()` |

### `fcpm/utils/` — Utilities

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports from `noise` and `metrics` |
| `noise.py` | `add_gaussian_noise()`, `add_poisson_noise()`, `add_salt_pepper_noise()`, `add_fcpm_realistic_noise()`, `estimate_noise_level()`, `signal_to_noise_ratio()` |
| `metrics.py` | `angular_error_nematic()` (arccos\|n1.n2\|), `angular_error_vector()`, `euclidean_error()`, `euclidean_error_nematic()`, `intensity_reconstruction_error()`, `qtensor_frobenius_error()`, `summary_metrics()`, `perfect_reconstruction_test()`, `sign_accuracy()`, `spatial_error_distribution()` |

### `fcpm/examples/` — Built-in Examples

Runnable via `python -m fcpm.examples.basic_simulation`.

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports `run_basic_simulation`, `run_full_reconstruction` |
| `basic_simulation.py` | Create cholesteric -> simulate FCPM -> plot |
| `full_reconstruction.py` | Create -> simulate -> add noise -> reconstruct -> evaluate -> compare |

---

## `tests/` — Test Suite

89 tests. Run with `pytest tests/ -v`.

| File | Tests | Coverage |
|------|-------|----------|
| `test_fcpm.py` | 35 tests | DirectorField, QTensor, simulation, reconstruction, preprocessing, I/O, metrics, pipeline, CLI |
| `test_optimizers.py` | 35 tests | All 6 optimizers: interface compliance, energy non-increase, shape preservation, imports, backward compat, energy module, Frank energy |
| `test_optimizer_correctness.py` | 19 tests | **New (audit).** Brute-force oracle (3x3x2), energy monotonicity (50 random fields x 6 optimizers), single-flip recovery, idempotency, GraphCuts edge cases |

---

## `examples/` — Notebooks and Scripts

| File | Description |
|------|-------------|
| `01_simulation_reconstruction.ipynb` | Full workflow: create director -> simulate -> reconstruct -> evaluate with ground truth |
| `02_fcpm_reconstruction.ipynb` | Load experimental FCPM data -> preprocess -> reconstruct (no ground truth) |
| `03_soliton_validation.ipynb` | Benchmark all 6 optimizers on cholesteric/soliton structures |
| `03_soliton_validation.py` | CLI script version: `python examples/03_soliton_validation.py [--input FILE]` |
| `04_v2_demo.ipynb` | V2 sign optimization demo: all optimizers, energy comparison, Frank decomposition |
| `05_real_data_benchmark.ipynb` | 19 LCSim structures x 6 optimizers. Aggregate accuracy statistics |
| `06_detailed_error_analysis.ipynb` | 35 cells, 17 figures. Error distributions, spatial patterns, category breakdown, failure analysis |
| `demo.py` | Python script demonstrating Workflow 1 and Workflow 2 |

---

## `docs/` — MkDocs Documentation Source

Built with `mkdocs build` or deployed with `mkdocs gh-deploy`.

| File | Page |
|------|------|
| `index.md` | Landing page with feature list |
| `installation.md` | Installation guide (uv, pip, extras) |
| `quickstart.md` | Quick start code examples |
| `workflows.md` | Two-workflow explanation with output structure |
| `sign_optimization.md` | Sign ambiguity problem, all 6 optimizers with configs |
| `preprocessing.md` | Cropping, filtering, noise tools |
| `visualization.md` | All plot functions with descriptions |
| `benchmarks.md` | Optimizer comparison table, instructions to run benchmarks |
| `references.md` | Scientific references |
| `api/core.md` | Auto-generated API docs for `fcpm.core` |
| `api/reconstruction.md` | Auto-generated API docs for `fcpm.reconstruction` |
| `api/optimizers.md` | Auto-generated API docs for all 6 optimizer classes |
| `api/io.md` | Auto-generated API docs for `fcpm.io` |
| `api/visualization.md` | Auto-generated API docs for `fcpm.visualization` |
| `api/utils.md` | Auto-generated API docs for `fcpm.utils` |

---

## `research/` — Research Notebooks

Working copies of benchmark and analysis notebooks, separate from the user-facing examples.

| File | Description |
|------|-------------|
| `05_real_data_benchmark.ipynb` | Copy of the 19-dataset benchmark |
| `06_detailed_error_analysis.ipynb` | Copy of the detailed error analysis |

---

## `archive/` — Historical Artifacts

Moved here during audit remediation (March 2026). Not part of the active library.

| File/Directory | What it was |
|----------------|-------------|
| `BENCHMARK_RESULTS.md` | V2 benchmark summary document |
| `BENCHMARK_PRESENTATION.md` | Presentation-ready benchmark results |
| `PRESENTATION_SCRIPT.md` | Presentation talking points |
| `FEEDBACK_RESPONSE.md` | Response to external feedback |
| `FEEDBACK_ANALYSIS.md` | Analysis of external feedback |
| `CHANGELOG_V2.md` | V2-only changelog (superseded by root `CHANGELOG.md`) |
| `fcpm_project_audit_plan_en.pdf` | The March 2026 external audit document |
| `demo_figures/` | V2 demo PNG screenshots |

---

## `v2/` — Deprecated V2 Development Directory

Contains **backward-compatibility shims** that re-export from `fcpm.reconstruction.optimizers`. Imports emit `DeprecationWarning`. All new code lives in `fcpm/`.

| File | Purpose |
|------|---------|
| `__init__.py` | Deprecation shim: re-exports `OptimizationResult`, `compute_gradient_energy` |
| `approaches/__init__.py` | Deprecation shim: re-exports all 4 advanced optimizer classes |
| `approaches/*.py` | Original implementations (superseded, not maintained) |
| `sign_optimization_v2.py` | Original layer-by-layer implementation |
| `benchmark_approaches.py` | Benchmark script for V2 approaches |
| `*.ipynb` | Development notebooks |
| `README.md`, `APPROACHES.md` | V2 documentation |

---

## `Archive/` — Full Development History

Complete history of all development iterations. Excluded from git (in `.gitignore`). Useful for understanding project evolution.

| Directory | Era | Contents |
|-----------|-----|----------|
| `Cholesteric Liquid Crystals/` | Iteration 1 | Original simulation and reconstruction scripts |
| `Clean Simulation/` | Iteration 2 | Cleaner code with notebooks, includes real FCPM data (`S1HighResol_6-10s.npz`) |
| `Enhanced Simulator Cholesteric LC/` | Iteration 3 | Enhanced simulator with experiment runner, toron support |
| `Iteration 3: Qtensor/` | Iteration 3b | Q-tensor formulation development |
| `Docs/` | All eras | Reference PDFs, analysis documents |
| `Images/` | All eras | Publication figure generators |

---

## `.github/workflows/`

| File | Purpose |
|------|---------|
| `docs.yml` | GitHub Actions: deploy MkDocs documentation on push to main |

---

## Hidden/Tooling (not tracked in git)

| Directory | Purpose |
|-----------|---------|
| `.venv/` | Python 3.11 virtual environment |
| `.idea/` | PyCharm IDE settings |
| `.serena/` | Serena AI assistant cache |
| `.claude/` | Claude Code settings |
| `output/` | Generated workflow outputs (data, plots, summaries) |
| `gitlab-codes/` | External LCSim simulation data (separate git repo) |
