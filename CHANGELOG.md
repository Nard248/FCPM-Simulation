# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Audit Remediation (Priorities 0–5)

Addresses the external audit (March 10, 2026) findings across all 6
priorities.  See `AUDIT_REMEDIATION_PLAN.md` for the full plan.

### Fixed — Critical Optimizer Bugs

- **GraphCuts (Fix 0.1):** Removed in-loop mutation of director array during
  graph construction that corrupted edge weights for subsequent nodes.
  Replaced with a clean BFS pre-alignment pass before graph building.
- **GraphCuts (Fix 0.2):** Fixed NetworkX fallback where the sink node had no
  edges, causing the min-cut to be trivially zero.  Added sink edges for all
  non-seed voxels.
- **GraphCuts (Fix 0.3):** Updated module docstring to accurately describe
  which energy is minimised and under which constraints. Removed the
  incorrect claim of "globally optimal" without qualification.
- **GraphCuts:** Added post-optimization global sign check (n → -n) to handle
  the nematic symmetry correctly.
- **SimulatedAnnealing (Fix 0.4):** Fixed cluster move to return the actual
  energy change (was returning 0.0) and update the energy tracker after
  cluster moves.
- **SimulatedAnnealing (Fix 0.5):** Fixed adaptive temperature window: now
  resets acceptance counters to 0 instead of halving them (which was a no-op).
- **LayerPropagation (Fix 0.6):** Added energy guard to iterative refinement:
  reverts if a simultaneous multi-voxel flip increases the total gradient
  energy.
- **Hierarchical (Fix 0.6):** Same energy guard added to
  `_iterative_refinement`.
- **Hierarchical (Fix 0.7):** Fixed Q-tensor normalization in `_coarsen`:
  was dividing by `block.size` (which includes the vector dimension) instead
  of the number of vectors.
- **CombinedOptimizer (Fix 0.8):** Now extracts and reports `total_flips`
  from the V1 history dict instead of always reporting 0.
- **BeliefPropagation (Fix 0.9):** Changed `_compute_potentials` to use
  sign-invariant potentials based on `|n_i · n_j|` instead of computing
  energies from the potentially sign-inconsistent input array.

### Fixed — Tests and Metadata

- **CLI test (Fix 0.10):** Fixed `TestCLI.test_cli_help` to properly catch
  `SystemExit(0)` from argparse `--help`.
- **CLI version (Fix 0.11):** `fcpm --version` now reads from
  `fcpm.__version__` instead of a hardcoded `'fcpm 1.0.0'` string.
- **setup.py (Fix 0.11):** Synchronised version to `2.0.0`, package name to
  `fcpm-simulation`, `python_requires` to `>=3.10`, added `scipy` to core
  deps, reconciled extras with `pyproject.toml`.
- **__init__.py (Fix 0.14):** Removed duplicate imports of
  `add_gaussian_noise` and `add_poisson_noise` (were imported from both
  `fcpm.core` and `fcpm.utils`, with the second silently shadowing the first).
- **Hierarchical:** Moved multi-resolution energy values from the misleading
  `energy_by_layer` field to `metadata['energy_by_level']`.

### Added — Correctness Tests

- **test_optimizer_correctness.py (Fix 0.13):** New test file with:
  - Brute-force oracle for small fields (exhaustive search over all 2^N
    sign configurations)
  - Correctness verification: Combined and GraphCuts find the true global
    minimum on a 3×3×2 cholesteric
  - Energy monotonicity: 50 random 6×6×4 fields × 6 optimizers, verify
    `final_energy ≤ initial_energy`
  - Single-flip recovery: verify a single wrong-sign voxel is corrected
  - Idempotency: running an optimizer twice gives the same result
  - GraphCuts-specific: uniform field, fully-flipped field, checkerboard
    scramble

### Added — Planning

- **AUDIT_REMEDIATION_PLAN.md:** Complete remediation plan mapping every audit
  item to concrete code changes with 6 priority levels and 35 action items.
- **CHANGELOG.md:** This file, replacing the v2-only `CHANGELOG_V2.md`.

### Added — Priority 1: Identifiability and Reconstruction Object

- **`ReconstructionResult`** dataclass in `fcpm/reconstruction/base.py`:
  structured return type with `observable_components`, `ambiguous_components`,
  `confidence_map`, `ambiguity_mask`, and `info`. Supports tuple unpacking
  for backward compatibility.
- **`compute_confidence_map()`** in `fcpm/reconstruction/qtensor_method.py`:
  per-voxel confidence combining in-plane magnitude and Q-tensor eigen-gap.
- **`compute_ambiguity_mask()`**: flags voxels where |nz| > threshold
  (nearly vertical director, numerically unstable in-plane angle).
- **Reconstruction modes**: `reconstruct_via_qtensor()` now accepts
  `mode='full'|'observed_Q'|'line_field'|'director'`.
- **`docs/identifiability.md`**: design note on what FCPM can and cannot
  recover, three identifiability regimes, and usage of diagnostics.

### Added — Priority 2: Topology Generators and Metrics

- **`create_disclination_director()`**: line disclination with configurable
  topological charge (+1, -1, +0.5, -0.5). Metadata includes
  `has_branch_cut` for half-integer charges.
- **`create_skyrmion_director()`**: 2D baby-Skyrmion with configurable
  radius and helicity (Neel vs Bloch).
- **`create_toron_director()`**: approximate toron (localized double-twist
  cylinder on cholesteric background).
- **`energy_recovery_fraction()`**: measures what fraction of the energy gap
  (scrambled minus ground truth) was recovered by optimization.
- **`compute_branch_cut_map()`**: detects director discontinuities where
  adjacent voxels have opposing signs.

### Added — Priority 3: Noise Model and Sensitivity

- **`fcpm/noise/` subpackage** with `NoiseModel` dataclass:
  - `apply()`: physically motivated noise chain (shot noise, read noise,
    dark current, background, saturation)
  - `log_likelihood()`: approximate log-likelihood for Poisson-Gaussian model
  - `estimate_from_data()`: estimates noise parameters from repeated
    measurements via mean-variance linear fit
- **`fcpm/benchmarks/` subpackage** with sensitivity analysis:
  - `parameter_sensitivity_study()`: sweeps a parameter and measures
    reconstruction quality at each value
  - `SensitivityResult` dataclass for per-trial results

### Added — Priority 4: Systematic Distortions

- **`fcpm/distortions.py`** with 7 distortion functions:
  - `apply_polarization_offset()`: misaligned polarizer
  - `apply_intensity_drift()`: photobleaching / lamp drift
  - `apply_slice_misregistration()`: sample drift between angles
  - `apply_psf_blur()`: anisotropic optical PSF
  - `apply_background_gradient()`: non-uniform illumination
  - `apply_saturation()`: detector clipping
  - `apply_bleaching()`: progressive signal loss

### Added — Priority 5: UX, Documentation, API

- **`docs/capabilities.md`**: honest assessment of what the library can
  and cannot do, reliability by field type, optimizer comparison.
- **`docs/failure_modes.md`**: catalog of 9 known failure modes with
  symptoms, causes, diagnostics, and mitigations.
- **`docs/identifiability.md`**: added to mkdocs navigation.
- **API surface expanded**: surfaced `load_director_tiff`,
  `load_fcpm_from_directory`, `load_fcpm_image_sequence`, `SAHistory`
  at the top-level `fcpm` namespace.
- **mkdocs.yml**: fixed placeholder URLs, added 3 new docs to navigation.

---

## [2.0.0] — 2025

See `CHANGELOG_V2.md` (archived) for the full v2.0.0 release notes.

### Added
- Unified sign optimization framework with 6 optimizers
- SignOptimizer ABC and OptimizationResult dataclass
- Frank elastic energy decomposition (splay/twist/bend)
- HDF5 I/O support
- LCSim NPZ loader
- Numba-accelerated SA kernels
- MkDocs documentation site
- Modern packaging via pyproject.toml

## [1.0.0] — 2024

Initial release with basic FCPM simulation, Q-tensor reconstruction,
chain propagation sign optimization, and I/O utilities.
