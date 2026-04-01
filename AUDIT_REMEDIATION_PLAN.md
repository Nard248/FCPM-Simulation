# FCPM-Simulation: Audit Remediation Plan

**Date**: March 18, 2026
**Audit source**: `docs/fcpm_project_audit_plan_en.pdf` (March 10, 2026)
**Scope**: Every item raised in the audit, mapped to concrete code changes

---

## How to read this document

Each **Priority** corresponds to the audit's priority levels. Within each priority, items are
listed as actionable tasks with:
- **[FILE]** — which file(s) to modify or create
- **[ISSUE]** — what the audit found (confirmed by independent code review)
- **[ACTION]** — exactly what to do
- **[TEST]** — how to verify the fix

Items marked with a severity tag:
- `CRITICAL` — code produces wrong results
- `HIGH` — misleading claims or broken infrastructure
- `MEDIUM` — missing functionality that the audit requires
- `LOW` — cosmetic or organizational

---

## Priority 0: Fix Correctness, Reproducibility, and Claims

> **Audit criterion**: "One can at least trust that the code matches its own claims
> and does not fail on basic counterexamples."

### 0.1 — GraphCuts: In-loop mutation corrupts graph weights `CRITICAL`

**[ISSUE]** In `graph_cuts.py`, the maxflow path (lines ~170) and networkx path (lines ~273)
mutate `n[y,x,z] = -n[y,x,z]` inside the triple loop that builds graph edges. This means
subsequent edge weights are computed from an inconsistently pre-flipped array. The resulting
min-cut does not minimize any coherent energy function.

**[FILE]** `fcpm/reconstruction/optimizers/graph_cuts.py`

**[ACTION]**
1. Remove all `if dot < 0: n[...] = -n[...]` lines from inside the graph-building loops
   (both `_optimize_maxflow` and `_optimize_networkx`).
2. After the graph is built and the min-cut is computed, apply sign assignments purely
   from the partition labels returned by `g.get_segment()` (maxflow) or the min-cut set
   (networkx).
3. Use the following sign-assignment logic after the cut:
   ```python
   # After min-cut: segment 0 keeps sign, segment 1 flips
   for idx in range(n_nodes):
       if g.get_segment(idx) == 1:  # sink side
           y, x, z = np.unravel_index(idx, shape[:3])
           n[y, x, z] = -n[y, x, z]
   ```
4. Compare the final energy against the initial energy. If the energy increased (possible
   due to seed-voxel constraint), flip ALL signs and recheck (accounts for global n→-n).

**[TEST]**
- Exhaustive test on 3×3×3 and 4×4×4 fields with known ground truth (brute-force
  search over all 2^N sign configurations to find the true minimum).
- Property test: `final_energy <= initial_energy + eps` on 100 random fields.

### 0.2 — GraphCuts: networkx fallback sink node disconnected `CRITICAL`

**[ISSUE]** In `_optimize_networkx`, a sink node `'T'` is added to the graph but no edges
connect any voxel to it. `nx.minimum_cut(G, source, 'T')` returns a trivially zero cut
with an arbitrary partition.

**[FILE]** `fcpm/reconstruction/optimizers/graph_cuts.py`

**[ACTION]**
1. Add edges from every voxel node to the sink `'T'` with capacity based on the unary
   potential (or a small constant if no unary is used).
2. Alternatively, reformulate: the s-t min-cut for binary MRF sign optimization requires
   source edges = unary cost of label 0, sink edges = unary cost of label 1, and inter-node
   edges = pairwise disagreement cost. If `unary_weight=0` (default), use the seed-based
   approach: source→seed with INF, all other nodes→sink with 0. This still needs proper
   wiring.

**[TEST]**
- Same exhaustive test as 0.1 but specifically with `pymaxflow` uninstalled so the
  networkx fallback is triggered.

### 0.3 — GraphCuts: energy mismatch with canonical gradient energy `HIGH`

**[ISSUE]** The graph encodes `E_GC = Σ 2|n_i·n_j| · [s_i≠s_j]`, but the reported energy uses
`compute_gradient_energy` which computes `Σ |n_i - n_j|²`. These are monotonically related
for fixed `n` magnitudes but not numerically equal. The docstring claims "globally optimal
solution" for "the gradient energy", which is misleading.

**[FILE]** `fcpm/reconstruction/optimizers/graph_cuts.py`

**[ACTION]**
1. Update the module docstring to state precisely which energy the graph-cut minimizes:
   `E_GC(S) = Σ_{(i,j) adjacent} 2|n_i·n_j| · 𝟙[s_i ≠ s_j]`.
2. Add a note: "For unit vectors, the gradient energy difference caused by a sign flip is
   `4(n_i·n_j)`. The graph-cut minimizer of `E_GC` is the global minimizer of the gradient
   energy over sign configurations, subject to the seed constraint."
3. Prove this equivalence in a docstring or remove the "globally optimal" claim.
   The proof: for unit vectors, `|s_i n_i - s_j n_j|² = 2 - 2 s_i s_j (n_i·n_j)`. The sign
   configuration that minimizes `Σ (2 - 2 s_i s_j d_ij)` is the same that minimizes
   `Σ -s_i s_j d_ij` = same that minimizes `Σ_{s_i≠s_j} 2|d_ij|` when `d_ij` has been
   oriented to be non-negative. But the in-loop flip was supposed to achieve this orientation
   — except it corrupts the array. After fix 0.1, the edge weights should use `abs(dot)`
   directly without flipping, which is already the case.

### 0.4 — SimulatedAnnealing: cluster moves flip aligned voxels `CRITICAL`

**[ISSUE]** `_cluster_move` builds a cluster of voxels aligned with the seed (dot > 0) and flips
them all. This is backwards — flipping aligned voxels increases the gradient energy. The
correct Wolff-like construction should either: (a) collect anti-aligned neighbors, or (b) use
the standard Swendsen-Wang bond probability `p = 1 - exp(-2β·max(0, s_i·s_j·n_i·n_j))`.

**[FILE]** `fcpm/reconstruction/optimizers/simulated_annealing.py`

**[ACTION]**
1. Fix `_cluster_move` to use the standard Wolff algorithm for the Ising-like energy:
   - Seed: pick a random voxel, flip its sign
   - For each neighbor of a cluster member: compute bond probability
     `p = 1 - exp(-2 * beta * max(0, n_i · n_j))` where `beta = 1/T`
   - If `rand() < p`, add the neighbor to the cluster and flip it
2. Compute and return the actual `delta_e` (sum over boundary edges of the cluster).
3. Update the energy tracker: `energy += delta_e` after the cluster move.

**[TEST]**
- On a known field, verify cluster moves never increase energy at T→0.
- Property test: `delta_e` matches `compute_gradient_energy(after) - compute_gradient_energy(before)`.

### 0.5 — SimulatedAnnealing: adaptive temperature window is a no-op `MEDIUM`

**[ISSUE]** Lines 165–166 halve both `accepted` and `total_attempts`. Since `accept_rate =
accepted / total_attempts`, halving both changes nothing. The intent was to reset the window.

**[FILE]** `fcpm/reconstruction/optimizers/simulated_annealing.py`

**[ACTION]**
```python
# Replace:
accepted = max(1, accepted // 2)
total_attempts = max(1, total_attempts // 2)
# With:
accepted = 0
total_attempts = 0
```

**[TEST]** Verify that the acceptance rate is responsive: on a cooling schedule, it should start
high (~0.8) and decrease to near 0.

### 0.6 — LayerProp/Hierarchical: simultaneous flip can increase energy `MEDIUM`

**[ISSUE]** `_iterative_refinement` (used by both LayerPropagation and Hierarchical) computes
per-voxel flip benefit independently, then flips ALL beneficial voxels simultaneously. Two
adjacent voxels that each individually benefit from flipping may jointly increase the energy
of their shared edge.

**[FILE]** `fcpm/reconstruction/optimizers/layer_propagation.py`, `hierarchical.py`

**[ACTION]**
Option A (recommended): Add a post-flip energy check. After the simultaneous flip,
recompute the total gradient energy. If it increased, revert and switch to sequential
(one-voxel-at-a-time) refinement for this iteration.

Option B: Replace with sequential Gauss-Seidel scan (flip one voxel, update neighbor
costs, move to next). This guarantees monotonic decrease but is slower.

**[TEST]**
- Property test: `final_energy <= initial_energy + 1e-10` on 200 random 8×8×8 fields.
- Regression test: verify convergence on the existing cholesteric test case.

### 0.7 — Hierarchical: Q-tensor normalization off by factor 3 `LOW`

**[ISSUE]** `_coarsen` divides `Q_sum` by `block.size` which is `cf³ × 3` (total scalar elements),
not `cf³` (number of vectors). The eigenvector direction is unaffected (scalar multiple of Q
has same eigenvectors), but the magnitude is wrong.

**[FILE]** `fcpm/reconstruction/optimizers/hierarchical.py`

**[ACTION]**
```python
# Replace:
Q_avg = Q_sum / block.size
# With:
n_vectors = block.shape[0] * block.shape[1] * block.shape[2]  # cf^3
Q_avg = Q_sum / n_vectors
```

### 0.8 — CombinedOptimizer: total_flips always 0 `LOW`

**[ISSUE]** `total_flips=0` is hardcoded. The V1 functions do track flips internally.

**[FILE]** `fcpm/reconstruction/optimizers/combined.py`

**[ACTION]** Extract flip count from `iterative_local_flip`'s history dict and sum them.

### 0.9 — BeliefPropagation: potentials computed on sign-inconsistent input `MEDIUM`

**[ISSUE]** `_compute_potentials` is called once on the raw input array which may have
arbitrary signs. `energy_same = |n_i - n_j|²` and `energy_diff = |n_i + n_j|²` may be
swapped relative to their intended meaning depending on the existing sign pattern.

**[FILE]** `fcpm/reconstruction/optimizers/belief_propagation.py`

**[ACTION]**
1. Pre-align signs using a cheap pass (e.g., chain propagation) before computing potentials.
   OR
2. Use sign-invariant potentials: `energy_aligned = 2 - 2|n_i·n_j|`,
   `energy_anti = 2 + 2|n_i·n_j|`. These don't depend on the input sign convention.

### 0.10 — Test suite: bring to fully green `HIGH`

**[ISSUE]** `TestCLI.test_cli_help` fails because `argparse --help` raises `SystemExit(0)`
which is not caught.

**[FILE]** `tests/test_fcpm.py`

**[ACTION]**
```python
def test_cli_help(self):
    from fcpm.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(['--help'])
    assert exc_info.value.code == 0
```

Also add a version consistency test:
```python
def test_cli_version_matches_package(self):
    import fcpm
    # Parse the version string from cli.py or just verify they match
    assert fcpm.__version__ == "2.0.0"  # or dynamically check cli output
```

**[TEST]** `pytest tests/ -v` must show 0 failures.

### 0.11 — Synchronize package metadata `HIGH`

**[ISSUE]** Four version strings disagree. Package name differs between `setup.py` (`fcpm`)
and `pyproject.toml` (`fcpm-simulation`). Python-requires differs. Dependencies differ.

**[FILE]** `setup.py`, `fcpm/cli.py`, `fcpm/__init__.py`, `pyproject.toml`

**[ACTION]**
1. `fcpm/cli.py` line 39: replace hardcoded `'fcpm 1.0.0'` with
   `f'fcpm {fcpm.__version__}'` (import `__version__` from `fcpm`).
2. `setup.py`: update `version='2.0.0'`, `name='fcpm-simulation'`,
   `python_requires='>=3.10'`, add `scipy` to core deps. Add a prominent deprecation
   comment pointing to `pyproject.toml`.
3. Verify all 4 locations report `2.0.0`.
4. Reconcile extras between `setup.py` and `pyproject.toml` (remove `topovec` from
   `setup.py`, add `perf`/`docs` extras, synchronize versions).

### 0.12 — Clean repository of service directories and artifacts `MEDIUM`

**[ISSUE]** `.idea/` tracked (10 files), `v2/` superseded (21 files + 5 PNGs), `Archive/`
tracked despite being gitignored, 9 `v2_demo_*.png` in `examples/`, presentation/feedback
markdown files at root, `examples/benchmark-restuls-march4.xlsx` untracked but not ignored.

**[FILE]** `.gitignore`, multiple directories

**[ACTION]**
1. Create `archive/` directory (lowercase) and move into it:
   - `v2/` → `archive/v2/`
   - `Archive/` → `archive/legacy/`
   - `BENCHMARK_RESULTS.md` → `archive/`
   - `BENCHMARK_PRESENTATION.md` → `archive/`
   - `PRESENTATION_SCRIPT.md` → `archive/`
   - `FEEDBACK_RESPONSE.md` → `archive/`
   - `FEEDBACK_ANALYSIS.md` → `archive/`
   - `examples/benchmark-restuls-march4.xlsx` → `archive/`
   - `examples/v2_demo_*.png` (9 files) → `archive/demo_figures/`
2. Update `.gitignore`:
   ```
   # IDE
   .idea/
   .claude/

   # Archives and artifacts
   archive/
   gitlab-codes/
   *.xlsx
   ```
3. `git rm --cached -r .idea/` to un-track IDE files.
4. `git rm --cached -r Archive/` to enforce the existing ignore rule.
5. Move `docs/fcpm_project_audit_plan_en.pdf` → `archive/`
6. Move research notebooks:
   - `examples/05_real_data_benchmark.ipynb` → `research/`
   - `examples/06_detailed_error_analysis.ipynb` → `research/`

### 0.13 — Add exhaustive small-field tests `HIGH`

**[ISSUE]** No test verifies optimizer correctness against a known global optimum.

**[FILE]** `tests/test_optimizer_correctness.py` (new)

**[ACTION]** Create a new test file with:
1. **Brute-force oracle**: For a small field (e.g., 3×3×3 = 27 voxels → 2²⁷ sign configs,
   feasible with pruning or up to ~20 voxels for full enumeration), find the true global
   minimum of the gradient energy over all sign configurations.
2. **Exhaustive correctness tests**: For each optimizer, run on the small field and verify
   the result matches the oracle's global minimum (or is within a stated tolerance for
   stochastic optimizers).
3. **Energy monotonicity property tests**: Using `hypothesis` or manual random
   generation, verify `final_energy <= initial_energy + eps` across 200+ random fields of
   size 6×6×6.
4. **Idempotency test**: Running the optimizer twice should not change the result
   (the output is already a fixed point).

### 0.14 — Fix `__init__.py` double imports `LOW`

**[ISSUE]** `add_gaussian_noise` and `add_poisson_noise` are imported twice from different
modules, with the second import silently shadowing the first.

**[FILE]** `fcpm/__init__.py`

**[ACTION]** Remove the duplicate imports. Keep only the canonical source (whichever is the
correct public API — likely `fcpm.core.simulation` for FCPM-specific noise,
`fcpm.utils` for generic noise utilities). Document which is which.

### 0.15 — Recheck claims in README, docs, and docstrings `HIGH`

**[ISSUE]** Multiple incorrect or misleading claims identified:
- GraphCuts docstring: "globally optimal solution" — false (see 0.1, 0.3)
- `energy_by_layer` in HierarchicalOptimizer: stores multi-resolution values, not
  per-z-layer values — semantics broken
- README examples table incomplete (missing notebooks 04, 05, 06)
- README placeholder URLs (`your-repo`, `your-org`)
- `__init__.py` docstring Quick Start uses different API than README Quick Start
- BibTeX `year={2024}` is outdated

**[FILE]** Multiple

**[ACTION]**
1. Audit every docstring that claims "optimal", "exact", or "guaranteed" — qualify or remove.
2. Rename `energy_by_layer` to `energy_by_level` in `OptimizationResult` for
   HierarchicalOptimizer, or document the multi-resolution semantics.
3. Update README: add notebooks 04–06 to examples table, fix placeholder URLs.
4. Align `__init__.py` docstring Quick Start with README.
5. Fix BibTeX year.

---

## Priority 1: Formalize Identifiability and the Reconstruction Object

> **Audit criterion**: "The library stops promising more than the measurement
> physics allows."

### 1.1 — Write identifiability design note `MEDIUM`

**[FILE]** `docs/identifiability.md` (new)

**[ACTION]** Write a document covering:
1. What the standard FCPM 4-angle setup measures: `I(α) = (nx cos α + ny sin α)⁴`
2. What is exactly determinable: `Q_xx, Q_yy, Q_xy` (in-plane Q-tensor)
3. What has sign ambiguity: `Q_xz, Q_yz` (only magnitudes, not signs)
4. What is unobservable: the sign of `nz` relative to `(nx, ny)`
5. When the director is uniquely recoverable (up to global sign): smooth fields without
   half-integer defects
6. When it is not: half-integer disclinations, weak-signal regions, `nz`-dominated regions
7. Add to mkdocs navigation

### 1.2 — Add `ReconstructionResult` structured return type `MEDIUM`

**[FILE]** `fcpm/reconstruction/base.py` (extend)

**[ACTION]** Create a dataclass:
```python
@dataclass
class ReconstructionResult:
    director: DirectorField
    qtensor: QTensor
    observable_components: List[str]  # ['Q_xx', 'Q_yy', 'Q_xy']
    ambiguous_components: List[str]   # ['Q_xz', 'Q_yz']
    confidence_map: Optional[np.ndarray]  # per-voxel, 0-1
    ambiguity_mask: Optional[np.ndarray]  # boolean, True = ambiguous
    info: Dict[str, Any]
```

Modify `reconstruct_via_qtensor` and `reconstruct()` to return this type.
Keep backward compatibility by making the result unpackable as a tuple.

### 1.3 — Add confidence and ambiguity computation `MEDIUM`

**[FILE]** `fcpm/reconstruction/qtensor_method.py` (extend)

**[ACTION]**
1. **Ambiguity mask**: flag voxels where `nz² > threshold` (e.g., 0.8), meaning the
   in-plane projection is weak and the angle estimate is numerically unstable.
2. **Confidence map**: compute per-voxel confidence based on:
   - In-plane magnitude: `conf_inplane = nx² + ny²` (0 to 1)
   - Q-tensor eigen-gap: `conf_gap = (λ₁ - λ₂) / (λ₁ + eps)` — large gap = confident
   - Combined: `confidence = conf_inplane * conf_gap`
3. Populate `ReconstructionResult.confidence_map` and `.ambiguity_mask`.

### 1.4 — Separate reconstruction modes `MEDIUM`

**[FILE]** `fcpm/reconstruction/qtensor_method.py`

**[ACTION]** Add a `mode` parameter to `reconstruct_via_qtensor`:
- `mode='full'` (default): current behavior — reconstruct full Q, extract director
- `mode='observed_Q'`: return only the three observable components, zero out Q_xz/Q_yz
  (uses existing `qtensor_from_fcpm_exact`)
- `mode='line_field'`: return director without sign fixing (explicitly a line field)
- `mode='director'`: full pipeline including sign optimization

### 1.5 — Add identifiability examples `LOW`

**[FILE]** `examples/07_identifiability.ipynb` (new)

**[ACTION]** Create a notebook showing:
1. A smooth cholesteric where the director is fully recoverable
2. A field with near-vertical `nz` where in-plane angle estimation fails
3. A half-integer defect where no smooth director exists
4. Comparison of observed-Q vs full-Q vs director reconstruction for each case

---

## Priority 2: Build Complete Benchmark Suite for Topologies and Defects

> **Audit criterion**: "There is a real answer to 'where and why do the algorithms fail'."

### 2.1 — Add topology generators to the library `MEDIUM`

**[FILE]** `fcpm/core/director.py` (extend)

**[ACTION]** Add factory functions (alongside existing `create_uniform_director`,
`create_cholesteric_director`, `create_radial_director`):
```python
def create_disclination_director(shape, charge, axis='z', center=None):
    """Create a 2D disclination with given topological charge.

    Args:
        charge: +1, -1, +0.5, -0.5
        axis: disclination line direction
    """

def create_toron_director(shape, pitch, position=None):
    """Create a toron (double twist cylinder + point defect pair)."""

def create_skyrmion_director(shape, radius, helicity=0):
    """Create a 2D Skyrmion configuration."""
```

### 2.2 — Define per-topology comparison metrics `MEDIUM`

**[FILE]** `fcpm/utils/metrics.py` (extend)

**[ACTION]**
1. For smooth fields: use existing `angular_error_nematic`, `sign_accuracy`
2. For half-integer defects: add `qtensor_frobenius_error` restricted to observable
   components, and a branch-cut localization metric
3. For all: add `energy_recovery_fraction`:
   `(E_scrambled - E_optimized) / (E_scrambled - E_ground_truth)`
4. Add `compute_branch_cut_map(director)`: detect discontinuities in the director
   field (where adjacent voxels have `dot(n_i, n_j) < -threshold`)

### 2.3 — Create systematic benchmark runner `MEDIUM`

**[FILE]** `fcpm/benchmarks/runner.py` (new), `fcpm/benchmarks/__init__.py` (new)

**[ACTION]** Create a benchmark framework:
```python
class BenchmarkSuite:
    def __init__(self, configs: List[BenchmarkConfig]):
        ...

    def run_all(self, optimizers, seeds, noise_levels) -> BenchmarkReport:
        """Run all configs × optimizers × conditions."""

    def to_dataframe(self) -> pd.DataFrame:
        """Consolidated results table."""

    def save_report(self, path):
        """Save reproducible report with versions, seeds, parameters."""

# Standard benchmark suite
STANDARD_SUITE = BenchmarkSuite([
    BenchmarkConfig('uniform', create_uniform_director, (32,32,32)),
    BenchmarkConfig('cholesteric', create_cholesteric_director, (32,32,32), pitch=8),
    BenchmarkConfig('disclination_+1', create_disclination_director, (32,32,32), charge=+1),
    BenchmarkConfig('disclination_+0.5', create_disclination_director, (32,32,32), charge=+0.5),
    # ... etc for all topology classes from audit §4.1
])
```

### 2.4 — Benchmark for non-uniqueness recognition `MEDIUM`

**[FILE]** Part of the benchmark suite above

**[ACTION]**
1. For half-integer defects: verify the library does NOT claim a smooth director exists.
   Instead it should return `ambiguity_mask=True` near the defect core.
2. For weak-signal regions (`nz ≈ 1`): verify `confidence_map` drops below threshold.
3. Create explicit test: "the library should not output a confident 'smooth director'
   where such a field is topologically impossible" (audit §4.3).

---

## Priority 3: Build a Statistically Grounded Reconstruction Layer

> **Audit criterion**: "The project gains not only 'energies' but also statistically
> interpretable quality criteria."

### 3.1 — Formalize the noise model `MEDIUM`

**[FILE]** `fcpm/noise/model.py` (new), `fcpm/noise/__init__.py` (new)

**[ACTION]**
```python
@dataclass
class NoiseModel:
    shot_noise: bool = True          # Poisson photon statistics
    read_noise_std: float = 0.0      # Gaussian read noise (ADU)
    background_level: float = 0.0    # constant background offset
    dark_current: float = 0.0        # per-pixel dark current
    gain: float = 1.0                # electrons per ADU
    saturation: float = float('inf') # detector saturation level

    def apply(self, I_clean: Dict[float, np.ndarray], seed=None) -> Dict[float, np.ndarray]:
        """Apply realistic noise to clean FCPM intensities."""

    def log_likelihood(self, I_observed, I_model) -> float:
        """Compute log-likelihood of observed data given model."""

    def estimate_from_data(self, I_data) -> 'NoiseModel':
        """Estimate noise parameters from repeated measurements."""
```

### 3.2 — Noise propagation through reconstruction `MEDIUM`

**[FILE]** `fcpm/reconstruction/qtensor_method.py` (extend)

**[ACTION]** Add optional noise propagation:
1. Compute Jacobian of the `I → (nx², ny², nx·ny)` inversion
2. Propagate input noise covariance to Q-tensor component variances
3. Store per-voxel variance estimates in `ReconstructionResult.confidence_map`

### 3.3 — Sensitivity study framework `MEDIUM`

**[FILE]** `fcpm/benchmarks/sensitivity.py` (new)

**[ACTION]** Create systematic sensitivity analysis:
```python
def parameter_sensitivity_study(
    director_gt,
    param_name: str,           # 'K1', 'pitch', 'z_scale', etc.
    param_range: np.ndarray,   # values to test
    reconstruction_fn,
) -> pd.DataFrame:
    """Sweep a parameter and measure reconstruction error at each value."""
```

Test sensitivity to: Frank constants, pitch, z-axis scale, intensity calibration,
polarization angle offsets.

---

## Priority 4: Systematic Distortions and Experimental Protocol

> **Audit criterion**: "The project begins to answer not only 'how to compute'
> but also 'how to measure correctly'."

### 4.1 — Catalog of systematic distortions `MEDIUM`

**[FILE]** `fcpm/simulation/distortions.py` (new)

**[ACTION]** Implement generators for:
```python
def apply_polarization_offset(I_fcpm, offset_deg):
    """Simulate misaligned polarizer."""

def apply_intensity_drift(I_fcpm, drift_rate, axis='z'):
    """Simulate photobleaching or lamp drift."""

def apply_slice_misregistration(I_fcpm, shift_xy, per_angle=False):
    """Simulate sample drift between angle acquisitions."""

def apply_psf_blur(I_fcpm, sigma_xy, sigma_z):
    """Simulate optical point spread function."""

def apply_background_gradient(I_fcpm, gradient_direction, magnitude):
    """Simulate non-uniform illumination."""
```

### 4.2 — Distortion benchmark matrix `LOW`

**[FILE]** Part of `fcpm/benchmarks/`

**[ACTION]** For each distortion type × magnitude × optimizer, measure angular error
degradation. Report which distortions are most damaging.

---

## Priority 5: UX, Documentation, and Packaging

> **Audit criterion**: "The library becomes usable not only by its author but also
> by an external user."

### 5.1 — Stabilize API and output format `MEDIUM`

**[FILE]** `fcpm/__init__.py`, `fcpm/reconstruction/__init__.py`

**[ACTION]**
1. Decide canonical public API: prefer the high-level convenience functions
   (`simulate_fcpm`, `reconstruct`, optimizer classes) over low-level internals.
2. Mark non-public modules with `_` prefix where appropriate.
3. Ensure `__all__` in each module matches what is actually documented.
4. Surface missing exports: `load_director_tiff`, `load_fcpm_from_directory`,
   `load_fcpm_image_sequence`, `combined_v1_optimization`, `SAHistory`.

### 5.2 — Benchmark CLI runner `MEDIUM`

**[FILE]** `fcpm/cli.py` (extend)

**[ACTION]** Add subcommands:
```
fcpm benchmark --suite standard --output results/
fcpm benchmark --suite lcsim --data-dir path/to/data/ --output results/
```

### 5.3 — Pin reproducibility `MEDIUM`

**[FILE]** `fcpm/benchmarks/runner.py`

**[ACTION]**
1. Every benchmark report includes: `fcpm.__version__`, `numpy.__version__`,
   `scipy.__version__`, Python version, platform, timestamp, git commit hash.
2. All random seeds are explicit and logged.
3. Expected baseline results are stored as JSON fixtures for regression testing.

### 5.4 — User-facing "what the library can and cannot do" document `MEDIUM`

**[FILE]** `docs/capabilities.md` (new)

**[ACTION]** Write a document with three sections:
1. **What the library can do**: list each capability with confidence level
2. **What it cannot do (yet)**: list known limitations honestly
3. **In which regimes the answer is reliable**: tabular summary by field type

### 5.5 — Failure-mode catalog `LOW`

**[FILE]** `docs/failure_modes.md` (new)

**[ACTION]** Document known failure modes:
- Low signal (`nz ≈ ±1`): in-plane angle numerically unstable
- Half-integer defects: no smooth director exists
- GraphCuts on multi-domain sign patterns: single cut insufficient
- Energy minimization as denoising: can introduce bias in rapid-rotation regions
- Wrong elastic constants: regularization fights the physics
- Grid anisotropy: derivatives along z may differ from xy due to voxel aspect ratio
- Periodic boundary artifacts from `np.roll` in energy computation

---

## Execution Order and Dependencies

```
Priority 0 (Correctness) ──────────────────────────────────┐
  0.1–0.4  Fix critical optimizer bugs                      │
  0.10–0.11 Fix tests and metadata                          │
  0.12 Clean repository                                     │
  0.13 Add exhaustive tests                                 │
  0.15 Recheck claims                                       │
                                                            ▼
Priority 1 (Identifiability) ──────────────────────────────┐
  1.1 Write identifiability note                            │
  1.2–1.3 Add ReconstructionResult, confidence, ambiguity   │
  1.4 Reconstruction modes                                  │
                                                            ▼
Priority 2 (Benchmarks) ──────────────────────────────────┐
  2.1 Topology generators (depends on 1.x for metrics)     │
  2.2 Per-topology metrics                                  │
  2.3 Benchmark runner                                      │
  2.4 Non-uniqueness benchmarks                             │
                                                            ▼
Priority 3 (Statistics)    ┐                                │
  3.1 Noise model          │  Can proceed in parallel       │
  3.2 Noise propagation    │  with Priority 2               │
  3.3 Sensitivity study    ┘                                │
                                                            ▼
Priority 4 (Distortions) ─────────────────────────────────┐
  4.1 Distortion catalog                                    │
  4.2 Distortion benchmarks                                 │
                                                            ▼
Priority 5 (UX/Docs)
  5.1–5.5 API, CLI, docs, failure modes
```

---

## Changelog Protocol

All changes made under this plan will be tracked in `CHANGELOG.md` (new file, replacing
the v2-only `CHANGELOG_V2.md`). Each entry will reference the audit item number
(e.g., "Fix 0.1: GraphCuts in-loop mutation removed").

The existing `CHANGELOG_V2.md` will be moved to `archive/`.

---

## Verification Criteria (from audit §12)

The library is ready for use when ALL of these hold:

1. **Claims are correct** — README/docs/docstrings match verified behavior
2. **Non-uniqueness is described explicitly** — user knows when result is unique vs not
3. **Benchmark suite for difficult topologies** — including half-integer defects
4. **Robustness to noise and systematics** — across a matrix of scenarios
5. **Uncertainty reporting** — confidence maps and ambiguity masks
6. **Reproducibility** — external user can run benchmarks and get comparable results
