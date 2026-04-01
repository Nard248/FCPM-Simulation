# FCPM - Fluorescence Confocal Polarizing Microscopy

A Python library for simulating and reconstructing liquid crystal director fields from Fluorescence Confocal Polarizing Microscopy (FCPM) data.

## Overview

FCPM is an optical microscopy technique used to probe the 3D orientation of liquid crystal molecules. The fluorescence intensity depends on the polarization angle:

```
I(alpha) ~ [nx * cos(alpha) + ny * sin(alpha)]^4
```

This library provides a complete pipeline from simulation through reconstruction:

- **Simulation**: Generate synthetic FCPM intensity images from known director fields
- **Reconstruction**: Recover director fields from FCPM measurements using Q-tensor methods
- **Sign Optimization**: Resolve the nematic sign ambiguity (n = -n) with 6 different algorithms
- **Analysis**: Evaluate reconstruction quality with comprehensive metrics and Frank energy decomposition
- **Visualization**: Plot director fields, FCPM intensities, error maps, and convergence

## Installation

```bash
# Clone the repository
git clone https://github.com/NarekMeloworksAt/FCPM-Simulation.git
cd FCPM-Simulation

# Recommended: using UV
uv sync                        # core dependencies
uv sync --extra full           # + pymaxflow, tifffile, h5py
uv sync --extra dev            # + pytest, ruff
uv sync --extra perf           # + numba acceleration

# Alternative: using pip
pip install -e .               # core dependencies
pip install -e ".[full]"       # + optional extras
pip install -e ".[dev]"        # + development tools
```

**Requirements**: Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, Matplotlib >= 3.7

## Quick Start

```python
import fcpm

# Create a cholesteric liquid crystal director field
director = fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)

# Simulate FCPM measurements
I_fcpm = fcpm.simulate_fcpm(director)

# Reconstruct director from FCPM
director_recon, info = fcpm.reconstruct(I_fcpm, fix_signs=True)

# Evaluate reconstruction quality
metrics = fcpm.summary_metrics(director_recon, director)
print(f"Angular error: {metrics['angular_error_mean_deg']:.2f} deg")
```

## Two Main Workflows

### Workflow 1: Simulation + Reconstruction (with ground truth)

```python
config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),
    noise_level=0.03,
    noise_model='mixed',
    fix_signs=True,
    save_plots=True,
    verbose=True,
)

results = fcpm.run_simulation_reconstruction(
    director_source='path/to/director.npz',
    output_dir='output/simulation_results',
    config=config
)
print(f"Angular error: {results.metrics['angular_error_mean_deg']:.2f} deg")
```

### Workflow 2: FCPM Reconstruction (no ground truth)

```python
config = fcpm.WorkflowConfig(
    filter_sigma=0.5,
    fix_signs=True,
    save_plots=True,
    verbose=True,
)

results = fcpm.run_reconstruction(
    fcpm_source='path/to/fcpm_data.npz',
    output_dir='output/reconstruction_results',
    config=config
)
```

## Sign Optimization

Version 2.0 provides a unified framework with six algorithms for resolving the nematic sign ambiguity. All optimizers inherit from `SignOptimizer` and return an `OptimizationResult`.

```python
optimizer = fcpm.CombinedOptimizer()
result = optimizer.optimize(director, verbose=True)

print(f"Energy: {result.initial_energy:.1f} -> {result.final_energy:.1f}")
print(f"Reduction: {result.energy_reduction_pct:.1f}%")
director_fixed = result.director
```

### Available Optimizers

| Optimizer | Method | Speed | Notes |
|-----------|--------|-------|-------|
| `CombinedOptimizer` | BFS chain propagation + iterative flip | Fast | Recommended general-purpose |
| `LayerPropagationOptimizer` | Layer-by-layer z-propagation + refinement | Fast | Good for layer-structured data |
| `GraphCutsOptimizer` | Min-cut/max-flow with BFS pre-alignment | Medium | Optimal subject to seed constraint |
| `SimulatedAnnealingOptimizer` | Metropolis-Hastings + Wolff cluster moves | Slow | Can escape local minima |
| `HierarchicalOptimizer` | Multi-scale coarse-to-fine via Q-tensor | Medium | Good speed/quality balance |
| `BeliefPropagationOptimizer` | Loopy belief propagation (message passing) | Medium | **Experimental** |

### Frank Energy Analysis

```python
from fcpm import FrankConstants, compute_frank_energy_anisotropic

frank = FrankConstants(K1=10.3, K2=7.4, K3=16.48)  # 5CB defaults (pN)
energy = compute_frank_energy_anisotropic(director.to_array(), frank)

print(f"Splay: {energy['splay_integrated']:.2f}")
print(f"Twist: {energy['twist_integrated']:.2f}")
print(f"Bend:  {energy['bend_integrated']:.2f}")
```

## File Formats

```python
# Loading (auto-detection)
director = fcpm.load_director('file.npz')  # or .npy, .mat, .h5
I_fcpm = fcpm.load_fcpm('file.npz')        # or TIFF stacks, .mat

# Saving
fcpm.save_director_npz(director, 'output.npz')
fcpm.save_fcpm_npz(I_fcpm, 'output.npz')

# Export for external tools
fcpm.export_for_matlab(director, 'output.mat')
fcpm.export_for_vtk(director, 'output.vtk')
```

Supported formats: NumPy `.npz` (recommended), `.npy`, MATLAB `.mat`, TIFF stacks, HDF5, VTK.

## Preprocessing

```python
# Cropping
director_crop = fcpm.crop_director_center(director, size=(64, 64, 32))
I_crop = fcpm.crop_fcpm_center(I_fcpm, size=(64, 64, 32))

# Filtering
I_smooth = fcpm.gaussian_filter_fcpm(I_fcpm, sigma=1.0)
I_clean = fcpm.remove_background_fcpm(I_fcpm, method='percentile')
I_norm = fcpm.normalize_fcpm(I_fcpm, method='global')

# Noise (for simulations)
I_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm, noise_model='mixed', gaussian_sigma=0.03)
```

## Visualization

```python
import matplotlib.pyplot as plt

fcpm.plot_director_slice(director, z_idx=16, step=2)  # Quiver plot
fcpm.plot_fcpm_intensities(I_fcpm, z_idx=16)           # 4-panel intensities
fcpm.compare_directors(director_gt, director_recon, z_idx=16)
fcpm.plot_error_map(director_recon, director_gt, z_idx=16)
fcpm.plot_error_histogram(director_recon, director_gt)
```

## Metrics

```python
metrics = fcpm.summary_metrics(director_recon, director_gt)
# Returns: angular_error_mean_deg, angular_error_median_deg,
#          angular_error_90th_deg, angular_error_95th_deg,
#          angular_error_99th_deg, angular_error_max_deg,
#          sign_accuracy, euclidean_error_mean, ...
```

## Command Line Interface

```bash
fcpm reconstruct input_fcpm.npz -o output_dir/ -v
fcpm simulate director.npz -o fcpm.npz --noise 0.05
fcpm info data.npz
fcpm convert director.npz -o director.mat --format mat
```

## Examples

| File | Description |
|------|-------------|
| `examples/01_simulation_reconstruction.ipynb` | Full workflow with ground truth comparison |
| `examples/02_fcpm_reconstruction.ipynb` | Reconstruction from experimental FCPM data |
| `examples/03_soliton_validation.ipynb` | Sign optimization benchmark on soliton structures |
| `examples/03_soliton_validation.py` | CLI version of the validation benchmark |
| `examples/04_v2_demo.ipynb` | V2 sign optimization demo |
| `examples/05_real_data_benchmark.ipynb` | 19 LCSim structures x 6 optimizers benchmark |
| `examples/06_detailed_error_analysis.ipynb` | Deep-dive error analysis with spatial patterns |
| `examples/demo.py` | Python script demonstrating both workflows |

## Repository Structure

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for a complete map of every file and directory.

### Key directories:

```
FCPM-Simulation/
|-- fcpm/                    # The library (installable Python package)
|   |-- core/                # Director, Q-tensor, FCPM simulation
|   |-- reconstruction/      # Q-tensor reconstruction + sign optimization
|   |   |-- optimizers/      # 6 sign optimizer implementations
|   |-- visualization/       # Matplotlib plots + optional 3D
|   |-- io/                  # Loaders and exporters for all formats
|   |-- preprocessing/       # Cropping, filtering, normalization
|   |-- utils/               # Noise generation and error metrics
|   |-- pipeline.py          # High-level FCPMPipeline class
|   |-- workflows.py         # Two-workflow convenience functions
|   +-- cli.py               # Command line interface
|-- tests/                   # pytest test suite (89 tests)
|-- examples/                # Jupyter notebooks and demo scripts
|-- docs/                    # MkDocs documentation source
|-- research/                # Research notebooks (benchmarks, analysis)
|-- archive/                 # Historical artifacts and old changelogs
+-- v2/                      # Deprecated backward-compat shims for V2
```

## Physics Background

In nematic liquid crystals, the director `n(r)` describes the average molecular orientation at each point. The key symmetry is **n = -n** (the "head" and "tail" are indistinguishable). This creates a sign ambiguity that must be resolved after reconstruction.

The Q-tensor representation `Q = S * (n (x) n - I/3)` is sign-invariant (`Q(n) = Q(-n)`), making it the natural object for reconstruction. The director is then extracted as the dominant eigenvector of Q.

The gradient energy `E = sum |n_i - n_j|^2` over neighbor pairs is the cost function minimised by all sign optimizers. The anisotropic Frank energy decomposes into splay (K1), twist (K2), and bend (K3) contributions.

## Project Status

**Version 2.0.0** with ongoing audit remediation. See [CHANGELOG.md](CHANGELOG.md) for details and [AUDIT_REMEDIATION_PLAN.md](AUDIT_REMEDIATION_PLAN.md) for the full roadmap toward research-grade reliability.

## References

- **FCPM technique**: Smalyukh, Shiyanovskii & Lavrentovich (2001)
- **LC solitons**: Ackerman & Smalyukh (2017)
- **Graph cuts**: Boykov & Kolmogorov (2004)
- **Simulated annealing**: Kirkpatrick, Gelatt & Vecchi (1983)
- **Frank elasticity**: de Gennes & Prost (1993)

See [REFERENCES.md](REFERENCES.md) for full citations.

## License

MIT License

## Citation

```bibtex
@software{fcpm_simulation,
  title={FCPM: Fluorescence Confocal Polarizing Microscopy Simulation and Reconstruction},
  author={FCPM Simulation Team},
  year={2025},
  url={https://github.com/NarekMeloworksAt/FCPM-Simulation}
}
```
