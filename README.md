# FCPM - Fluorescence Confocal Polarizing Microscopy

A Python library for simulating and reconstructing liquid crystal director fields from Fluorescence Confocal Polarizing Microscopy (FCPM) data.

## Overview

FCPM is an optical microscopy technique used to probe the 3D orientation of liquid crystal molecules. This library provides:

- **Simulation**: Generate synthetic FCPM intensity images from known director fields
- **Reconstruction**: Recover director fields from FCPM measurements using Q-tensor methods
- **Analysis**: Evaluate reconstruction quality with comprehensive metrics
- **Visualization**: Plot director fields, FCPM intensities, and error maps

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/FCPM-Simulation.git
cd FCPM-Simulation

# Install dependencies
pip install numpy scipy matplotlib

# Optional: for MATLAB file support
pip install scipy

# Optional: for TIFF file support
pip install tifffile
```

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
print(f"Angular error: {metrics['angular_error_mean_deg']:.2f}°")
```

## Two Main Workflows

The library supports two primary use cases:

### Workflow 1: Simulation + Reconstruction (with ground truth)

Use this when you have a known director field and want to test reconstruction quality.

```python
import fcpm

# Configure the workflow
config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),  # Crop to manageable size
    noise_level=0.03,         # Add realistic noise
    noise_model='mixed',      # Gaussian + Poisson noise
    fix_signs=True,           # Apply sign optimization
    save_plots=True,          # Generate visualization plots
    save_data=True,           # Save NPZ data files
    verbose=True,             # Print progress
)

# Run the full pipeline
results = fcpm.run_simulation_reconstruction(
    director_source='path/to/director.npz',
    output_dir='output/simulation_results',
    config=config
)

# Access results
print(f"Angular error: {results.metrics['angular_error_mean_deg']:.2f}°")
print(f"Shape: {results.director_recon.shape}")
```

### Workflow 2: FCPM Reconstruction (no ground truth)

Use this when you have experimental FCPM data and want to reconstruct the director field.

```python
import fcpm

config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),  # Optional cropping
    filter_sigma=0.5,         # Gaussian smoothing for noisy data
    fix_signs=True,
    save_plots=True,
    save_data=True,
    verbose=True,
)

results = fcpm.run_reconstruction(
    fcpm_source='path/to/fcpm_data.npz',
    output_dir='output/reconstruction_results',
    config=config
)

# Access results (no ground truth comparison available)
print(f"Gradient energy: {results.metrics['gradient_energy']:.2f}")
print(f"Intensity RMSE: {results.metrics['intensity_rmse_mean']:.2e}")
```

## Output Directory Structure

Both workflows save results to the specified output directory:

```
output_dir/
├── data/
│   ├── director_reconstructed.npz   # Reconstructed director field
│   ├── director_ground_truth.npz    # Ground truth (workflow 1 only)
│   ├── qtensor.npz                  # Q-tensor field
│   ├── fcpm_input.npz               # Input FCPM data
│   └── fcpm_clean.npz               # Clean FCPM (workflow 1 only)
├── plots/
│   ├── fcpm_z*.png                  # FCPM intensity slices
│   ├── director_recon_z*.png        # Reconstructed director slices
│   ├── comparison_z*.png            # Side-by-side comparison (workflow 1)
│   ├── error_map_z*.png             # Angular error maps (workflow 1)
│   ├── error_histogram.png          # Error distribution (workflow 1)
│   └── qtensor_components_z*.png    # Q-tensor visualization
├── summary.json                     # Machine-readable results
└── summary.txt                      # Human-readable summary
```

## Supported File Formats

### Loading Director Fields

```python
# Auto-detection (recommended)
director = fcpm.load_director('file.npz')  # or .npy, .mat

# Specific formats
director = fcpm.load_director_npz('file.npz')
director = fcpm.load_director_npy('file.npy')
director = fcpm.load_director_mat('file.mat')
```

Supported NPZ keys: `director`, `n`, `field`, `data`, `PATH` (experimental format)

### Loading FCPM Data

```python
# Auto-detection
I_fcpm = fcpm.load_fcpm('file.npz')

# Specific formats
I_fcpm = fcpm.load_fcpm_npz('file.npz')
I_fcpm = fcpm.load_fcpm_tiff_stack('directory/', angles=[0, 0.785, 1.571, 2.356])
I_fcpm = fcpm.load_fcpm_mat('file.mat')
```

### Saving Results

```python
# Director fields
fcpm.save_director_npz(director, 'output.npz')
fcpm.save_director_npy(director, 'output.npy')

# FCPM data
fcpm.save_fcpm_npz(I_fcpm, 'output.npz')
fcpm.save_fcpm_tiff(I_fcpm, 'output_dir/')

# For external tools
fcpm.export_for_matlab(director, 'output.mat')
fcpm.export_for_vtk(director, 'output.vtk')
```

## Preprocessing Tools

### Cropping

```python
# Crop to specific region
director_crop = fcpm.crop_director(director,
    y_range=(100, 200),
    x_range=(100, 200),
    z_range=(10, 30))

# Crop from center
director_crop = fcpm.crop_director_center(director, size=(64, 64, 32))

# Same for FCPM data
I_fcpm_crop = fcpm.crop_fcpm_center(I_fcpm, size=(64, 64, 32))
```

### Filtering

```python
# Gaussian smoothing
I_fcpm_smooth = fcpm.gaussian_filter_fcpm(I_fcpm, sigma=1.0)

# Median filter (good for salt-and-pepper noise)
I_fcpm_filtered = fcpm.median_filter_fcpm(I_fcpm, size=3)

# Background removal
I_fcpm_clean = fcpm.remove_background_fcpm(I_fcpm, method='percentile')

# Normalization
I_fcpm_norm = fcpm.normalize_fcpm(I_fcpm, method='global')
```

### Adding Noise (for simulations)

```python
# Realistic mixed noise (Gaussian + Poisson)
I_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm,
    noise_model='mixed',
    gaussian_sigma=0.03,
    seed=42)

# Individual noise types
I_noisy = fcpm.add_gaussian_noise(I_fcpm, sigma=0.05)
I_noisy = fcpm.add_poisson_noise(I_fcpm, scale=100)
```

## Creating Test Director Fields

```python
# Uniform director (tilted)
director = fcpm.create_uniform_director(
    shape=(64, 64, 32),
    direction=(0.3, 0.4, 0.87)  # (nx, ny, nz) - will be normalized
)

# Cholesteric (helical structure)
director = fcpm.create_cholesteric_director(
    shape=(64, 64, 32),
    pitch=8.0,        # helix pitch in voxels
    axis='z'          # helix axis
)

# Radial (point defect)
director = fcpm.create_radial_director(
    shape=(64, 64, 32),
    center=(32, 32, 16)
)
```

## Visualization

```python
import matplotlib.pyplot as plt

# Director field slice (quiver plot)
fig = plt.figure()
ax = fig.add_subplot(111)
fcpm.plot_director_slice(director, z_idx=16, step=2, ax=ax)
plt.show()

# FCPM intensities
fig = fcpm.plot_fcpm_intensities(I_fcpm, z_idx=16)
plt.show()

# Compare ground truth vs reconstructed
fig = fcpm.compare_directors(director_gt, director_recon, z_idx=16)
plt.show()

# Error map
fig = plt.figure()
ax = fig.add_subplot(111)
fcpm.plot_error_map(director_recon, director_gt, z_idx=16, ax=ax)
plt.show()

# Error histogram
fig = fcpm.plot_error_histogram(director_recon, director_gt)
plt.show()
```

## Metrics and Evaluation

```python
# Full summary metrics
metrics = fcpm.summary_metrics(director_recon, director_gt)
print(f"Angular error (mean):   {metrics['angular_error_mean_deg']:.2f}°")
print(f"Angular error (median): {metrics['angular_error_median_deg']:.2f}°")
print(f"Angular error (max):    {metrics['angular_error_max_deg']:.2f}°")
print(f"Intensity RMSE:         {metrics['intensity_rmse_mean']:.4f}")

# Individual metrics
ang_err = fcpm.angular_error_nematic(director_recon, director_gt)  # Per-voxel
print(f"Mean angular error: {np.mean(ang_err) * 180/np.pi:.2f}°")

# Intensity reconstruction error
I_recon = fcpm.simulate_fcpm(director_recon)
err = fcpm.intensity_reconstruction_error(I_fcpm, I_recon)
print(f"RMSE: {err['rmse_mean']:.4f}")
```

## Command Line Interface

```bash
# Reconstruct from FCPM data
python -m fcpm reconstruct input_fcpm.npz -o output_dir/

# Simulate FCPM from director
python -m fcpm simulate director.npz -o fcpm_output.npz --noise 0.05

# Get file info
python -m fcpm info data.npz
```

## Physics Background

FCPM uses polarized two-photon excitation to probe local molecular orientation. The fluorescence intensity depends on the angle between the polarization direction and the director:

```
I(α) ∝ [nx·cos(α) + ny·sin(α)]⁴
```

where:
- `α` is the polarization angle (typically 0°, 45°, 90°, 135°)
- `(nx, ny, nz)` is the local director (unit vector)

The Q-tensor representation `Q = n⊗n - I/3` is used for reconstruction because it eliminates the sign ambiguity inherent in nematic liquid crystals (n ≡ -n).

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `DirectorField` | 3D director field container with nx, ny, nz components |
| `QTensor` | Q-tensor field representation (5 independent components) |
| `FCPMSimulator` | FCPM intensity simulation engine |

### Main Functions

| Function | Description |
|----------|-------------|
| `simulate_fcpm(director)` | Generate FCPM intensities from director |
| `reconstruct(I_fcpm)` | Full reconstruction pipeline |
| `reconstruct_via_qtensor(I_fcpm)` | Q-tensor based reconstruction |
| `combined_optimization(director)` | Sign ambiguity resolution |

### Workflow Functions

| Function | Description |
|----------|-------------|
| `run_simulation_reconstruction(...)` | Full pipeline with ground truth |
| `run_reconstruction(...)` | Reconstruction from FCPM data only |

### Configuration

| Class | Description |
|-------|-------------|
| `WorkflowConfig` | Configuration for workflow functions |
| `WorkflowResults` | Results container with metrics and data |

## Examples

The `examples/` directory contains:

| File | Description |
|------|-------------|
| `01_simulation_reconstruction.ipynb` | Full workflow with ground truth comparison |
| `02_fcpm_reconstruction.ipynb` | Reconstruction from experimental FCPM data |
| `demo.py` | Python script demonstrating both workflows |

Run the demo script:
```bash
python examples/demo.py
```

Or open the Jupyter notebooks:
```bash
jupyter notebook examples/
```

Results are saved to `output/` with data files, plots, and summaries

## Troubleshooting

### "No valid angle data found"

Your file contains a director field, not FCPM data. Use `fcpm.load_director()` instead of `fcpm.load_fcpm()`.

### High angular errors

1. Try increasing noise filtering: `filter_sigma=1.0`
2. Ensure proper normalization: `fcpm.normalize_fcpm(I_fcpm)`
3. Check if data needs cropping to remove edge artifacts

### Memory issues with large datasets

Use cropping to process smaller regions:
```python
director_crop = fcpm.crop_director_center(director, size=(64, 64, 32))
```

## License

MIT License

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fcpm_simulation,
  title={FCPM: Fluorescence Confocal Polarizing Microscopy Simulation and Reconstruction},
  author={FCPM Simulation Team},
  year={2024},
  url={https://github.com/your-repo/FCPM-Simulation}
}
```
