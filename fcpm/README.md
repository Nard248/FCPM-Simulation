# FCPM - Fluorescence Confocal Polarizing Microscopy

A Python library for simulating and reconstructing liquid crystal director fields from Fluorescence Confocal Polarizing Microscopy (FCPM) data.

## Installation

```bash
# Basic installation
pip install -e .

# With all optional dependencies
pip install -e ".[full]"

# With TopoVec visualization
pip install -e ".[topovec]"
```

## Quick Start

```python
import fcpm

# Create a cholesteric director field
director = fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)

# Simulate FCPM measurements
I_fcpm = fcpm.simulate_fcpm(director)

# Reconstruct director from FCPM (one-line convenience function)
director_recon, info = fcpm.reconstruct(I_fcpm)

# Or step-by-step:
director_recon, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)
director_fixed, opt_info = fcpm.combined_optimization(director_recon)

# Evaluate reconstruction quality
metrics = fcpm.summary_metrics(director_fixed, director)
print(f"Mean angular error: {metrics['angular_error_mean_deg']:.2f}°")

# Visualize
fcpm.plot_director_slice(director_fixed, z_idx=16)
```

## Physics Background

FCPM uses polarized two-photon excitation to probe the local molecular orientation in liquid crystals. The fluorescence intensity depends on the angle between the polarization direction and the director:

```
I(α) ∝ [nx·cos(α) + ny·sin(α)]⁴
```

where α is the polarization angle and (nx, ny, nz) is the director.

### Key Features

- **Q-Tensor Approach**: Eliminates sign ambiguity since Q(n) = Q(-n)
- **Sign Optimization**: BFS-based propagation and iterative local flipping
- **Nematic-Aware Metrics**: Proper handling of n ≡ -n symmetry

## Package Structure

```
fcpm/
├── core/           # Director field, Q-tensor, simulation
├── reconstruction/ # Reconstruction algorithms
├── visualization/  # Plotting and TopoVec integration
├── io/            # Data loading and saving
├── utils/         # Noise, metrics utilities
└── examples/      # Example scripts
```

## Core Classes

### DirectorField

```python
# Create from arrays
director = fcpm.DirectorField(nx=nx_array, ny=ny_array, nz=nz_array)

# Create synthetic structures
director = fcpm.create_cholesteric_director(shape, pitch=10)
director = fcpm.create_radial_director(shape)
director = fcpm.create_uniform_director(shape, direction=(0, 0, 1))

# Properties
director.shape      # (ny, nx, nz)
director.is_normalized()
director.magnitude()

# Conversion
arr = director.to_array()  # Shape: (..., 3)
director = fcpm.DirectorField.from_array(arr)
```

### QTensor

```python
# Convert director to Q-tensor
Q = fcpm.director_to_qtensor(director, S=1.0)

# Extract director from Q-tensor
director = Q.to_director_vectorized()

# Properties
Q.shape
Q.Q_xx, Q.Q_yy, Q.Q_xy, Q.Q_xz, Q.Q_yz
Q.Q_zz  # Computed from tracelessness
Q.scalar_order_parameter()
```

## Reconstruction Methods

### Direct Inversion

```python
# Extract component magnitudes
nx_sq, ny_sq, nz_sq = fcpm.extract_magnitudes(I_fcpm)

# Extract signed cross term (nx·ny)
nx_ny = fcpm.extract_cross_term(I_fcpm)

# Full direct reconstruction
director = fcpm.reconstruct_director_direct(I_fcpm)
```

### Q-Tensor Method (Recommended)

```python
# Q-tensor reconstruction
director, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)

# Sign optimization
director_fixed, opt_info = fcpm.combined_optimization(director)
```

### Sign Optimization Algorithms

```python
# Chain propagation (BFS from center)
director = fcpm.chain_propagation(director)

# Iterative local flipping
director, info = fcpm.iterative_local_flip(director, max_iter=100)

# Wavefront propagation
director = fcpm.wavefront_propagation(director, direction='z+')

# Combined (chain + iterative)
director, info = fcpm.combined_optimization(director)
```

## Visualization

```python
# 2D slice visualization
fcpm.plot_director_slice(director, z_idx=16, step=2, color_by='angle')
fcpm.plot_director_rgb(director, z_idx=16)
fcpm.plot_fcpm_intensities(I_fcpm, z_idx=16)

# Comparison
fcpm.compare_directors(director_gt, director_recon, z_idx=16)
fcpm.plot_error_map(director_recon, director_gt, z_idx=16)

# Analysis plots
fcpm.plot_error_histogram(director_recon, director_gt)
fcpm.plot_error_by_depth(director_recon, director_gt)
fcpm.plot_qtensor_components(Q, z_idx=16)

# 3D visualization (with TopoVec)
if fcpm.check_topovec_available():
    fcpm.visualize_topovec(director)

# Fallback 3D (matplotlib)
fcpm.visualize_3d_matplotlib(director, subsample=3)
```

## I/O Operations

```python
# Save/load director
fcpm.save_director_npz(director, 'director.npz')
director = fcpm.load_director_npz('director.npz')

# Save/load FCPM intensities
fcpm.save_fcpm_npz(I_fcpm, 'intensities.npz')
I_fcpm = fcpm.load_fcpm_npz('intensities.npz')

# Export for other tools
fcpm.export_for_matlab(director, 'director.mat')
fcpm.export_for_vtk(director, 'director.vtk')
fcpm.export_for_paraview(director, 'director.vts')
```

## Error Metrics

```python
# Nematic-aware angular error (accounts for n ≡ -n)
error = fcpm.angular_error_nematic(director_recon, director_gt)

# Intensity reconstruction error (definitive test)
I_recon = fcpm.simulate_fcpm(director_recon)
intensity_error = fcpm.intensity_reconstruction_error(I_original, I_recon)

# Complete summary
metrics = fcpm.summary_metrics(director_recon, director_gt, I_original, I_recon)
```

## Noise Simulation

```python
# Add Gaussian noise
I_noisy = fcpm.add_gaussian_noise(I_fcpm, sigma=0.05)

# Add Poisson (shot) noise
I_noisy = fcpm.add_poisson_noise(I_fcpm, photon_count=1000)

# Realistic mixed noise
I_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm, noise_model='mixed')
```

## Examples

Run example scripts:

```bash
# Basic simulation
python -m fcpm.examples.basic_simulation

# Full reconstruction pipeline
python -m fcpm.examples.full_reconstruction
```

## Theory Notes

### What FCPM Can Determine

From FCPM intensities at 4 angles [0, π/4, π/2, 3π/4]:
- **nx²** from I(0)
- **ny²** from I(π/2)
- **nx·ny** (with sign!) from I(π/4) - I(3π/4)
- **nz²** from unit constraint

### What FCPM Cannot Determine

- Individual signs of nx, ny, nz
- Q_xz, Q_yz (only magnitudes, not signs)

### Resolution

The Q-tensor approach handles this elegantly:
- Q(n) = Q(-n), so eigendecomposition gives the correct physics
- Sign ambiguity is resolved by spatial propagation for visualization

## License

MIT License
