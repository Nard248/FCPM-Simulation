# FCPM — Fluorescence Confocal Polarizing Microscopy

A Python library for simulating and reconstructing liquid crystal director fields
from Fluorescence Confocal Polarizing Microscopy (FCPM) data.

## Features

- **Simulation**: Generate synthetic FCPM intensity images from known director fields
- **Reconstruction**: Recover director fields from FCPM measurements using Q-tensor methods
- **Sign Optimization**: Six approaches for resolving n/-n ambiguity (graph cuts, simulated annealing, hierarchical, belief propagation, layer propagation, combined)
- **Frank Energy**: Anisotropic elastic energy decomposition (splay, twist, bend)
- **Analysis**: Comprehensive metrics including sign accuracy, angular error, spatial distributions
- **Visualization**: Director field plots, error maps, FCPM intensities
- **I/O**: NPZ, HDF5, MATLAB, TIFF, VTK formats

## Quick Example

```python
import fcpm

# Create a cholesteric liquid crystal
director = fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)

# Simulate FCPM measurements
I_fcpm = fcpm.simulate_fcpm(director)

# Reconstruct and fix signs
director_recon, info = fcpm.reconstruct(I_fcpm, fix_signs=True)

# Evaluate
metrics = fcpm.summary_metrics(director_recon, director)
print(f"Angular error: {metrics['angular_error_mean_deg']:.2f} degrees")
```

## Version

Current version: **2.0.0**

See the [Installation](installation.md) guide to get started.
