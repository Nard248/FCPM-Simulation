# Quick Start

## Workflow 1: Simulation + Reconstruction

Use this when you have a known director field and want to test reconstruction.

```python
import fcpm

config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),
    noise_level=0.03,
    noise_model='mixed',
    fix_signs=True,
    verbose=True,
)

results = fcpm.run_simulation_reconstruction(
    director_source='path/to/director.npz',
    output_dir='output/',
    config=config,
)

print(f"Angular error: {results.metrics['angular_error_mean_deg']:.2f} degrees")
```

## Workflow 2: FCPM Reconstruction

Use this when you have experimental FCPM data.

```python
import fcpm

config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),
    filter_sigma=0.5,
    fix_signs=True,
    verbose=True,
)

results = fcpm.run_reconstruction(
    fcpm_source='path/to/fcpm_data.npz',
    output_dir='output/',
    config=config,
)
```

## Step-by-Step

```python
import fcpm

# 1. Create or load director
director = fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)

# 2. Simulate FCPM
I_fcpm = fcpm.simulate_fcpm(director)

# 3. Add noise
I_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm, gaussian_sigma=0.03, seed=42)
I_noisy = fcpm.normalize_fcpm(I_noisy)

# 4. Reconstruct
director_recon, info = fcpm.reconstruct(I_noisy, fix_signs=True)

# 5. Evaluate
metrics = fcpm.summary_metrics(director_recon, director)
print(f"Mean angular error: {metrics['angular_error_mean_deg']:.2f} degrees")
```
