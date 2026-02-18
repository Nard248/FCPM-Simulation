# Workflows

The library provides two high-level workflow functions that handle the full pipeline.

## Simulation + Reconstruction

When you have a known director field (ground truth):

```python
import fcpm

config = fcpm.WorkflowConfig(
    crop_size=(64, 64, 32),
    noise_level=0.03,
    noise_model='mixed',    # 'gaussian', 'poisson', or 'mixed'
    noise_seed=42,
    fix_signs=True,
    save_plots=True,
    save_data=True,
    verbose=True,
)

results = fcpm.run_simulation_reconstruction(
    director_source='director.npz',
    output_dir='output/',
    config=config,
)
```

## FCPM Reconstruction

When you have experimental FCPM data (no ground truth):

```python
results = fcpm.run_reconstruction(
    fcpm_source='fcpm_data.npz',
    output_dir='output/',
    config=config,
)
```

## Output Structure

```
output/
├── data/
│   ├── director_reconstructed.npz
│   ├── director_ground_truth.npz   # simulation only
│   ├── qtensor.npz
│   └── fcpm_input.npz
├── plots/
│   ├── director_recon_z*.png
│   ├── comparison_z*.png           # simulation only
│   ├── error_map_z*.png            # simulation only
│   └── error_histogram.png         # simulation only
├── summary.json
└── summary.txt
```

## Configuration Reference

::: fcpm.workflows.WorkflowConfig
    options:
      show_root_heading: true
      members: false
