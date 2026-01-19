# FCPM Experiment Runner - User Guide

## Overview

The `run_experiment.py` script provides a unified interface to run complete FCPM toron simulations with organized outputs. Each experiment run creates a timestamped folder containing all results, configurations, and visualizations.

## Quick Start

### 1. Run with Preset Configuration (Recommended)

```bash
python run_experiment.py --preset t3-1-standard
```

This will:
- Generate a T3-1 toron structure (48×48×48 grid)
- Run optical simulation with crossed polarizers
- Create all visualizations
- Save everything to `experiments/T3-1_Standard_Resolution_YYYYMMDD_HHMMSS/`

### 2. List Available Presets

```bash
python run_experiment.py --list-presets
```

Available presets:
- `t3-1-standard` - Standard T3-1 toron (2 point defects)
- `t3-1-highres` - High-resolution T3-1 (64³ grid)
- `t3-2-standard` - T3-2 toron (1 point + 1 ring)
- `t3-3-standard` - T3-3 toron (2 rings)
- `uniform-helical` - Uniform helical structure (baseline)
- `comparison-suite` - Standard parameters for comparisons

### 3. Run with Custom Config File

```bash
python run_experiment.py --config example_configs/custom_t3-1.json
```

### 4. Interactive Mode

```bash
python run_experiment.py
```

Follow the prompts to configure your experiment.

### 5. Custom Output Directory

```bash
python run_experiment.py --preset t3-1-standard --output-dir my_results/test1
```

## Output Structure

Each experiment creates a folder with:

```
experiments/
└── T3-1_Standard_Resolution_20251023_143022/
    ├── config.json                    # Experiment configuration
    ├── REPORT.md                      # Markdown summary report
    ├── director_field.npz             # 3D director field (lcsim-compatible)
    ├── fcpm_intensity.npy             # Raw FCPM intensity data
    ├── 00_comprehensive.png           # 9-panel comprehensive figure
    ├── 01_director_field.png          # 6-panel director visualization
    ├── 02_cross_sections.png          # Multiple Z-slices
    ├── 03_isosurfaces.png             # Level surfaces (4 views)
    └── 04_optical_simulation.png      # Optical analysis (6 panels)
```

## Configuration Files

### Creating a Custom Configuration

Create a JSON file with your parameters:

```json
{
  "experiment_name": "My Custom Experiment",
  "description": "Testing larger toron radius",

  "structure_type": "t3-1",
  "n_x": 48,
  "n_y": 48,
  "n_z": 48,
  "x_size": 6.0,
  "y_size": 6.0,
  "z_size": 6.0,
  "pitch": 2.0,
  "toron_radius": 2.5,
  "toron_center": [3.0, 3.0, 3.0],

  "wavelength": 0.55,
  "no": 1.5,
  "ne": 1.7,
  "polarizer_angle": 0.0,
  "analyzer_angle": 1.5707963267948966,

  "generate_director_field_viz": true,
  "generate_cross_sections": true,
  "generate_isosurfaces": true,
  "generate_optical_simulation": true,
  "generate_comprehensive_figure": true,

  "save_npz": true,
  "save_raw_intensity": true,
  "dpi": 150
}
```

### Configuration Parameters

#### Structure Parameters
- `structure_type`: `'t3-1'`, `'t3-2'`, `'t3-3'`, `'uniform_vertical'`, `'uniform_helical'`
- `n_x`, `n_y`, `n_z`: Grid resolution (32 = fast, 48 = standard, 64 = high quality)
- `x_size`, `y_size`, `z_size`: Physical dimensions in microns
- `pitch`: Cholesteric pitch in microns
- `toron_radius`: Toron radius in microns (for toron structures)
- `toron_center`: `[x, y, z]` center position in microns

#### Optical Parameters
- `wavelength`: Light wavelength in microns (0.55 = green)
- `no`: Ordinary refractive index
- `ne`: Extraordinary refractive index
- `polarizer_angle`: Input polarizer angle in radians
- `analyzer_angle`: Output analyzer angle in radians (π/2 = crossed)
- `add_noise`: Boolean - add realistic noise
- `noise_level`: Noise amplitude (if enabled)

#### Visualization Control
- `generate_director_field_viz`: 6-panel director field visualization
- `generate_cross_sections`: Multiple Z-slice views
- `generate_isosurfaces`: Level surface rendering
- `generate_optical_simulation`: Optical analysis plots
- `generate_comprehensive_figure`: 9-panel Nature Materials style figure

#### Output Control
- `save_npz`: Save director field in lcsim-compatible NPZ format
- `save_raw_intensity`: Save raw FCPM intensity as .npy
- `dpi`: Image resolution (150 recommended, 300 for publication)

## Example Workflows

### Systematic Parameter Study

```bash
# Study effect of toron radius
python run_experiment.py --config configs/toron_r1.5.json
python run_experiment.py --config configs/toron_r2.0.json
python run_experiment.py --config configs/toron_r2.5.json
python run_experiment.py --config configs/toron_r3.0.json
```

### Compare All Toron Types

```bash
python run_experiment.py --preset t3-1-standard
python run_experiment.py --preset t3-2-standard
python run_experiment.py --preset t3-3-standard
```

### High-Resolution Publication Figure

```bash
# Edit config to set dpi: 300 and higher resolution
python run_experiment.py --config configs/publication_quality.json
```

### Quick Test (Fast Execution)

```bash
# Use lower resolution for testing
python run_experiment.py --config example_configs/custom_t3-1.json
```

## Tips & Best Practices

### 1. Resolution Guidelines

| Resolution | Grid Size | Use Case | Time |
|------------|-----------|----------|------|
| Low | 32³ | Quick tests | ~10s |
| Standard | 48³ | General use | ~30s |
| High | 64³ | Publication | ~2min |
| Ultra | 96³ | Special cases | ~10min |

### 2. Organizing Experiments

Create a logical folder structure:

```
experiments/
├── baseline/
│   ├── uniform_helical_YYYYMMDD_HHMMSS/
│   └── uniform_vertical_YYYYMMDD_HHMMSS/
├── toron_radius_study/
│   ├── t3-1_r1.5_YYYYMMDD_HHMMSS/
│   ├── t3-1_r2.0_YYYYMMDD_HHMMSS/
│   └── t3-1_r2.5_YYYYMMDD_HHMMSS/
└── wavelength_study/
    ├── t3-1_lambda450_YYYYMMDD_HHMMSS/
    ├── t3-1_lambda550_YYYYMMDD_HHMMSS/
    └── t3-1_lambda650_YYYYMMDD_HHMMSS/
```

Use `--output-dir` to control placement:

```bash
python run_experiment.py --preset t3-1-standard --output-dir experiments/baseline/t3-1_standard
```

### 3. Documentation

Each experiment automatically generates:
- `config.json` - Full configuration (can be reused with `--config`)
- `REPORT.md` - Human-readable summary with statistics

### 4. Reproducibility

To exactly reproduce an experiment:

```bash
# The config.json from any experiment can be reused
python run_experiment.py --config experiments/old_experiment_20251020_123456/config.json
```

### 5. Batch Processing

Create a bash script for multiple runs:

```bash
#!/bin/bash
# run_parameter_sweep.sh

for radius in 1.5 2.0 2.5 3.0; do
    echo "Running with radius=$radius"
    python run_experiment.py --config "configs/toron_r${radius}.json"
done
```

## Integration with External Tools

### Use with lcsim/optics.py

The generated NPZ files are compatible:

```bash
# Your NPZ files work with the reference implementation
python /path/to/lcsim/bin/optics.py experiments/my_exp_*/director_field.npz
```

### Use with topovec

Load director fields in Jupyter:

```python
import numpy as np

# Load your experiment data
data = np.load('experiments/my_exp_20251023_*/director_field.npz')
director = data['PATH'][0]  # Shape: (nx, ny, nz, 3)

# Now use topovec visualization tools
```

### Post-Processing

Load intensity data for custom analysis:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load FCPM intensity
intensity = np.load('experiments/my_exp_*/fcpm_intensity.npy')

# Custom analysis
plt.imshow(intensity.T, cmap='gray')
plt.show()
```

## Troubleshooting

### Out of Memory

- Reduce grid resolution: `n_x = 32` instead of `64`
- Disable some visualizations in config
- Run on a machine with more RAM

### Slow Execution

- Use lower resolution for testing
- Disable isosurface generation (slowest step)
- Run optical simulation separately if needed

### File Not Found

Make sure you're in the correct directory:

```bash
cd "Enhanced Simulator Cholesteric LC"
python run_experiment.py --preset t3-1-standard
```

## Advanced Usage

### Disable Specific Visualizations

Modify config to skip certain outputs:

```json
{
  ...
  "generate_director_field_viz": true,
  "generate_cross_sections": false,      // Skip this
  "generate_isosurfaces": false,         // Skip this (saves time)
  "generate_optical_simulation": true,
  "generate_comprehensive_figure": true
}
```

### Custom Polarizer Angles

Study different polarizer configurations:

```json
{
  "polarizer_angle": 0.785,      // 45 degrees
  "analyzer_angle": 2.356        // 135 degrees (still crossed)
}
```

### Different Wavelengths

```json
{
  "wavelength": 0.45,   // Blue light
  "wavelength": 0.55,   // Green light (standard)
  "wavelength": 0.65    // Red light
}
```