# Quick Run Guide - FCPM Experiment Runner

## TL;DR - Just Run This

```bash
cd "Enhanced Simulator Cholesteric LC"
python run_experiment.py --preset t3-1-standard
```

Your results will be in `experiments/T3-1_Standard_Resolution_YYYYMMDD_HHMMSS/`

---

## What You Get

Every experiment run creates a **timestamped folder** with:

### 📊 Visualizations
- `00_comprehensive.png` - **9-panel figure** (Nature Materials style)
- `01_director_field.png` - Director field with defects (6 panels)
- `02_cross_sections.png` - Multiple Z-slices showing structure evolution
- `03_isosurfaces.png` - 3D level surfaces (4 different views)
- `04_optical_simulation.png` - FCPM optical analysis

### 💾 Data Files
- `director_field.npz` - 3D director field (lcsim-compatible)
- `fcpm_intensity.npy` - Raw FCPM intensity data
- `config.json` - Full experiment configuration (reusable!)
- `REPORT.md` - Markdown summary with all statistics

---

## Common Use Cases

### 1. Quick Test (30 seconds)
```bash
python run_experiment.py --preset t3-1-standard
```

### 2. High-Resolution Publication Figure
```bash
python run_experiment.py --preset t3-1-highres
```

### 3. Compare All Toron Types
```bash
python run_experiment.py --preset t3-1-standard
python run_experiment.py --preset t3-2-standard
python run_experiment.py --preset t3-3-standard
```

### 4. Custom Configuration
```bash
# Edit example_configs/custom_t3-1.json first
python run_experiment.py --config example_configs/custom_t3-1.json
```

### 5. Interactive Setup
```bash
python run_experiment.py
# Follow the prompts
```

---

## Available Presets

| Preset | Description | Grid | Time |
|--------|-------------|------|------|
| `t3-1-standard` | Standard T3-1 (2 point defects) | 48³ | ~30s |
| `t3-1-highres` | High-res T3-1 | 64³ | ~2min |
| `t3-2-standard` | T3-2 (1 point + 1 ring) | 48³ | ~30s |
| `t3-3-standard` | T3-3 (2 rings) | 48³ | ~30s |
| `uniform-helical` | Uniform helix (baseline) | 48×48×64 | ~25s |

List all presets:
```bash
python run_experiment.py --list-presets
```

---

## Output Organization

```
experiments/
├── T3-1_Standard_Resolution_20251023_142059/
│   ├── 00_comprehensive.png          ← Main figure
│   ├── 01_director_field.png
│   ├── 02_cross_sections.png
│   ├── 03_isosurfaces.png
│   ├── 04_optical_simulation.png
│   ├── director_field.npz            ← Load with lcsim/optics.py
│   ├── fcpm_intensity.npy            ← Raw data for analysis
│   ├── config.json                   ← Reuse with --config
│   └── REPORT.md                     ← Human-readable summary
└── Custom_T3-1_Small_Test_20251023_143045/
    └── ...
```

---

## Customizing an Experiment

### Step 1: Copy example config
```bash
cp example_configs/custom_t3-1.json my_experiment.json
```

### Step 2: Edit parameters
```json
{
  "experiment_name": "My Research Run",
  "description": "Testing effect of larger toron radius",

  "structure_type": "t3-1",
  "n_x": 48,              // Grid resolution
  "toron_radius": 2.5,    // Increase radius

  "wavelength": 0.55,     // Green light
  "dpi": 150              // Image quality
}
```

### Step 3: Run
```bash
python run_experiment.py --config my_experiment.json
```

---

## Key Parameters to Adjust

### Structure
- `structure_type`: `'t3-1'`, `'t3-2'`, `'t3-3'`, `'uniform_helical'`
- `n_x`, `n_y`, `n_z`: Grid resolution (32=fast, 48=standard, 64=high-quality)
- `toron_radius`: Size of toron (1.5 - 3.0 μm typical)
- `pitch`: Cholesteric pitch (1.5 - 3.0 μm typical)

### Optics
- `wavelength`: 0.45 (blue), 0.55 (green), 0.65 (red) μm
- `no`, `ne`: Refractive indices (typical: 1.5, 1.7)

### Output
- `dpi`: 150 (standard), 300 (publication)
- `generate_*`: Enable/disable specific visualizations

---

## Reproducing a Previous Experiment

Every experiment saves its configuration. To re-run:

```bash
python run_experiment.py --config experiments/old_experiment_20251020_123456/config.json
```

---

## Integration with Other Tools

### Use with lcsim/optics.py
```bash
python /path/to/lcsim/bin/optics.py experiments/my_experiment_*/director_field.npz
```

### Load in Python
```python
import numpy as np

# Load director field
data = np.load('experiments/my_experiment_20251023_*/director_field.npz')
director = data['PATH'][0]  # Shape: (nx, ny, nz, 3)

# Load FCPM intensity
intensity = np.load('experiments/my_experiment_*/fcpm_intensity.npy')
```

---

## Tips

### Fast Testing
Start with low resolution:
```json
{"n_x": 32, "n_y": 32, "n_z": 32}
```

### Publication Quality
Use high resolution and DPI:
```json
{"n_x": 64, "n_y": 64, "n_z": 64, "dpi": 300}
```

### Skip Slow Visualizations
Disable isosurfaces for faster runs:
```json
{"generate_isosurfaces": false}
```

### Organize by Topic
Use custom output directories:
```bash
python run_experiment.py --preset t3-1-standard --output-dir paper_figures/figure3
```

---

## Example Workflow: Parameter Study

```bash
# Create configs for different radii
# (Edit toron_radius in each)

python run_experiment.py --config study/radius_1.5.json
python run_experiment.py --config study/radius_2.0.json
python run_experiment.py --config study/radius_2.5.json
python run_experiment.py --config study/radius_3.0.json

# Now compare the 00_comprehensive.png files from each experiment!
```

---