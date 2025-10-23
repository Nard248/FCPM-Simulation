# FCPM Toron Simulation - Complete Implementation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**Three-dimensional simulation and visualization of toron structures in cholesteric liquid crystals using Fluorescence Confocal Polarizing Microscopy (FCPM).**

Based on: Smalyukh et al., "Three-dimensional structure and multistable optical switching of triple-twisted particle-like excitations in anisotropic fluids" _Nature Materials_ (2009)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Scientific Background](#scientific-background)
- [Implementation Details](#implementation-details)
- [Example Gallery](#example-gallery)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

This project provides a complete computational framework for:

1. **Generating** 3D toron structures (T3-1, T3-2, T3-3) in cholesteric liquid crystals
2. **Simulating** optical FCPM images using Jones calculus
3. **Visualizing** director fields with publication-quality figures
4. **Analyzing** topological defects and charge conservation
5. **Exporting** data in formats compatible with external tools (lcsim, topovec)

### What are Torons?

Torons are localized particle-like excitations in chiral liquid crystals featuring:
- **Triple-twist** director configuration
- **Toroidal** double-twist cylinders
- **Topological defects** (points and disclination rings)
- **Hopf fibration** and skyrmion-like structures
- **Topological charge conservation**

---

## ✨ Key Features

### 🎯 Core Capabilities

- ✅ **Full 3D Director Field**: Arbitrary orientation with `(nx, ny, nz)` components
- ✅ **Three Toron Types**: T3-1, T3-2, T3-3 with different defect configurations
- ✅ **Jones Calculus Optics**: Layer-by-layer FCPM simulation with crossed polarizers
- ✅ **Topological Analysis**: Defect detection, charge calculation, singularity tracking
- ✅ **Publication-Quality Viz**: Nature Materials style figures with nested isosurfaces

### 🔧 Technical Highlights

- **Arbitrary 3D Director**: Handles `nz ≠ 0` (critical fix from reference implementation)
- **lcsim-Compatible**: NPZ format works with `lcsim/optics.py`
- **topovec Integration**: Export format for advanced topological visualization
- **Organized Experiments**: Timestamped folders with complete documentation
- **Reproducible**: Configuration-based runs with automatic report generation

---

## 📁 Project Structure

```
Enhanced Simulator Cholesteric LC/
├── README.md                           # This file
├── QUICK_RUN_GUIDE.md                 # TL;DR usage guide
├── EXPERIMENT_GUIDE.md                # Complete experiment manual
├── ENHANCEMENTS_SUMMARY.md            # Recent visualization improvements
│
├── Core Modules                        # Main simulation code
│   ├── toron_simulator_3d.py          # Toron structure generation (Phases 1-2)
│   ├── optical_simulator.py           # FCPM optical simulation (Phase 3)
│   ├── advanced_visualization.py      # Isosurfaces & defect detection (Phase 4)
│   ├── enhanced_visualization.py      # Publication-quality figures (NEW!)
│   └── visualize_torons.py            # Basic visualization tools
│
├── Experiment System                   # Unified workflow
│   ├── run_experiment.py              # Main entry point (RECOMMENDED)
│   ├── example_configs/               # Example configurations
│   │   └── custom_t3-1.json
│   └── experiments/                   # Output directory (auto-created)
│
├── Legacy/Demos                        # Individual component demos
│   ├── final_demo.py                  # Complete pipeline demo
│   ├── comprehensive_demo.py          # Legacy comprehensive demo
│   └── demo_new_features.py           # Feature demonstrations
│
├── Reconstruction Tools               # Advanced reconstruction (optional)
│   ├── enhanced_fcpm_simulator.py     # Enhanced FCPM features
│   └── advanced_reconstruction_tools.py # Director field reconstruction
│
└── Documentation                       # Additional docs
    ├── Docs/                          # Parent directory docs
    ├── .gitignore                     # Git ignore rules
    └── cleanup.sh                     # Cleanup script
```

---

## 🚀 Installation

### Requirements

- Python 3.11+
- NumPy, SciPy, Matplotlib, scikit-image

### Setup

```bash
# Clone the repository
cd "Enhanced Simulator Cholesteric LC"

# Install dependencies
pip install numpy scipy matplotlib scikit-image

# Verify installation
python run_experiment.py --list-presets
```

**No additional setup required!** All dependencies are standard scientific Python packages.

---

## ⚡ Quick Start

### 1. Run Your First Experiment (30 seconds)

```bash
python run_experiment.py --preset t3-1-standard
```

This creates a complete experiment folder with:
- ✅ All visualizations (5 publication-quality figures)
- ✅ Director field data (NPZ format)
- ✅ FCPM optical simulation results
- ✅ Markdown report with statistics
- ✅ Reusable configuration file

### 2. Check the Results

```bash
ls experiments/T3-1_Standard_Resolution_*/
```

**Output:**
```
00_comprehensive.png           # 9-panel comprehensive figure
01_director_field.png          # Director field with defects
02_cross_sections.png          # Multiple Z-slices
03_isosurfaces.png            # Nested "onion" surfaces
04_optical_simulation.png      # FCPM analysis
director_field.npz             # 3D director field data
fcpm_intensity.npy             # Raw FCPM intensity
topovec_export.npz            # topovec-compatible export
config.json                    # Experiment configuration
REPORT.md                      # Detailed summary
```

### 3. View the Comprehensive Figure

Open `experiments/*/00_comprehensive.png` to see:
- 3D nested toroidal surfaces (onion structure)
- Cross-sections with director field arrows
- FCPM simulated images
- Topological defect summary
- No overlapping elements!

---

## 📖 Usage Guide

### Method 1: Preset Configurations (Recommended)

```bash
# List available presets
python run_experiment.py --list-presets

# Run preset experiments
python run_experiment.py --preset t3-1-standard   # Standard resolution
python run_experiment.py --preset t3-1-highres    # High resolution (64³)
python run_experiment.py --preset t3-2-standard   # Different defect structure
python run_experiment.py --preset t3-3-standard   # All ring defects
python run_experiment.py --preset uniform-helical # Baseline comparison
```

### Method 2: Custom Configuration

```bash
# Create custom config (edit example_configs/custom_t3-1.json)
python run_experiment.py --config my_experiment.json
```

**Example configuration:**
```json
{
  "experiment_name": "High Resolution T3-1",
  "description": "Publication-quality toron visualization",

  "structure_type": "t3-1",
  "n_x": 64, "n_y": 64, "n_z": 64,
  "x_size": 6.0, "y_size": 6.0, "z_size": 6.0,
  "pitch": 2.0,
  "toron_radius": 2.0,
  "toron_center": [3.0, 3.0, 3.0],

  "wavelength": 0.55,
  "no": 1.5,
  "ne": 1.7,

  "dpi": 300
}
```

### Method 3: Python API

```python
from toron_simulator_3d import ToronSimulator3D, SimulationParams3D
from optical_simulator import CrossPolarizerSimulator, OpticalParams
from enhanced_visualization import create_onion_visualization

# Generate toron
params = SimulationParams3D(structure_type='t3-1')
sim = ToronSimulator3D(params)
sim.generate_structure()

# Optical simulation
optical_params = OpticalParams(wavelength=0.55, no=1.5, ne=1.7)
optical_sim = CrossPolarizerSimulator(optical_params)
intensity = optical_sim.simulate_fcpm(sim.director_field,
                                      (params.x_size, params.y_size, params.z_size))

# Visualize
fig = create_onion_visualization(sim)
fig.savefig('my_toron.png', dpi=150)
```

---

## 🔬 Scientific Background

### Toron Structures

**T3-1: Two Hyperbolic Point Defects**
- Toroidal double-twist cylinder
- Two charge -1 point defects (top/bottom)
- One s=+1 disclination ring (toroid axis)
- Total charge: +2 - 2 = 0 ✓

**T3-2: One Point + One Ring**
- Mixed defect structure
- One charge -1 point defect
- One s=-1/2 disclination ring
- Total charge: +2 - 1 - 1 = 0 ✓

**T3-3: Two Disclination Rings**
- No point defects
- Two s=-1/2 disclination rings
- Total charge: +2 - 1 - 1 = 0 ✓

### Physical Parameters

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| Cholesteric pitch | p | 1-10 μm | Helical periodicity |
| Toron radius | R | 1-5 μm | Toroid major radius |
| Grid resolution | N | 32-64³ | Simulation grid |
| Wavelength | λ | 0.55 μm | Green light |
| Ordinary index | no | 1.5 | Refractive index ⊥ director |
| Extraordinary index | ne | 1.7 | Refractive index ∥ director |
| Birefringence | Δn | 0.2 | ne - no |

### Optical Simulation

Uses **Jones calculus** with layer-by-layer propagation:

1. **Electric field**: E = [Ex, Ey] (complex)
2. **Effective index**: neff(nx,ny,nz) - **handles arbitrary 3D orientation**
3. **Phase retardation**: γ = π·dz·(neff - no)/λ
4. **Jones matrix**: Transformation for each layer
5. **Crossed polarizers**: 90° analyzer angle

**Key Innovation:** Correctly handles `nz ≠ 0` (director tilt out of plane)

```python
# Critical formula (optical_simulator.py:205-207)
neff = no / sqrt((no/ne)² × (nx² + ny²) + nz²)
```

---

## 🔧 Implementation Details

### Five-Phase Architecture

| Phase | Module | Description |
|-------|--------|-------------|
| **Phase 1** | `toron_simulator_3d.py` | Infrastructure: 4D director field, NPZ I/O |
| **Phase 2** | `toron_simulator_3d.py` | Toron generation: T3-1, T3-2, T3-3 structures |
| **Phase 3** | `optical_simulator.py` | Cross-polarizer FCPM simulation |
| **Phase 4** | `advanced_visualization.py` | Isosurfaces, defect detection |
| **Phase 5** | `enhanced_visualization.py` | Publication figures, topovec export |

### Data Format

**NPZ File Structure (lcsim-compatible):**
```python
data = np.load('director_field.npz')

# Director field: (1, nx, ny, nz, 3)
# First axis is time (we use single timestep)
director = data['PATH'][0]  # Shape: (nx, ny, nz, 3)

# Settings (JSON string)
settings = json.loads(str(data['settings']))

# Defects (JSON string)
defects = json.loads(str(data['defects']))
```

### Topological Features

**Defect Detection:**
- NaN-based point defect identification
- Winding number calculation
- Charge conservation verification

**Visualization:**
- Nested isosurfaces (nz level sets)
- Director field streamlines
- Defect highlighting

---

## 🎨 Example Gallery

### Comprehensive Figure (00_comprehensive.png)
9-panel publication-quality figure with:
- 3D nested isosurfaces (onion structure)
- X-Y and X-Z cross-sections
- Multiple Z-slice views
- FCPM simulated images
- Topological summary

### Cross-Sections (02_cross_sections.png)
**Top row:** nz component (blue ↔ red)
**Bottom row:** In-plane director angle φ (reveals spiral structure)

### Isosurfaces (03_isosurfaces.png)
Beautiful nested "onion" layers matching Nature Materials Figure 1:
- 7 semi-transparent surfaces
- Gradient coloring
- Defects as red spheres
- Optimal viewing angle

---

### In-Code Documentation
All modules have comprehensive docstrings:
```python
help(ToronSimulator3D)
help(CrossPolarizerSimulator)
help(create_onion_visualization)
```

---

## 🎯 Common Use Cases

### 1. Parameter Study
```bash
# Create configs with different radii
for r in 1.5 2.0 2.5 3.0; do
  # Edit config to set toron_radius = $r
  python run_experiment.py --config "study/radius_${r}.json"
done
```

### 2. Compare Toron Types
```bash
python run_experiment.py --preset t3-1-standard
python run_experiment.py --preset t3-2-standard
python run_experiment.py --preset t3-3-standard
```

### 3. Publication Figure
```bash
# High resolution, high DPI
python run_experiment.py --preset t3-1-highres
# Edit config.json to set dpi: 300
python run_experiment.py --config experiments/*/config.json
```

### 4. Integration with External Tools

**lcsim/optics.py:**
```bash
python /path/to/lcsim/bin/optics.py experiments/*/director_field.npz
```

**topovec:**
```python
# In Jupyter notebook
import numpy as np
data = np.load('experiments/*/topovec_export.npz')
director = data['director_field']
# Use topovec visualization tools
```

---

## 🔬 Validation

### Topological Charge Conservation
All generated structures satisfy:
```
Total charge = Σ defect_charges = 0
```

### Comparison with Literature
- Structure matches Smalyukh et al. Nature Materials (2009)
- Optical simulation comparable to experimental FCPM images
- Defect positions consistent with theory

### Numerical Verification
- Director normalization: |n| = 1 everywhere (except at defects)
- Energy minimization: Frank elastic energy computed
- lcsim compatibility: NPZ files load correctly

---