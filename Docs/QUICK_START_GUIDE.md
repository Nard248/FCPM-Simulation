# Quick Start Guide: 3D Toron Simulator

## Installation

No additional packages needed beyond your existing setup:
```bash
# You already have:
numpy, matplotlib, scipy
```

## 5-Minute Quick Start

### 1. Generate Your First Toron

```python
from toron_simulator_3d import ToronSimulator3D, SimulationParams3D

# Create T3-1 toron
params = SimulationParams3D(structure_type='t3-1')
sim = ToronSimulator3D(params)
sim.generate_structure()
sim.print_summary()
```

**Output:**
```
✓ Generated structure: t3-1
Grid Size: (64, 64, 64)
Physical Size: (5.00, 5.00, 5.00) μm
Topological Defects: 3
  1. hyperbolic_point (charge: -1)
  2. hyperbolic_point (charge: -1)
  3. disclination_ring (charge: 2)
TOTAL CHARGE: 0 (conserved!)
```

### 2. Save and Load

```python
# Save
sim.save_to_npz('my_toron.npz')

# Load later
sim2 = ToronSimulator3D.load_from_npz('my_toron.npz')
```

### 3. Visualize

```python
from visualize_torons import visualize_director_field_3d
import matplotlib.pyplot as plt

fig = visualize_director_field_3d(sim)
plt.show()
```

That's it! You've created, saved, and visualized a 3D toron structure.

---

## Structure Types

### Uniform Structures (Testing)

**Uniform Vertical:**
```python
params = SimulationParams3D(structure_type='uniform_vertical')
# All directors point up: (0, 0, 1)
```

**Uniform Helical:**
```python
params = SimulationParams3D(
    structure_type='uniform_helical',
    pitch=2.0
)
# Directors rotate in x-y plane as function of z
```

### Toron Structures (Research)

**T3-1 (Two Point Defects):**
```python
params = SimulationParams3D(
    structure_type='t3-1',
    toron_radius=2.0,
    toron_center=(3.0, 3.0, 3.0)
)
# Features: 2 hyperbolic points + 1 disclination ring
```

**T3-2 (Hybrid):**
```python
params = SimulationParams3D(
    structure_type='t3-2',
    toron_radius=2.0
)
# Features: 1 point + 1 s=-1/2 ring + toroid ring
```

**T3-3 (Two Rings):**
```python
params = SimulationParams3D(
    structure_type='t3-3',
    toron_radius=2.0
)
# Features: 2 s=-1/2 rings + toroid ring
```

---

## Common Tasks

### Adjust Grid Resolution

```python
# Low resolution (fast, for testing)
params = SimulationParams3D(
    n_x=32, n_y=32, n_z=32,
    structure_type='t3-1'
)

# High resolution (slow, for publication)
params = SimulationParams3D(
    n_x=128, n_y=128, n_z=128,
    structure_type='t3-1'
)
```

### Change Physical Size

```python
params = SimulationParams3D(
    x_size=10.0,  # 10 microns
    y_size=10.0,
    z_size=10.0,
    toron_radius=3.0,  # Larger toron
    toron_center=(5.0, 5.0, 5.0)  # Adjust center
)
```

### Access Director Field Data

```python
# Generate structure
sim = ToronSimulator3D(params)
sim.generate_structure()

# Access director field
director = sim.director_field  # Shape: (n_x, n_y, n_z, 3)

# Get components at specific point
i, j, k = 32, 32, 32
nx = director[i, j, k, 0]
ny = director[i, j, k, 1]
nz = director[i, j, k, 2]

print(f"Director at ({i},{j},{k}): ({nx:.3f}, {ny:.3f}, {nz:.3f})")

# Get slice (x-y plane at z=k)
nx_slice = director[:, :, k, 0]
ny_slice = director[:, :, k, 1]
nz_slice = director[:, :, k, 2]
```

### Extract Grid Information

```python
info = sim.get_grid_info()

print(f"Lattice size: {info['lattice_size']}")
print(f"Physical size: {info['physical_size']}")
print(f"Step size: {info['step_size']}")
print(f"Sample thickness: {info['thickness']} μm")
```

---

## Visualization Options

### Comprehensive View (6 Panels)

```python
from visualize_torons import visualize_director_field_3d

fig = visualize_director_field_3d(sim, figsize=(18, 12))
fig.savefig('my_toron_analysis.png', dpi=150)
```

**Panels:**
1. 3D quiver plot
2. X-Y midplane with director arrows
3. X-Z cross-section
4. Director components along z-axis
5. nz component distribution
6. Defect summary

### Cross-Sections (Multiple Slices)

```python
from visualize_torons import plot_multiple_cross_sections

fig = plot_multiple_cross_sections(sim, n_slices=5)
fig.savefig('my_toron_slices.png', dpi=150)
```

### Compare Multiple Structures

```python
from visualize_torons import compare_toron_types

fig = compare_toron_types()  # T3-1 vs T3-2 vs T3-3
fig.savefig('toron_comparison.png', dpi=150)
```

---

## Advanced Usage

### Custom Toron Parameters

```python
params = SimulationParams3D(
    # Grid
    n_x=64, n_y=64, n_z=64,

    # Physical size
    x_size=8.0, y_size=8.0, z_size=8.0,

    # Liquid crystal properties
    pitch=2.5,
    no=1.5,  # Ordinary refractive index
    ne=1.7,  # Extraordinary refractive index

    # Toron structure
    structure_type='t3-1',
    toron_radius=2.5,
    toron_center=(4.0, 4.0, 4.0),

    # Optional
    phase_offset=0.0,
    include_defects=True
)
```

### Batch Processing

```python
# Generate multiple torons with different radii
for radius in [1.0, 1.5, 2.0, 2.5]:
    params = SimulationParams3D(
        structure_type='t3-1',
        toron_radius=radius
    )
    sim = ToronSimulator3D(params)
    sim.generate_structure()
    sim.save_to_npz(f'toron_R_{radius:.1f}.npz')
    print(f"✓ Generated toron with R = {radius:.1f} μm")
```

### Extract Defect Information

```python
sim = ToronSimulator3D(params)
sim.generate_structure()

# Get all defects
for defect in sim.defect_locations:
    print(f"Type: {defect['type']}")
    print(f"Charge: {defect.get('charge', 'N/A')}")
    if 'physical_position' in defect:
        print(f"Position: {defect['physical_position']}")
    print()
```

---

## Troubleshooting

### Q: Director field has NaN values

**A:** This is normal! NaN marks topological defect locations (point defects).

```python
# Count NaN values
import numpy as np
n_defect_points = np.sum(np.isnan(director[:,:,:,0]))
print(f"Defect points: {n_defect_points}")
```

### Q: Visualization looks strange

**A:** Try different views and cross-sections:

```python
# View different planes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# X-Y plane
axes[0].imshow(director[:, :, n_z//2, 2].T, cmap='coolwarm')
axes[0].set_title('X-Y plane')

# X-Z plane
axes[1].imshow(director[:, n_y//2, :, 1].T, cmap='coolwarm')
axes[1].set_title('X-Z plane')

# Y-Z plane
axes[2].imshow(director[n_x//2, :, :, 0].T, cmap='coolwarm')
axes[2].set_title('Y-Z plane')

plt.show()
```

### Q: Want to see director magnitude

**A:** Calculate and visualize:

```python
magnitude = np.sqrt(
    director[:,:,:,0]**2 +
    director[:,:,:,1]**2 +
    director[:,:,:,2]**2
)

# Should be ~1.0 everywhere (except defects where it's NaN)
print(f"Mean magnitude: {np.nanmean(magnitude):.3f}")
print(f"Std magnitude: {np.nanstd(magnitude):.6f}")
```

---

## Integration with optics.py

Your toron structures are compatible with the lcsim optical simulator:

```bash
# First, make sure you have lcsim/optics.py
cd /path/to/lcsim

# Simulate optics
python lcsim/bin/optics.py path/to/toron_t3_1.npz \
    --output optical_result.npz \
    --wavelength 0.65 \
    --no 1.5 \
    --ne 1.7 \
    --polarizer 0 \
    --analyzer 90
```

This will simulate light propagation through your toron structure!

---

## Next Steps

### Learn More:
- Read `PHASE_1_2_SUMMARY.md` for technical details
- Read `COMPREHENSIVE_PROJECT_ANALYSIS.md` for full context
- Read `DETAILED_CODE_ANALYSIS.md` for optics.py integration

### Explore Code:
- `toron_simulator_3d.py` - Main simulator
- `visualize_torons.py` - Visualization suite
- `optics.py` - Optical simulation (from lcsim)
- `demo1.ipynb` - Advanced visualization (from topovec)

### What's Next:
**Phase 3:** Cross-polarizer optical simulation
- Implement Jones calculus
- Layer-by-layer propagation
- FCPM image generation

**Phase 4:** Advanced visualization
- Integrate topovec library
- Isosurface rendering (level surfaces)
- Topological skeleton
- Magnus-style plots

---

## Cheat Sheet

```python
# === GENERATE ===
from toron_simulator_3d import ToronSimulator3D, SimulationParams3D
params = SimulationParams3D(structure_type='t3-1')
sim = ToronSimulator3D(params)
sim.generate_structure()

# === SAVE/LOAD ===
sim.save_to_npz('toron.npz')
sim = ToronSimulator3D.load_from_npz('toron.npz')

# === VISUALIZE ===
from visualize_torons import visualize_director_field_3d
import matplotlib.pyplot as plt
fig = visualize_director_field_3d(sim)
plt.show()

# === ACCESS DATA ===
director = sim.director_field  # (n_x, n_y, n_z, 3)
info = sim.get_grid_info()
defects = sim.defect_locations

# === PRINT INFO ===
sim.print_summary()
```

---

## Examples

See these demo scripts:
- `toron_simulator_3d.py` - Run with `python toron_simulator_3d.py`
- `visualize_torons.py` - Run with `python visualize_torons.py`

Both include comprehensive examples and will generate test files and images.

---

## Getting Help

If something doesn't work:
1. Check that numpy, matplotlib, scipy are installed
2. Check file paths are correct
3. Try with smaller grid first (32³)
4. Read error messages carefully
5. Check the comprehensive documentation

Happy simulating! 🎉
