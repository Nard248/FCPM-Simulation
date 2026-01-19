# Comprehensive Project Analysis: Toron Simulation & Visualization

## Executive Summary

You are being asked to **upgrade your 2D cholesteric liquid crystal simulation to a full 3D toron structure simulator** that can:
1. Generate and store 3D director fields with **arbitrary orientation** (not just in-plane)
2. Visualize these fields using **level surfaces**, **singularities**, and **topological features**
3. Simulate realistic FCPM images from these 3D structures
4. Match the toroidal "onion-like" structures shown in the Nature Materials paper

---

## What ARE Torons? (From the Paper)

### Physical Structure
**Torons** ("Triple-Twist Torons" or T3s) are:
- **Localized 3D particle-like excitations** in chiral nematic liquid crystals
- A **double-twist cylinder looped on itself** forming a torus (donut shape)
- The director field (molecular orientation) twists in **ALL THREE dimensions** simultaneously
- Size: approximately equal to the cholesteric pitch (~500nm to 200µm)

### Key Features from Figure 1 (Your Screenshot):
```
┌─────────────────────────────────────────────────────────┐
│  a. Full toroid structure (onion-like layers)          │
│     - Nested tori with twisted director field           │
│     - Red line: s=+1 disclination ring (toroid axis)    │
│     - Topological charge: +2                             │
│                                                          │
│  b,c. Defects that embed toron in uniform field:        │
│     - Hyperbolic point defects (charge -1)              │
│     - s=-1/2 disclination rings                         │
│                                                          │
│  d,e,f. Three types of torons (T3-1, T3-2, T3-3):      │
│     - T3-1: Two -1 point defects (top/bottom)           │
│     - T3-2: One -1 point + one -1/2 ring                │
│     - T3-3: Two -1/2 rings (top/bottom)                 │
│                                                          │
│  g-j. Laguerre-Gaussian beams used to create them       │
│     - Vortex beams with helical phase fronts            │
│     - Topological charge l = 0, +2, -1, -1/2             │
└─────────────────────────────────────────────────────────┘
```

### Director Field Mathematics
From the paper (page 3):
```
Director Field n(r) in 3D:
- At toroid center: vertical (along z)
- Twists radially outward by 180°
- Twists tangentially around the circular toroid axis
- Twists axially along the toroid thickness

This is called "TRIPLE TWIST" - twist in all 3 spatial directions
```

**Topological Features:**
- Skyrmion-like structure in the midplane
- Resembles Hopf fibration (nested tori)
- Topological defects stabilize the structure

---

## What You Currently Have

### ✅ Strengths of Your Current Code:
1. **Excellent 2D cholesteric simulation**
   - 7 types of defects (dislocations, disclinations, kinks, etc.)
   - Realistic noise models (Gaussian, Poisson, salt-pepper)
   - Proper pseudo-vector physics (π intensity period)

2. **Advanced reconstruction tools**
   - 4 reconstruction methods
   - 3 defect detection algorithms
   - Quality metrics and confidence estimation

3. **Good visualization**
   - 2D intensity patterns
   - Cross-sections
   - Power spectrum analysis

### ❌ Current Limitations:
1. **Director field is ONLY in x-y plane**
   - Currently: `n = [cos(β(z)), sin(β(z)), 0]`
   - Needed: Full 3D arbitrary orientation

2. **No 3D spatial structure**
   - Currently: 2D pattern (z vs x)
   - Needed: Full 3D grid (x, y, z)

3. **No toroidal structures**
   - Currently: Uniform helical twist
   - Needed: Localized toron with nested tori

4. **Visualization is 2D-focused**
   - Currently: Heatmaps and 2D quiver plots
   - Needed: Level surfaces, isosurfaces, topological skeleton

---

## What You Need to Build: The Feedback Decoded

### 1. **"Config to simulate Toron"**
**Translation:** Create a simulation that generates the 3D director field for toron structures (T3-1, T3-2, T3-3) instead of just uniform cholesteric patterns.

**What this means:**
- Generate the toroidal double-twist cylinder structure
- Include topological defects (point defects and disclination rings)
- Embed the toron in a uniform background field

---

### 2. **"Use Magnus to visualize the results"**
**Translation:** Use the visualization style/tools from the Magnus library (likely referring to magnetic visualization tools that can also apply to liquid crystals).

**What this means:**
- Magnus likely refers to magnetic structure visualization
- Similar topology between magnetic skyrmions and liquid crystal torons
- You need to adopt their visualization techniques

---

### 3. **"Level Surfaces, Singularities in visualizations"**
**Translation:** Your visualizations should show:

**Level Surfaces (Isosurfaces):**
- Surfaces where a quantity (e.g., director z-component) is constant
- The "onion layers" visible in Figure 1a
- Created using marching cubes or similar algorithm

**Singularities:**
- Topological defect points and lines
- Point defects: locations where director field is undefined
- Disclination lines: continuous lines of defects
- Visualize as points, lines, or tubes

---

### 4. **"Visualizer from Magnus"**
**What you need to check:**
- The GitLab repos mentioned have visualization notebooks
- You should download the actual notebooks to see the visualization code
- They likely use Python libraries like:
  - `mayavi` or `pyvista` for 3D visualization
  - `scikit-image` for isosurface extraction
  - Custom topology analysis tools

---

### 5. **"Store data as 5 Dimensional Array"**
**This is CRITICAL - here's the exact format:**

```python
director_field = np.zeros((n_x, n_y, n_z, 3))
                          ↑    ↑    ↑    ↑
                          x    y    z    director components [nx, ny, nz]
```

**Breakdown:**
- **Axes 0-2:** Spatial coordinates (x, y, z grid points)
- **Axis 3:** Director vector components
  - `field[i, j, k, 0]` = n_x component at position (i,j,k)
  - `field[i, j, k, 1]` = n_y component at position (i,j,k)
  - `field[i, j, k, 2]` = n_z component at position (i,j,k)

**Note:** They said "5D" but likely meant 4D (3 spatial + 1 vector). The 4th dimension in some contexts could be time or different field types, but for static director fields it's 4D total.

**Why this format?**
- Compatible with NPZ file format
- Easy to extract slices: `field[:, :, k, :]` gives x-y plane at height k
- Standard format for vector fields in scientific computing

---

### 6. **"Extract size of the lattice and thickness of the field"**
**Translation:** Your code should be able to read NPZ files and extract:

```python
# Lattice size
n_x, n_y, n_z = director_field.shape[:3]  # Grid resolution
dx, dy, dz = ...  # Physical spacing between grid points

# Thickness of the field
sample_thickness = n_z * dz  # Physical sample thickness
lattice_extent = (n_x * dx, n_y * dy, n_z * dz)  # Total volume
```

**From the lcsim optics.py link:** This file likely shows how to:
- Open NPZ files: `data = np.load('file.npz')`
- Extract metadata about grid spacing
- Compute physical dimensions

---

### 7. **"Cross polarizer - all layers included"**
**Translation:** When simulating FCPM images, you need to:

**Current (wrong):** Simple intensity = cos⁴(β)

**Needed (correct):**
- Simulate light propagation through **ALL z-layers**
- Account for birefringence in each layer
- Compute total transmitted intensity considering:
  - Crossed polarizers (input and output perpendicular)
  - Jones calculus or Mueller calculus
  - Phase retardation accumulation through sample

**From optics.py:** This file shows how to:
```python
# Pseudocode from typical optical simulation
for each z-layer:
    retardation = 2π * Δn * thickness / λ
    jones_matrix = rotation_matrix @ retarder_matrix @ rotation_matrix.T

total_transmission = polarizer_output @ (Π jones_matrices) @ polarizer_input
intensity = |total_transmission|²
```

---

### 8. **"Functions that open NPZ files and incorporate to code or do custom"**
**Translation:** Create utility functions like:

```python
def load_director_field(filename):
    """Load director field from NPZ file"""
    data = np.load(filename)
    director = data['director_field']  # Shape: (nx, ny, nz, 3)
    metadata = {
        'pitch': data['pitch'],
        'lattice_size': data['lattice_size'],
        'physical_size': data['physical_size']
    }
    return director, metadata

def save_director_field(filename, director, **metadata):
    """Save director field to NPZ file"""
    np.savez(filename,
             director_field=director,
             **metadata)
```

---

### 9. **"Does not work with arbitrary orientation of the director field, only on the plane - modify that part"**
**This is the KEY LIMITATION they identified!**

**Current limitation in your code (line in simulator):**
```python
# From enhanced_fcpm_simulator.py
# Director field is ONLY in x-y plane:
n_x = np.cos(beta)
n_y = np.sin(beta)
n_z = 0  # ← ALWAYS ZERO - this is the problem!
```

**What you need:**
```python
# Full 3D director with ALL components:
def compute_toron_director_field(x, y, z):
    """
    Compute director field for toron structure
    Returns: n_x, n_y, n_z (all non-zero!)
    """
    # Example: skyrmion-like structure
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Director tilts from vertical to horizontal
    tilt = np.pi * r / R_toron  # R_toron = toron radius

    # Twist around toroid
    twist = theta + (2*np.pi * z / pitch)

    # Full 3D components:
    n_x = np.sin(tilt) * np.cos(twist)
    n_y = np.sin(tilt) * np.sin(twist)
    n_z = np.cos(tilt)  # ← NOW NON-ZERO!

    return n_x, n_y, n_z
```

---

## The Goal: Match Figure 1 from the Paper

### Visual Target
You need to be able to create visualizations that look like Figure 1 (your screenshot):

1. **Panel a:** 3D rendering showing nested toroidal surfaces
2. **Panels b,c,d,e,f:** Cross-sections showing director field with defects
3. **Panels g,h,i,j:** Simulated FCPM images (you're close to this already!)

### Key visual elements to replicate:
- ✅ Nested curved surfaces (like onion layers)
- ✅ Director field arrows showing twist
- ✅ Red lines for disclinations
- ✅ Blue circles for point defects
- ✅ Coordinate axes and scale bars

---

## References You Need to Get

Since the web fetches failed, you need to manually obtain:

### 1. **optics.py from lcsim**
**URL:** https://gitlab.com/alepoydes/lcsim/-/blob/nonorientable/lcsim/bin/optics.py

**How to get it:**
```bash
# Option 1: Direct download
wget https://gitlab.com/alepoydes/lcsim/-/raw/nonorientable/lcsim/bin/optics.py

# Option 2: Clone the repo
git clone https://gitlab.com/alepoydes/lcsim.git
cd lcsim
git checkout nonorientable
cat lcsim/bin/optics.py
```

**What you'll learn from it:**
- How to load NPZ files with 5D/4D data
- Cross-polarizer optical simulation
- Proper Jones calculus implementation

---

### 2. **demo1.ipynb from topovec**
**URL:** https://gitlab.com/alepoydes/topovec/-/blob/master/notebooks/demo1.ipynb

**How to get it:**
```bash
# Option 1: Direct download
wget https://gitlab.com/alepoydes/topovec/-/raw/master/notebooks/demo1.ipynb

# Option 2: Clone the repo
git clone https://gitlab.com/alepoydes/topovec.git
cd topovec/notebooks
jupyter notebook demo1.ipynb
```

**What you'll learn from it:**
- Level surface extraction (isosurfaces)
- Topological skeleton computation
- Singularity detection
- Visualization with `mayavi` or `pyvista`
- Magnus-style visualization techniques

---

## Step-by-Step Action Plan

### Phase 1: Data Structure Upgrade (Week 1)
**Tasks:**
1. ✅ Modify `SimulationParams` to include 3D grid:
   ```python
   n_x: int = 64  # Add x-dimension
   n_y: int = 64  # Add y-dimension
   n_z: int = 64  # Keep z-dimension
   ```

2. ✅ Create 4D director field storage:
   ```python
   self.director_field = np.zeros((n_x, n_y, n_z, 3))
   ```

3. ✅ Add save/load for NPZ format:
   ```python
   def save_npz(self, filename):
       np.savez(filename,
                director_field=self.director_field,
                pitch=self.params.pitch,
                lattice_size=(self.params.n_x, self.params.n_y, self.params.n_z),
                physical_size=(self.params.x_size, self.params.y_size, self.params.z_size))
   ```

---

### Phase 2: Toron Structure Generation (Week 2-3)
**Tasks:**
1. ✅ Implement toron director field equations:
   - Use toroidal coordinates
   - Implement double-twist cylinder
   - Add twist in radial, tangential, and axial directions

2. ✅ Add topological defects:
   - Point defects (hyperbolic): locations where director is undefined
   - Disclination rings: continuous line defects

3. ✅ Create toron configurations:
   - T3-1: with two point defects
   - T3-2: with one point defect and one ring
   - T3-3: with two disclination rings

**Mathematical Reference (from paper):**
```python
def generate_t3_1_structure(grid_x, grid_y, grid_z, pitch, R_toron):
    """
    Generate T3-1 toron structure

    Parameters:
    - grid_x, grid_y, grid_z: 3D meshgrid
    - pitch: cholesteric pitch
    - R_toron: toron radius
    """
    # Convert to cylindrical coordinates
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)

    # Distance from toroid center
    rho = np.sqrt((r - R_toron)**2 + grid_z**2)
    phi = np.arctan2(grid_z, r - R_toron)

    # Director tilt (skyrmion-like)
    tilt_angle = np.pi * (1 - np.exp(-rho / R_toron))

    # Azimuthal twist
    twist_angle = theta + 2*np.pi * grid_z / pitch

    # 3D director components
    n_x = np.sin(tilt_angle) * np.cos(twist_angle)
    n_y = np.sin(tilt_angle) * np.sin(twist_angle)
    n_z = np.cos(tilt_angle)

    # Normalize
    norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
    n_x /= norm
    n_y /= norm
    n_z /= norm

    # Add point defects at top/bottom
    # (set director to NaN at defect locations)
    defect_top = (grid_x == 0) & (grid_y == 0) & (grid_z == z_top)
    defect_bottom = (grid_x == 0) & (grid_y == 0) & (grid_z == z_bottom)

    n_x[defect_top | defect_bottom] = np.nan
    n_y[defect_top | defect_bottom] = np.nan
    n_z[defect_top | defect_bottom] = np.nan

    return np.stack([n_x, n_y, n_z], axis=-1)
```

---

### Phase 3: Advanced Visualization (Week 4-5)
**Tasks:**
1. ✅ Install visualization libraries:
   ```bash
   pip install pyvista mayavi scikit-image
   ```

2. ✅ Implement level surface visualization:
   ```python
   from skimage import measure
   import pyvista as pv

   def plot_level_surfaces(director_field):
       """Plot isosurfaces of director z-component"""
       n_z_component = director_field[:, :, :, 2]

       # Extract isosurfaces at different levels
       levels = [-0.8, -0.4, 0, 0.4, 0.8]

       plotter = pv.Plotter()
       for level in levels:
           verts, faces, _, _ = measure.marching_cubes(n_z_component, level=level)
           mesh = pv.PolyData(verts, faces)
           plotter.add_mesh(mesh, opacity=0.5)

       plotter.show()
   ```

3. ✅ Implement singularity detection:
   ```python
   def detect_singularities(director_field):
       """Detect topological defects"""
       # Point defects: locations with NaN
       point_defects = np.isnan(director_field[:, :, :, 0])

       # Disclination lines: track winding number
       # (More complex - use topological charge calculation)

       return point_defects, disclination_lines
   ```

4. ✅ Create topological skeleton visualization:
   ```python
   def plot_topological_skeleton(director_field, point_defects, disclination_rings):
       """Visualize topological features"""
       plotter = pv.Plotter()

       # Add director field as streamlines
       # Add point defects as spheres
       # Add disclination rings as tubes

       plotter.show()
   ```

---

### Phase 4: Optical Simulation Upgrade (Week 6)
**Tasks:**
1. ✅ Implement Jones calculus for cross-polarizer:
   ```python
   def simulate_fcpm_3d(director_field, wavelength=550e-9, delta_n=0.1):
       """
       Simulate FCPM with crossed polarizers

       Parameters:
       - director_field: (nx, ny, nz, 3) array
       - wavelength: light wavelength
       - delta_n: birefringence
       """
       nx, ny, nz, _ = director_field.shape
       intensity = np.zeros((nx, ny))

       for i in range(nx):
           for j in range(ny):
               # Extract director along z-column
               n_column = director_field[i, j, :, :]

               # Compute Jones matrix for each layer
               jones_total = np.eye(2, dtype=complex)

               for k in range(nz):
                   n_x, n_y, n_z_val = n_column[k, :]

                   # Rotation angle
                   theta = np.arctan2(n_y, n_x)

                   # Retardation
                   delta = 2 * np.pi * delta_n * dz / wavelength

                   # Jones matrix for this layer
                   R = rotation_matrix(theta)
                   retarder = np.array([[np.exp(1j*delta/2), 0],
                                       [0, np.exp(-1j*delta/2)]])
                   jones_layer = R.T @ retarder @ R

                   jones_total = jones_layer @ jones_total

               # Crossed polarizers
               polarizer_in = np.array([1, 0])
               polarizer_out = np.array([0, 1])

               E_out = polarizer_out @ jones_total @ polarizer_in
               intensity[i, j] = np.abs(E_out)**2

       return intensity
   ```

---

### Phase 5: Integration & Testing (Week 7)
**Tasks:**
1. ✅ Create comprehensive demo showing all features
2. ✅ Generate figures matching paper Figure 1
3. ✅ Validate against experimental data (if available)
4. ✅ Write documentation

---

## Technical Requirements Summary

### New Classes Needed:
```python
class ToronSimulator:
    """Generate 3D toron structures"""
    def generate_t3_1(self, pitch, radius): ...
    def generate_t3_2(self, pitch, radius): ...
    def generate_t3_3(self, pitch, radius): ...
    def add_uniform_background(self): ...

class DirectorField3D:
    """Handle 3D director fields"""
    def __init__(self, shape_3d): ...
    def save_npz(self, filename): ...
    def load_npz(self, filename): ...
    def get_slice(self, axis, index): ...
    def compute_topological_charge(self): ...

class TopologicalAnalyzer:
    """Detect and analyze topological features"""
    def detect_point_defects(self): ...
    def detect_disclination_lines(self): ...
    def compute_winding_number(self): ...
    def extract_skeleton(self): ...

class AdvancedVisualizer:
    """3D visualization with level surfaces"""
    def plot_isosurfaces(self): ...
    def plot_director_field_3d(self): ...
    def plot_topological_skeleton(self): ...
    def plot_defects(self): ...

class OpticalSimulator3D:
    """Cross-polarizer optical simulation"""
    def jones_calculus_propagation(self): ...
    def simulate_fcpm_image(self): ...
    def apply_cross_polarizers(self): ...
```

---

## Key Physics to Implement

### 1. Toron Director Field Equations
From the paper (page 2-3), the director field has:
- **Radial twist:** θ_r = f(ρ) where ρ is distance from toroid axis
- **Tangential twist:** θ_t = f(θ) around toroid
- **Axial twist:** θ_z = f(z) along vertical

### 2. Topological Charge Conservation
- Toron has charge +2 (from s=+1 disclination ring)
- Embedded defects have total charge -2
- T3-1: two -1 point defects
- T3-2: one -1 point + one -1/2 ring
- T3-3: two -1/2 rings

### 3. Hopf Fibration Structure
The director field lines form nested tori (Hopf fibration):
- Each field line is a closed loop
- Loops are linked together
- Mapping to S³ sphere

---

## Deliverables Checklist

- [ ] 4D director field data structure (3D spatial + 3-component vector)
- [ ] NPZ file I/O functions
- [ ] Toron structure generator (T3-1, T3-2, T3-3)
- [ ] Topological defect implementation
- [ ] Level surface visualization
- [ ] Singularity visualization
- [ ] Topological skeleton extraction
- [ ] Cross-polarizer optical simulation
- [ ] FCPM image generation from 3D structure
- [ ] Figures matching Nature Materials paper
- [ ] Documentation and examples

---

## Next Immediate Steps

1. **URGENT:** Get the actual code files from GitLab:
   - Download `optics.py` from lcsim repo
   - Download `demo1.ipynb` from topovec repo
   - Share them with me so I can analyze the exact implementation

2. **Start with data structure:**
   - Modify your `SimulationParams` to include 3D grid
   - Create 4D array storage
   - Implement save/load NPZ functions

3. **Simple test case:**
   - Generate a simple 3D structure (e.g., uniform twist with out-of-plane component)
   - Verify the 4D storage works
   - Test NPZ save/load

4. **Study the toron mathematics:**
   - Read the paper's supplementary information
   - Understand the toroidal coordinate system
   - Plan the mathematical implementation

Would you like me to start implementing any specific part of this plan?
