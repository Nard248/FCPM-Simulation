# Detailed Analysis: optics.py and demo1.ipynb

## Executive Summary

I've successfully obtained and analyzed both reference files. Here's what they reveal about the required implementation:

---

## 1. Data Structure from optics.py

### NPZ File Format (lines 15-30)

```python
def lcs_load(filename):
    """
    Load LC simulation state from NPZ file

    Returns:
    - state_n[x,y,z,d] - director field (THIS IS THE KEY!)
    - state_alpha[x,y,z] - concentration field
    - settings - JSON metadata
    """
    data = np.load(filename)
    settings = json.loads(data['settings'][()])
    state_n = data['PATH'][0]

    # Handle 5D case (time series)
    if len(state_n.shape)==5:
        assert state_n.shape[-2]==1
        state_n = state_n[:,:,:,0]  # Extract single time point

    return settings, state_n, state_alpha
```

**Key Findings:**
- **Data format:** `state_n[x, y, z, d]` where `d` is 0,1,2 for director components
- **Shape:** `(n_x, n_y, n_z, 3)` - This is the 4D array they mentioned!
- **The "5D" case:** When including time or multiple states, it's `(n_time, n_x, n_y, n_z, 3)`
- **Storage key:** Data stored in NPZ with key `'PATH'`
- **Metadata:** Settings stored as JSON string in `'settings'` key

**Critical insight:** Your feedback said "5D array" but actually meant:
- 3 spatial dimensions (x, y, z)
- 1 director component dimension (3 values: nx, ny, nz)
- Total: 4D static field, or 5D with time/multiple states

---

## 2. Lattice Size Extraction (lines 308-311)

```python
# Extract settings from JSON
thickness = settings['L']           # Sample thickness in microns
stepsize = thickness/settings['sz'] # Grid spacing
sx, sy, sz, _ = state_n.shape      # Grid dimensions

# Computed quantities:
# lz = sz * stepsize  # Physical length in z-direction
# Same for x and y directions
```

**What you need to implement:**
```python
def extract_lattice_info(npz_file):
    """Extract size of lattice and thickness of field"""
    data = np.load(npz_file)

    # Get director field
    director = data['PATH'][0]
    n_x, n_y, n_z, _ = director.shape

    # Get settings
    settings = json.loads(data['settings'][()])

    # Physical dimensions
    thickness = settings['L']  # Usually in microns
    stepsize = thickness / n_z

    # Full lattice size
    lattice_size = (n_x, n_y, n_z)
    physical_size = (n_x * stepsize, n_y * stepsize, n_z * stepsize)

    return {
        'lattice_size': lattice_size,
        'physical_size': physical_size,
        'stepsize': stepsize,
        'thickness': thickness
    }
```

---

## 3. Cross-Polarizer Optical Simulation (lines 202-251)

### The LCFilm Class - THIS IS CRITICAL!

```python
@dataclass
class LCFilm:
    thickness: float      # Film thickness (microns)
    dz: float            # Integration step
    stepsize: float      # Grid spacing
    director: np.ndarray # Director field [x,y,z,3]
    no: float           # Ordinary refractive index
    ne: float           # Extraordinary refractive index
```

### Layer-by-Layer Propagation Algorithm

**Key concept:** Light propagates through the sample layer by layer, accumulating phase retardation.

```python
def apply(self, beam):
    """
    Propagate light through LC film using Jones calculus

    Algorithm:
    1. Start at z=0 with incident beam
    2. For each layer at height z:
        a. Interpolate director field at beam positions
        b. Compute effective refractive index
        c. Calculate phase retardation
        d. Apply Jones matrix transformation
        e. Move to next layer
    """
    xx, yy = beam.cartesian_coordinates()  # Beam positions
    E = beam.E.copy()  # Electric field [x, y, 2] (Ex, Ey components)

    z = 0
    while z < self.thickness:
        # Layer thickness
        dz = min(self.dz, self.thickness - z)

        # Interpolate director at this z-plane
        m = interpolate(xx, yy, z, self.stepsize, self.director)
        # m[i,j,:] = [nx, ny, nz] at position (xx[i,j], yy[i,j], z)

        # Normalize in-plane components
        l2 = m[...,0]**2 + m[...,1]**2  # nx² + ny²
        ll = np.sqrt(l2) + 1e-15        # Avoid division by zero
        nx = m[...,0] / ll              # Normalized nx
        ny = m[...,1] / ll              # Normalized ny

        # Project E-field onto director
        p = E[...,0]*nx + E[...,1]*ny   # E·n projection

        # Effective refractive index (depends on tilt!)
        # This is the key formula for arbitrary orientation!
        neff = self.no / np.sqrt((self.no/self.ne)**2 * l2 + m[...,2]**2)

        # Phase retardation for this layer
        gammaz = np.pi * dz / wavelength * (neff - self.no)

        # Jones matrix application (compact form)
        # This accounts for both rotation and retardation
        factor = p * (1 - np.cos(2*gammaz) + 1j*np.sin(2*gammaz))
        E[...,0] -= factor * nx
        E[...,1] -= factor * ny

        z += dz

    return E  # Final electric field
```

**Critical Physics Formulas:**

1. **Effective refractive index for arbitrary orientation:**
   ```python
   neff = no / sqrt((no/ne)² · (nx² + ny²) + nz²)
   ```
   This accounts for director tilt out of the x-y plane!

2. **Phase retardation per layer:**
   ```python
   γ = π · Δz · (neff - no) / λ
   ```

3. **Jones matrix (implicit in the factor calculation):**
   The compact form avoids explicit matrix multiplication but is equivalent to:
   ```
   J = R(-θ) · [[exp(iγ), 0], [0, exp(-iγ)]] · R(θ)
   ```
   where R(θ) is rotation matrix and θ = atan2(ny, nx)

---

## 4. Interpolation for Arbitrary Positions (lines 163-199)

**Why this matters:** The simulation grid might not align with the beam positions.

```python
def interpolate(x, y, z, stepsize, v):
    """
    Trilinear interpolation of vector field v at positions (x, y, z)

    Parameters:
    - x, y, z: Query coordinates (can be arrays)
    - stepsize: Grid spacing
    - v: Vector field [sx, sy, sz, 3]

    Returns:
    - Interpolated vectors at (x, y, z) positions
    """
    sx, sy, sz, _ = v.shape

    # Convert coordinates to grid indices
    x0, x1, xl = getindices(x, sx, stepsize, sx/2)
    y0, y1, yl = getindices(y, sy, stepsize, sy/2)
    z0, z1, zl = getindices(z, sz, stepsize, 0)

    # Trilinear interpolation (8 corner weights)
    return (
        v[x0,y0,z0]*(1-xl)*(1-yl)*(1-zl) +
        v[x0,y0,z1]*(1-xl)*(1-yl)*(  zl) +
        v[x0,y1,z0]*(1-xl)*(  yl)*(1-zl) +
        v[x0,y1,z1]*(1-xl)*(  yl)*(  zl) +
        v[x1,y0,z0]*(  xl)*(1-yl)*(1-zl) +
        v[x1,y0,z1]*(  xl)*(1-yl)*(  zl) +
        v[x1,y1,z0]*(  xl)*(  yl)*(1-zl) +
        v[x1,y1,z1]*(  xl)*(  yl)*(  zl)
    )
```

**Key point:** You need trilinear interpolation when:
- Simulating FCPM at non-grid positions
- Handling different resolutions for structure vs. imaging
- Computing director field at arbitrary query points

---

## 5. Visualization from demo1.ipynb

### The topovec Library Structure

```python
import topovec as tv

# Load data (similar to optics.py)
images, system, settings = tv.core.load_lcsim_npz(filename)

# `images` is a list of director fields
director = images[0]  # Shape: (nx, ny, nz, 3)
print(f"Director grid {director.shape}")
```

### Key Visualization Functions

#### A. Layer Rendering (Cross-section)
```python
# Render a single x-y plane at height z=layer
ic = tv.mgl.render_layer(
    system,           # System object (defines grid)
    layer=10,         # Which z-plane to show
    axis=1            # Which axis is "vertical" (0=x, 1=y, 2=z)
)
ic.upload(director)   # Upload director field
img = ic.save()       # Render to PIL image
```

#### B. Isosurface Rendering (Level Surfaces!)
```python
# This is what you need for Figure 1a "onion layers"!
ic = tv.mgl.render_isosurface(
    system,
    preset=0          # Camera preset
)
ic.upload(director)
img = ic.save()
```

**What this creates:** 3D surfaces where a component (e.g., nz) is constant

#### C. Isolines Rendering
```python
# 2D contour lines
ic = tv.mgl.render_isolines(
    system,
    preset=0,
    nlines=10         # Number of contour lines
)
ic.upload(director)
img = ic.save()
```

#### D. General Scene Rendering
```python
# List all available scenes
tv.mgl.print_scenes()

# Prepare custom scene
ic = tv.mgl.render_prepare(
    scenecls='Astra',     # Scene type
    system=system,
    imgsize=512
)

# Configure scene properties
ic.scene['Axes']['Show axes'] = True
ic.scene['Axes']['Length'] = 1.0

# Render
ic.upload(director)
img = ic.save()
```

### Available Scene Types (from notebook)
1. **LayerScene** - Single cross-section with director arrows
2. **FlowScene** - Isosurfaces (level surfaces)
3. **IsolinesScene** - Contour lines
4. **Astra** - Advanced rendering with multiple features

---

## 6. Color Mapping (HSV for Director Orientation)

```python
# HSV color scheme maps director orientation to color
# - Hue: in-plane angle (atan2(ny, nx))
# - Saturation: tilt from vertical (sqrt(nx² + ny²))
# - Value: constant or based on intensity

# Visualize the mapping
tv.matplotlib.plot_bloch_sphere(axis=1)
```

**Color interpretation:**
- Red: director along +x
- Green: director along +y
- Blue: director along +z
- Intermediate colors: intermediate orientations
- Saturation: how much director tilts from z-axis

---

## 7. Cross-Polarizer Simulation in topovec

```python
# Advanced optical simulation with multiple wavelengths
import taichi as ti
ti.init()

lambdas = np.linspace(0.380, 1.000, 13)  # Wavelengths (microns)

rgb = tv.ti.render_cp_multi(
    nn=director[:,:,:,0,:],           # Director field (what's the [0] for?)
    wavelengths=lambdas,              # Array of wavelengths
    ne=tv.ti.ConstantFunction(1.7),   # Extraordinary index
    no=tv.ti.ConstantFunction(1.5),   # Ordinary index
    emission=tv.ti.BlackbodyRadiation(),  # Light source spectrum
    efficiency=tv.ti.SimplifiedRGB(),     # Color response
    thickness=settings['L'],          # Sample thickness
    polarizer=0.,                     # Input polarizer angle
    deltafilter=np.pi/2              # Crossed polarizers (90° difference)
)

plt.imshow(rgb)
```

**Key parameters:**
- `deltafilter=np.pi/2`: Crossed polarizers (this is the critical part!)
- Multi-wavelength simulation: More accurate than monochromatic
- Taichi backend: GPU acceleration

---

## 8. The "System" Object

From the notebook, the `system` object encapsulates:
```python
# When loading:
images, system, settings = tv.core.load_lcsim_npz(filename)

# System contains:
# - Grid dimensions (nx, ny, nz)
# - Physical size (Lx, Ly, Lz)
# - Step size (dx, dy, dz)

# Can create sparse version:
sparse_director, sparse_system = system.thinned(data=director, steps=4)
# This reduces resolution by factor of 4 (faster rendering)

# Camera adjustment:
camera = tv.mgl.Camera()
camera.adjust_fov(system)      # Set field of view based on system size
camera.adjust_scale(system)    # Set scale
```

---

## 9. Key Implementation Requirements for Your Code

### A. Data Structure Modifications

**Current (your code):**
```python
# 2D simulation
self.intensity_data = np.zeros((n_z, n_x))  # Wrong!
```

**Required:**
```python
# 3D director field with arbitrary orientation
self.director_field = np.zeros((n_x, n_y, n_z, 3))
# [i, j, k, 0] = nx at position (i, j, k)
# [i, j, k, 1] = ny at position (i, j, k)
# [i, j, k, 2] = nz at position (i, j, k)
```

### B. NPZ Save/Load Functions

```python
def save_to_npz(self, filename):
    """Save in lcsim-compatible format"""
    # Create settings dictionary
    settings = {
        'L': self.params.z_size,  # Thickness in microns
        'sx': self.params.n_x,
        'sy': self.params.n_y,
        'sz': self.params.n_z,
        'pitch': self.params.pitch,
        # Add other relevant parameters
    }

    # Save with correct structure
    np.savez_compressed(
        filename,
        PATH=self.director_field[np.newaxis, ...],  # Add time dimension
        settings=json.dumps(settings),
        # Optional: add defect information, etc.
    )

def load_from_npz(filename):
    """Load from lcsim-compatible format"""
    data = np.load(filename)
    settings = json.loads(data['settings'][()])
    director = data['PATH'][0]  # Remove time dimension

    return director, settings
```

### C. Cross-Polarizer FCPM Simulation

**You must replace your current simple intensity calculation with:**

```python
def simulate_fcpm_with_crosspol(self,
                                wavelength=0.55,  # microns
                                no=1.5, ne=1.7,
                                polarizer_angle=0,
                                analyzer_angle=np.pi/2):
    """
    Simulate FCPM with crossed polarizers

    This replaces your current: intensity = cos^4(beta)
    """
    nx, ny, nz = self.params.n_x, self.params.n_y, self.params.n_z
    intensity_image = np.zeros((nx, ny))

    # Grid spacing
    dz = self.params.z_size / nz

    # Input polarization
    E_in = np.array([np.cos(polarizer_angle), np.sin(polarizer_angle)])

    # Output analyzer
    E_out = np.array([np.cos(analyzer_angle), np.sin(analyzer_angle)])

    # For each x-y position:
    for i in range(nx):
        for j in range(ny):
            # Start with input field
            E = E_in.copy().astype(complex)

            # Propagate through all z-layers
            for k in range(nz):
                # Get director at this position
                director = self.director_field[i, j, k, :]
                nx_val, ny_val, nz_val = director

                # Normalize in-plane components
                l2 = nx_val**2 + ny_val**2
                ll = np.sqrt(l2) + 1e-15
                nx_norm = nx_val / ll
                ny_norm = ny_val / ll

                # Effective refractive index
                neff = no / np.sqrt((no/ne)**2 * l2 + nz_val**2)

                # Phase retardation
                gamma = np.pi * dz * (neff - no) / wavelength

                # Apply Jones matrix (compact form from optics.py)
                p = E[0]*nx_norm + E[1]*ny_norm
                factor = p * (1 - np.cos(2*gamma) + 1j*np.sin(2*gamma))
                E[0] -= factor * nx_norm
                E[1] -= factor * ny_norm

            # Apply analyzer and compute intensity
            E_final = np.dot(E_out, E)
            intensity_image[i, j] = np.abs(E_final)**2

    return intensity_image
```

### D. Visualization Pipeline

**You need to add (using existing libraries):**

```python
def visualize_with_topovec(self):
    """
    Create Magnus-style visualizations
    Requires: pip install topovec
    """
    import topovec as tv

    # Create system object
    system = tv.core.System(
        shape=(self.params.n_x, self.params.n_y, self.params.n_z),
        size=(self.params.x_size, self.params.y_size, self.params.z_size)
    )

    # 1. Render layer (cross-section)
    ic = tv.mgl.render_layer(system, layer=self.params.n_z//2)
    ic.upload(self.director_field)
    layer_img = ic.save()

    # 2. Render isosurfaces (level surfaces)
    ic = tv.mgl.render_isosurface(system)
    ic.upload(self.director_field)
    isosurface_img = ic.save()

    # 3. Detect and visualize singularities
    # (You'll need to implement topological analysis)
    defects = self.detect_singularities()
    # Overlay defects on visualization

    return {
        'layer': layer_img,
        'isosurface': isosurface_img,
        'defects': defects
    }
```

---

## 10. Action Items Based on Code Analysis

### Immediate (Week 1):
1. ✅ **Install topovec library:**
   ```bash
   pip install topovec
   ```

2. ✅ **Modify data structure:**
   ```python
   # In SimulationParams:
   n_x: int = 64
   n_y: int = 64
   n_z: int = 64

   # In EnhancedFCPMSimulator:
   self.director_field = np.zeros((n_x, n_y, n_z, 3))
   ```

3. ✅ **Implement NPZ I/O:**
   - `save_to_npz()` - Save in lcsim format
   - `load_from_npz()` - Load lcsim format
   - Test compatibility with topovec

### Medium-term (Week 2-3):
4. ✅ **Implement cross-polarizer simulation:**
   - Replace `cos^4(beta)` with Jones calculus
   - Layer-by-layer propagation
   - Arbitrary director orientation support

5. ✅ **Create 3D director field generators:**
   - Uniform vertical alignment (test case)
   - Uniform helical (current, but 3D)
   - Simple skyrmion structure
   - Eventually: full toron structures

### Long-term (Week 4-6):
6. ✅ **Integrate topovec visualization:**
   - Layer rendering
   - Isosurface rendering
   - Custom camera angles
   - Export figures

7. ✅ **Implement toron structures:**
   - T3-1 with point defects
   - T3-2 hybrid
   - T3-3 with disclination rings

8. ✅ **Topological analysis:**
   - Defect detection
   - Winding number calculation
   - Topological skeleton
   - Charge conservation verification

---

## 11. Critical Formulas Summary

### Effective Refractive Index (Arbitrary Orientation):
```python
neff = no / sqrt((no/ne)² · (nx² + ny²) + nz²)
```

### Phase Retardation per Layer:
```python
gamma = π · dz · (neff - no) / λ
```

### Jones Matrix Transformation (Compact Form):
```python
p = E·n  # Project E onto director
factor = p · (1 - cos(2γ) + i·sin(2γ))
E_new = E_old - factor · n
```

### Trilinear Interpolation:
```python
v_interp = sum over 8 corners of:
    v[corner] × w_x × w_y × w_z
```

---

## 12. Libraries You Need to Install

```bash
# Core dependencies
pip install numpy scipy matplotlib

# Image handling
pip install pillow

# Visualization (choose one or both):
pip install topovec         # For Magnus-style visualization
pip install pyvista         # Alternative 3D visualization
pip install mayavi          # Another alternative

# Optional (for GPU acceleration):
pip install taichi          # For fast cross-polarizer simulation

# For testing compatibility:
git clone https://gitlab.com/alepoydes/lcsim.git
git clone https://gitlab.com/alepoydes/topovec.git
```

---

## Next Steps

1. **Test the optics.py code:**
   - Try loading one of your existing simulations with it
   - See what errors occur (will help debug format issues)

2. **Create minimal test case:**
   - Generate a simple 3D director field (e.g., uniform)
   - Save in NPZ format
   - Load with optics.py and visualize with topovec

3. **Gradually add complexity:**
   - Start with uniform field → helical → skyrmion → toron
   - Test visualization at each step
   - Verify optical simulation produces expected results

Would you like me to start implementing the first step (data structure modification)?
