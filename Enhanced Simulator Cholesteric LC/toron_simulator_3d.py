"""
PHASE 1 & 2: 3D Toron Simulator
Upgrades the 2D cholesteric simulator to full 3D with toron structures

Key Changes:
- 4D director field storage: (n_x, n_y, n_z, 3) for [nx, ny, nz]
- NPZ I/O compatible with lcsim/optics.py format
- Toron structure generators (T3-1, T3-2, T3-3)
- Topological defect implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, List, Tuple, Dict
import json


@dataclass
class SimulationParams3D:
    """Parameters for 3D FCPM simulation with toron structures"""
    # 3D Grid parameters (PHASE 1 CHANGE: Added n_y)
    n_x: int = 64      # Points along X
    n_y: int = 64      # Points along Y (NEW!)
    n_z: int = 64      # Points along Z

    # Physical dimensions in microns
    x_size: float = 5.0   # Physical size in X direction (microns)
    y_size: float = 5.0   # Physical size in Y direction (microns)
    z_size: float = 5.0   # Physical size in Z direction (microns)

    # Liquid crystal parameters
    pitch: float = 2.0        # Cholesteric pitch (microns)
    phase_offset: float = 0.0  # Initial phase

    # Structure type
    structure_type: str = 'uniform_vertical'  # 'uniform_vertical', 'uniform_helical', 't3-1', 't3-2', 't3-3'

    # Toron-specific parameters
    toron_radius: float = 1.5  # Radius of toron structure (microns)
    toron_center: Tuple[float, float, float] = (2.5, 2.5, 2.5)  # Center position

    # Defect parameters
    include_defects: bool = True
    defect_types: List[str] = dataclass_field(default_factory=list)

    # Optical parameters
    no: float = 1.5   # Ordinary refractive index
    ne: float = 1.7   # Extraordinary refractive index

    def __post_init__(self):
        if not self.defect_types:
            self.defect_types = []


class ToronSimulator3D:
    """
    3D Toron Simulator implementing PHASE 1 and PHASE 2

    PHASE 1: Infrastructure
    - 4D director field storage
    - NPZ save/load (lcsim format)
    - Basic 3D visualization

    PHASE 2: Toron Generation
    - Toroidal coordinate system
    - T3-1, T3-2, T3-3 structures
    - Topological defects (points and rings)
    """

    def __init__(self, params: SimulationParams3D = None):
        self.params = params or SimulationParams3D()

        # PHASE 1: 4D director field [n_x, n_y, n_z, 3]
        self.director_field = np.zeros((
            self.params.n_x,
            self.params.n_y,
            self.params.n_z,
            3  # Components: [nx, ny, nz]
        ))

        # Defect tracking
        self.defect_locations = []
        self.metadata = {}

        # Optical simulation results
        self.intensity_image = None

    # ============================================================================
    # PHASE 1: INFRASTRUCTURE
    # ============================================================================

    def save_to_npz(self, filename: str):
        """
        Save director field in lcsim-compatible NPZ format
        Compatible with optics.py
        """
        # Prepare settings dictionary (lcsim format)
        settings = {
            'L': self.params.z_size,      # Thickness in microns
            'sx': self.params.n_x,         # Grid size x
            'sy': self.params.n_y,         # Grid size y
            'sz': self.params.n_z,         # Grid size z
            'pitch': self.params.pitch,
            'structure_type': self.params.structure_type,
            'no': self.params.no,
            'ne': self.params.ne,
            'x_size': self.params.x_size,
            'y_size': self.params.y_size,
            'z_size': self.params.z_size,
            'toron_radius': self.params.toron_radius,
            'toron_center': self.params.toron_center,
        }

        # Save in lcsim format
        # Note: Add time dimension as expected by lcsim (even though we have one timestep)
        director_with_time = self.director_field[np.newaxis, ...]  # Shape: (1, nx, ny, nz, 3)

        np.savez_compressed(
            filename,
            PATH=director_with_time,           # Shape: (1, nx, ny, nz, 3)
            settings=json.dumps(settings),     # JSON string
            defects=json.dumps(self.defect_locations)  # Defect info
        )

        print(f"✓ Saved to {filename}")
        print(f"  Format: lcsim-compatible NPZ")
        print(f"  Director field shape: {self.director_field.shape}")
        print(f"  Settings: {len(settings)} parameters")

    @classmethod
    def load_from_npz(cls, filename: str):
        """
        Load director field from lcsim-compatible NPZ format
        Compatible with optics.py
        """
        data = np.load(filename, allow_pickle=True)

        # Load settings
        settings = json.loads(str(data['settings']))

        # Load director field
        director = data['PATH'][0]  # Remove time dimension: (nx, ny, nz, 3)

        # Handle 5D case (if present)
        if len(director.shape) == 5:
            director = director[:, :, :, 0, :]  # Remove extra dimension

        # Create parameters from settings
        params = SimulationParams3D(
            n_x=settings.get('sx', director.shape[0]),
            n_y=settings.get('sy', director.shape[1]),
            n_z=settings.get('sz', director.shape[2]),
            x_size=settings.get('x_size', 5.0),
            y_size=settings.get('y_size', 5.0),
            z_size=settings.get('L', 5.0),
            pitch=settings.get('pitch', 2.0),
            structure_type=settings.get('structure_type', 'unknown'),
            no=settings.get('no', 1.5),
            ne=settings.get('ne', 1.7),
            toron_radius=settings.get('toron_radius', 1.5),
            toron_center=tuple(settings.get('toron_center', [2.5, 2.5, 2.5]))
        )

        # Create simulator
        simulator = cls(params)
        simulator.director_field = director

        # Load defects if available
        if 'defects' in data:
            simulator.defect_locations = json.loads(str(data['defects']))

        print(f"✓ Loaded from {filename}")
        print(f"  Director field shape: {director.shape}")
        print(f"  Structure type: {params.structure_type}")

        return simulator

    def get_grid_info(self):
        """Extract lattice size and thickness information"""
        dx = self.params.x_size / self.params.n_x
        dy = self.params.y_size / self.params.n_y
        dz = self.params.z_size / self.params.n_z

        return {
            'lattice_size': (self.params.n_x, self.params.n_y, self.params.n_z),
            'physical_size': (self.params.x_size, self.params.y_size, self.params.z_size),
            'step_size': (dx, dy, dz),
            'thickness': self.params.z_size,
            'volume': self.params.x_size * self.params.y_size * self.params.z_size
        }

    # ============================================================================
    # PHASE 2: STRUCTURE GENERATION
    # ============================================================================

    def generate_structure(self):
        """
        Main entry point for structure generation
        Dispatches to specific generators based on structure_type
        """
        structure_type = self.params.structure_type.lower()

        if structure_type == 'uniform_vertical':
            self._generate_uniform_vertical()
        elif structure_type == 'uniform_helical':
            self._generate_uniform_helical()
        elif structure_type == 't3-1':
            self._generate_t3_1()
        elif structure_type == 't3-2':
            self._generate_t3_2()
        elif structure_type == 't3-3':
            self._generate_t3_3()
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")

        self._generate_metadata()
        print(f"✓ Generated structure: {structure_type}")
        return self.director_field

    def _generate_uniform_vertical(self):
        """
        Test case: Uniform vertical alignment
        All directors point along +z
        """
        self.director_field[:, :, :, 0] = 0  # nx = 0
        self.director_field[:, :, :, 1] = 0  # ny = 0
        self.director_field[:, :, :, 2] = 1  # nz = 1

        print("  Director: uniform vertical (0, 0, 1)")

    def _generate_uniform_helical(self):
        """
        Uniform helical cholesteric structure
        Director rotates in x-y plane as function of z
        Similar to old 2D simulation but in 3D
        """
        # Create z-coordinates
        z_coords = np.linspace(0, self.params.z_size, self.params.n_z)

        # Calculate angles at each z
        angles = 2 * np.pi * z_coords / self.params.pitch + self.params.phase_offset

        # Set director components (same for all x, y at given z)
        for k, angle in enumerate(angles):
            self.director_field[:, :, k, 0] = np.cos(angle)  # nx
            self.director_field[:, :, k, 1] = np.sin(angle)  # ny
            self.director_field[:, :, k, 2] = 0              # nz = 0 (in-plane)

        print("  Director: uniform helical rotation in x-y plane")

    def _generate_t3_1(self):
        """
        Generate T3-1 toron structure

        Features:
        - Toroidal double-twist cylinder
        - Two hyperbolic point defects (charge -1 each) at top/bottom
        - One s=+1 twist-escaped disclination ring (toroid axis)
        - Total topological charge: +2 (toron) -2 (defects) = 0

        Mathematical structure:
        - Skyrmion-like in midplane
        - Hopf fibration (nested tori)
        - Triple twist (radial + tangential + axial)
        """
        # Create coordinate grids
        x = np.linspace(0, self.params.x_size, self.params.n_x)
        y = np.linspace(0, self.params.y_size, self.params.n_y)
        z = np.linspace(0, self.params.z_size, self.params.n_z)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Toron center
        x0, y0, z0 = self.params.toron_center
        R_toron = self.params.toron_radius

        # Translate coordinates to center
        X_c = X - x0
        Y_c = Y - y0
        Z_c = Z - z0

        # Convert to cylindrical coordinates (r, theta, z)
        r_cyl = np.sqrt(X_c**2 + Y_c**2)
        theta_cyl = np.arctan2(Y_c, X_c)

        # Distance from toroid's circular axis
        # The toroid's axis is a circle of radius R_toron in the x-y plane at z=z0
        rho = np.sqrt((r_cyl - R_toron)**2 + Z_c**2)
        phi = np.arctan2(Z_c, r_cyl - R_toron)

        # Director field for skyrmion-like structure
        # Tilt angle: vertical at center, horizontal at edge
        tilt_angle = np.pi * (1 - np.exp(-rho / (R_toron * 0.5)))

        # Azimuthal twist (around toroid)
        twist_angle = theta_cyl + 2 * np.pi * Z_c / self.params.pitch + phi

        # Director components
        nx = np.sin(tilt_angle) * np.cos(twist_angle)
        ny = np.sin(tilt_angle) * np.sin(twist_angle)
        nz = np.cos(tilt_angle)

        # Normalize
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        # Add point defects at top and bottom (on z-axis at toron center)
        # These are hyperbolic point defects (charge -1 each)
        defect_top_z = int((z0 + R_toron * 0.8) / self.params.z_size * self.params.n_z)
        defect_bottom_z = int((z0 - R_toron * 0.8) / self.params.z_size * self.params.n_z)

        center_x = int(x0 / self.params.x_size * self.params.n_x)
        center_y = int(y0 / self.params.y_size * self.params.n_y)

        # Mark defects (set to NaN)
        defect_radius = 2  # grid points
        for dx in range(-defect_radius, defect_radius + 1):
            for dy in range(-defect_radius, defect_radius + 1):
                if dx**2 + dy**2 <= defect_radius**2:
                    i = np.clip(center_x + dx, 0, self.params.n_x - 1)
                    j = np.clip(center_y + dy, 0, self.params.n_y - 1)

                    if defect_top_z < self.params.n_z:
                        nx[i, j, defect_top_z] = np.nan
                        ny[i, j, defect_top_z] = np.nan
                        nz[i, j, defect_top_z] = np.nan

                    if defect_bottom_z >= 0:
                        nx[i, j, defect_bottom_z] = np.nan
                        ny[i, j, defect_bottom_z] = np.nan
                        nz[i, j, defect_bottom_z] = np.nan

        # Store director field
        self.director_field[:, :, :, 0] = nx
        self.director_field[:, :, :, 1] = ny
        self.director_field[:, :, :, 2] = nz

        # Record defects
        self.defect_locations = [
            {
                'type': 'hyperbolic_point',
                'charge': -1,
                'position': (center_x, center_y, defect_top_z),
                'physical_position': (x0, y0, z0 + R_toron * 0.8)
            },
            {
                'type': 'hyperbolic_point',
                'charge': -1,
                'position': (center_x, center_y, defect_bottom_z),
                'physical_position': (x0, y0, z0 - R_toron * 0.8)
            },
            {
                'type': 'disclination_ring',
                'strength': 1,
                'charge': 2,
                'radius': R_toron,
                'center': (x0, y0, z0),
                'description': 'Twist-escaped s=+1 ring on toroid axis'
            }
        ]

        print(f"  Toron T3-1 structure:")
        print(f"    Center: ({x0:.2f}, {y0:.2f}, {z0:.2f}) μm")
        print(f"    Radius: {R_toron:.2f} μm")
        print(f"    Defects: 2 point defects + 1 disclination ring")
        print(f"    Total charge: 0 (conserved)")

    def _generate_t3_2(self):
        """
        Generate T3-2 toron structure

        Features:
        - Toroidal double-twist cylinder (same as T3-1)
        - One hyperbolic point defect (charge -1)
        - One s=-1/2 disclination ring (charge -1)
        - Total charge: +2 - 1 - 1 = 0
        """
        # Start with T3-1 structure
        self._generate_t3_1()

        # Modify to have one point defect + one -1/2 ring
        # Remove one point defect and add disclination ring instead

        # Keep only bottom point defect
        self.defect_locations = [
            self.defect_locations[1],  # Bottom point defect
            self.defect_locations[2],  # Toroid axis ring
        ]

        # Add s=-1/2 disclination ring at top
        x0, y0, z0 = self.params.toron_center
        R_toron = self.params.toron_radius

        self.defect_locations.append({
            'type': 'disclination_ring',
            'strength': -0.5,
            'charge': -1,
            'radius': R_toron * 0.6,
            'center': (x0, y0, z0 + R_toron * 0.8),
            'description': 's=-1/2 disclination ring at top'
        })

        # Update structure type
        self.params.structure_type = 't3-2'

        print(f"  Toron T3-2 structure:")
        print(f"    Defects: 1 point defect + 1 s=-1/2 ring + 1 toroid ring")

    def _generate_t3_3(self):
        """
        Generate T3-3 toron structure

        Features:
        - Toroidal double-twist cylinder (same as T3-1)
        - Two s=-1/2 disclination rings (charge -1 each)
        - Total charge: +2 - 1 - 1 = 0
        """
        # Start with T3-1 structure
        self._generate_t3_1()

        # Replace both point defects with -1/2 rings
        x0, y0, z0 = self.params.toron_center
        R_toron = self.params.toron_radius

        self.defect_locations = [
            {
                'type': 'disclination_ring',
                'strength': -0.5,
                'charge': -1,
                'radius': R_toron * 0.6,
                'center': (x0, y0, z0 + R_toron * 0.8),
                'description': 's=-1/2 disclination ring at top'
            },
            {
                'type': 'disclination_ring',
                'strength': -0.5,
                'charge': -1,
                'radius': R_toron * 0.6,
                'center': (x0, y0, z0 - R_toron * 0.8),
                'description': 's=-1/2 disclination ring at bottom'
            },
            self.defect_locations[2]  # Keep toroid axis ring
        ]

        # Update structure type
        self.params.structure_type = 't3-3'

        print(f"  Toron T3-3 structure:")
        print(f"    Defects: 2 s=-1/2 rings + 1 toroid ring")

    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================

    def _generate_metadata(self):
        """Generate metadata about the simulation"""
        # Calculate statistics
        nx_component = self.director_field[:, :, :, 0]
        ny_component = self.director_field[:, :, :, 1]
        nz_component = self.director_field[:, :, :, 2]

        # Ignore NaN values (defect locations)
        nx_valid = nx_component[~np.isnan(nx_component)]
        ny_valid = ny_component[~np.isnan(ny_component)]
        nz_valid = nz_component[~np.isnan(nz_component)]

        self.metadata = {
            'structure_type': self.params.structure_type,
            'grid_info': self.get_grid_info(),
            'parameters': {
                'pitch': self.params.pitch,
                'toron_radius': self.params.toron_radius,
                'toron_center': self.params.toron_center,
                'no': self.params.no,
                'ne': self.params.ne,
            },
            'statistics': {
                'nx_mean': float(np.mean(nx_valid)) if len(nx_valid) > 0 else 0,
                'ny_mean': float(np.mean(ny_valid)) if len(ny_valid) > 0 else 0,
                'nz_mean': float(np.mean(nz_valid)) if len(nz_valid) > 0 else 0,
                'nx_std': float(np.std(nx_valid)) if len(nx_valid) > 0 else 0,
                'ny_std': float(np.std(ny_valid)) if len(ny_valid) > 0 else 0,
                'nz_std': float(np.std(nz_valid)) if len(nz_valid) > 0 else 0,
            },
            'defects': {
                'count': len(self.defect_locations),
                'details': self.defect_locations
            }
        }

    def print_summary(self):
        """Print summary of the structure"""
        print("\n" + "="*60)
        print("3D TORON SIMULATOR - STRUCTURE SUMMARY")
        print("="*60)

        info = self.get_grid_info()
        print(f"Structure Type: {self.params.structure_type}")
        print(f"Grid Size: {info['lattice_size']}")
        print(f"Physical Size: ({info['physical_size'][0]:.2f}, "
              f"{info['physical_size'][1]:.2f}, {info['physical_size'][2]:.2f}) μm")
        print(f"Resolution: ({info['step_size'][0]:.3f}, "
              f"{info['step_size'][1]:.3f}, {info['step_size'][2]:.3f}) μm/gridpoint")

        if hasattr(self, 'metadata') and self.metadata:
            stats = self.metadata['statistics']
            print(f"\nDirector Field Statistics:")
            print(f"  <nx> = {stats['nx_mean']:.3f} ± {stats['nx_std']:.3f}")
            print(f"  <ny> = {stats['ny_mean']:.3f} ± {stats['ny_std']:.3f}")
            print(f"  <nz> = {stats['nz_mean']:.3f} ± {stats['nz_std']:.3f}")

            if self.defect_locations:
                print(f"\nTopological Defects: {len(self.defect_locations)}")
                for i, defect in enumerate(self.defect_locations, 1):
                    print(f"  {i}. {defect['type']} (charge: {defect.get('charge', '?')})")
                    if 'physical_position' in defect:
                        pos = defect['physical_position']
                        print(f"     Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) μm")

        print("="*60 + "\n")


# ============================================================================
# DEMO AND TESTING
# ============================================================================

def demo_phase1():
    """Demo Phase 1: Infrastructure and I/O"""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Infrastructure")
    print("="*60 + "\n")

    # Test 1: Uniform vertical structure
    print("Test 1: Uniform Vertical Structure")
    print("-" * 40)
    params = SimulationParams3D(
        n_x=32, n_y=32, n_z=32,
        structure_type='uniform_vertical'
    )
    sim = ToronSimulator3D(params)
    sim.generate_structure()
    sim.print_summary()

    # Save to NPZ
    sim.save_to_npz('test_uniform_vertical.npz')

    # Load back
    sim_loaded = ToronSimulator3D.load_from_npz('test_uniform_vertical.npz')

    # Verify
    assert np.allclose(sim.director_field, sim_loaded.director_field, equal_nan=True)
    print("✓ NPZ save/load verified!\n")

    # Test 2: Uniform helical structure
    print("\nTest 2: Uniform Helical Structure")
    print("-" * 40)
    params2 = SimulationParams3D(
        n_x=32, n_y=32, n_z=64,
        pitch=2.0,
        structure_type='uniform_helical'
    )
    sim2 = ToronSimulator3D(params2)
    sim2.generate_structure()
    sim2.print_summary()
    sim2.save_to_npz('test_uniform_helical.npz')
    print("✓ Phase 1 tests passed!\n")


def demo_phase2():
    """Demo Phase 2: Toron Generation"""
    print("\n" + "="*60)
    print("PHASE 2 DEMO: Toron Structures")
    print("="*60 + "\n")

    # Test T3-1
    print("Test 1: T3-1 Toron Structure")
    print("-" * 40)
    params_t3_1 = SimulationParams3D(
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0),
        structure_type='t3-1'
    )
    sim_t3_1 = ToronSimulator3D(params_t3_1)
    sim_t3_1.generate_structure()
    sim_t3_1.print_summary()
    sim_t3_1.save_to_npz('toron_t3_1.npz')

    # Test T3-2
    print("\nTest 2: T3-2 Toron Structure")
    print("-" * 40)
    params_t3_2 = params_t3_1
    params_t3_2.structure_type = 't3-2'
    sim_t3_2 = ToronSimulator3D(params_t3_2)
    sim_t3_2.generate_structure()
    sim_t3_2.print_summary()
    sim_t3_2.save_to_npz('toron_t3_2.npz')

    # Test T3-3
    print("\nTest 3: T3-3 Toron Structure")
    print("-" * 40)
    params_t3_3 = params_t3_1
    params_t3_3.structure_type = 't3-3'
    sim_t3_3 = ToronSimulator3D(params_t3_3)
    sim_t3_3.generate_structure()
    sim_t3_3.print_summary()
    sim_t3_3.save_to_npz('toron_t3_3.npz')

    print("✓ Phase 2 tests passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("3D TORON SIMULATOR - PHASES 1 & 2")
    print("="*60)

    # Run Phase 1 demo
    demo_phase1()

    # Run Phase 2 demo
    demo_phase2()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_uniform_vertical.npz")
    print("  - test_uniform_helical.npz")
    print("  - toron_t3_1.npz")
    print("  - toron_t3_2.npz")
    print("  - toron_t3_3.npz")
    print("\nThese files can be loaded with:")
    print("  - ToronSimulator3D.load_from_npz('filename.npz')")
    print("  - python optics.py filename.npz (from lcsim)")
    print("="*60 + "\n")
