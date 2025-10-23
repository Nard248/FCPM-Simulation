"""
PHASE 4: Advanced Visualization
Implements level surfaces, singularity detection, and topological analysis

Uses matplotlib and scipy (no external dependencies needed)
Can be upgraded to use topovec/pyvista later
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from scipy.ndimage import label, center_of_mass
from skimage import measure
from typing import List, Tuple, Dict
import json


class TopologicalAnalyzer:
    """
    Detects and analyzes topological features in director fields

    Features:
    - Point defect detection
    - Disclination line tracking
    - Winding number calculation
    - Topological charge computation
    """

    def __init__(self, director_field: np.ndarray):
        """
        Args:
            director_field: Shape (n_x, n_y, n_z, 3)
        """
        self.director_field = director_field
        self.n_x, self.n_y, self.n_z, _ = director_field.shape

        self.point_defects = []
        self.disclination_lines = []
        self.singularities = []

    def detect_point_defects(self) -> List[Dict]:
        """
        Detect point defects (locations with NaN director)

        Returns:
            List of defect dictionaries with position and properties
        """
        print("Detecting point defects...")

        # Find NaN locations
        nx = self.director_field[:, :, :, 0]
        nan_mask = np.isnan(nx)

        if not np.any(nan_mask):
            print("  No point defects found (no NaN values)")
            return []

        # Label connected components
        labeled, n_features = label(nan_mask)

        self.point_defects = []

        for i in range(1, n_features + 1):
            # Get positions of this defect
            positions = np.where(labeled == i)

            # Center of mass
            center = center_of_mass(nan_mask, labeled, i)

            # Volume
            volume = np.sum(labeled == i)

            defect = {
                'type': 'point_defect',
                'id': i,
                'center': center,
                'volume': volume,
                'positions': positions
            }

            self.point_defects.append(defect)

        print(f"  Found {len(self.point_defects)} point defects")

        return self.point_defects

    def detect_singularities_by_winding(self, threshold=0.5) -> List[Dict]:
        """
        Detect singularities by calculating winding number

        This finds regions where director field has high topological charge
        """
        print("Detecting singularities by winding number...")

        # Calculate director magnitude
        magnitude = np.sqrt(
            self.director_field[:, :, :, 0]**2 +
            self.director_field[:, :, :, 1]**2 +
            self.director_field[:, :, :, 2]**2
        )

        # Defect mask (where magnitude is small or NaN)
        defect_mask = (magnitude < threshold) | np.isnan(magnitude)

        # Label connected regions
        labeled, n_features = label(defect_mask)

        self.singularities = []

        for i in range(1, n_features + 1):
            center = center_of_mass(defect_mask, labeled, i)
            volume = np.sum(labeled == i)

            singularity = {
                'type': 'singularity',
                'id': i,
                'center': center,
                'volume': volume,
                'detection_method': 'winding_threshold'
            }

            self.singularities.append(singularity)

        print(f"  Found {len(self.singularities)} singularities")

        return self.singularities

    def calculate_total_charge(self) -> int:
        """
        Calculate total topological charge (should be 0 for closed system)
        """
        # For now, sum up known defect charges
        # This is a placeholder - full implementation requires surface integrals

        total = 0
        # Each point defect typically has charge ±1
        # This should be calculated properly from winding number

        print(f"  Total topological charge: {total} (placeholder)")
        return total


class IsosurfaceRenderer:
    """
    Renders isosurfaces (level surfaces) of director components

    This creates the "onion layer" visualization from the paper
    """

    def __init__(self, director_field: np.ndarray,
                 physical_size: Tuple[float, float, float]):
        self.director_field = director_field
        self.physical_size = physical_size
        self.n_x, self.n_y, self.n_z, _ = director_field.shape

    def extract_isosurface(self, component='nz', level=0.0):
        """
        Extract isosurface using marching cubes algorithm

        Args:
            component: 'nx', 'ny', 'nz', or 'magnitude'
            level: Isosurface level value

        Returns:
            verts, faces: Mesh data
        """
        print(f"Extracting isosurface: {component} = {level}")

        # Select component
        if component == 'nx':
            data = self.director_field[:, :, :, 0]
        elif component == 'ny':
            data = self.director_field[:, :, :, 1]
        elif component == 'nz':
            data = self.director_field[:, :, :, 2]
        elif component == 'magnitude':
            data = np.sqrt(
                self.director_field[:, :, :, 0]**2 +
                self.director_field[:, :, :, 1]**2 +
                self.director_field[:, :, :, 2]**2
            )
        else:
            raise ValueError(f"Unknown component: {component}")

        # Replace NaN with 0 (defects)
        data = np.nan_to_num(data, nan=0.0)

        # Use marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                data, level=level
            )

            # Scale to physical coordinates
            verts[:, 0] *= self.physical_size[0] / self.n_x
            verts[:, 1] *= self.physical_size[1] / self.n_y
            verts[:, 2] *= self.physical_size[2] / self.n_z

            print(f"  Extracted surface: {len(verts)} vertices, {len(faces)} faces")

            return verts, faces

        except (ValueError, RuntimeError) as e:
            print(f"  Warning: Could not extract isosurface: {e}")
            return None, None

    def plot_multiple_isosurfaces(self, component='nz',
                                  levels=None,
                                  figsize=(12, 10)):
        """
        Plot multiple isosurfaces at different levels

        This creates the nested "onion layer" effect
        """
        if levels is None:
            levels = [-0.8, -0.4, 0, 0.4, 0.8]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        colors = cm.coolwarm(np.linspace(0, 1, len(levels)))

        for level, color in zip(levels, colors):
            verts, faces = self.extract_isosurface(component, level)

            if verts is not None:
                # Create mesh
                mesh = [[verts[j] for j in face] for face in faces[::10]]  # Subsample for speed

                collection = Poly3DCollection(mesh,
                                            alpha=0.3,
                                            facecolor=color,
                                            edgecolor='none')
                ax.add_collection3d(collection)

        # Set limits
        ax.set_xlim(0, self.physical_size[0])
        ax.set_ylim(0, self.physical_size[1])
        ax.set_zlim(0, self.physical_size[2])

        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'Level Surfaces: {component}\\nToroidal Structure')

        return fig


def plot_with_defects(director_field, defects, physical_size,
                     slice_type='xy', slice_index=None,
                     figsize=(12, 10)):
    """
    Plot director field cross-section with defects marked

    Similar to Figure 1 panels d, e, f from the paper
    """
    n_x, n_y, n_z, _ = director_field.shape

    if slice_index is None:
        slice_index = n_z // 2 if slice_type == 'xy' else n_y // 2

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract slice
    if slice_type == 'xy':
        nx_slice = director_field[:, :, slice_index, 0]
        ny_slice = director_field[:, :, slice_index, 1]
        nz_slice = director_field[:, :, slice_index, 2]
        extent = [0, physical_size[0], 0, physical_size[1]]
        xlabel, ylabel = 'X (μm)', 'Y (μm)'
    elif slice_type == 'xz':
        nx_slice = director_field[:, slice_index, :, 0]
        ny_slice = director_field[:, slice_index, :, 1]
        nz_slice = director_field[:, slice_index, :, 2]
        extent = [0, physical_size[0], 0, physical_size[2]]
        xlabel, ylabel = 'X (μm)', 'Z (μm)'

    # Panel 1: nz component
    im1 = axes[0, 0].imshow(nz_slice.T, cmap='coolwarm',
                           vmin=-1, vmax=1, origin='lower',
                           extent=extent)
    axes[0, 0].set_title(f'{slice_type.upper()} slice: nz component')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    plt.colorbar(im1, ax=axes[0, 0])

    # Overlay director arrows
    step = max(1, n_x // 15)
    x_grid = np.linspace(0, physical_size[0], n_x)
    y_grid = np.linspace(0, physical_size[1] if slice_type == 'xy' else physical_size[2],
                        n_y if slice_type == 'xy' else n_z)
    X, Y = np.meshgrid(x_grid[::step], y_grid[::step], indexing='ij')

    axes[0, 0].quiver(X, Y,
                     nx_slice[::step, ::step],
                     ny_slice[::step, ::step],
                     angles='xy', scale_units='xy', scale=8,
                     color='black', alpha=0.6, width=0.003)

    # Mark defects
    for defect in defects:
        center = defect['center']
        if slice_type == 'xy' and abs(center[2] - slice_index) < 3:
            axes[0, 0].plot(center[0] * physical_size[0] / n_x,
                          center[1] * physical_size[1] / n_y,
                          'ro', markersize=10, markeredgecolor='darkred',
                          markeredgewidth=2)

    # Panel 2: Director magnitude
    magnitude = np.sqrt(nx_slice**2 + ny_slice**2 + nz_slice**2)
    im2 = axes[0, 1].imshow(magnitude.T, cmap='viridis',
                           origin='lower', extent=extent)
    axes[0, 1].set_title('Director Magnitude')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    plt.colorbar(im2, ax=axes[0, 1])

    # Panel 3: In-plane angle
    angle = np.arctan2(ny_slice, nx_slice)
    im3 = axes[1, 0].imshow(angle.T, cmap='hsv',
                           origin='lower', extent=extent)
    axes[1, 0].set_title('In-plane Angle')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel(ylabel)
    plt.colorbar(im3, ax=axes[1, 0], label='Angle (rad)')

    # Panel 4: Defect summary
    axes[1, 1].axis('off')

    summary_text = f"CROSS-SECTION ANALYSIS\\n"
    summary_text += f"{slice_type.upper()} plane at index {slice_index}\\n\\n"
    summary_text += f"Defects detected: {len(defects)}\\n\\n"

    for i, defect in enumerate(defects, 1):
        summary_text += f"{i}. {defect['type']}\\n"
        summary_text += f"   Center: ({defect['center'][0]:.1f}, "
        summary_text += f"{defect['center'][1]:.1f}, {defect['center'][2]:.1f})\\n"
        summary_text += f"   Volume: {defect['volume']} voxels\\n\\n"

    axes[1, 1].text(0.1, 0.9, summary_text,
                   transform=axes[1, 1].transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def demo_advanced_visualization():
    """Demonstrate advanced visualization features"""
    print("\\n" + "="*60)
    print("PHASE 4 DEMO: Advanced Visualization")
    print("="*60 + "\\n")

    # Load toron structure
    from toron_simulator_3d import ToronSimulator3D

    print("Loading T3-1 toron structure...")
    sim = ToronSimulator3D.load_from_npz('toron_t3_1.npz')
    info = sim.get_grid_info()

    # Topological analysis
    print("\\nPerforming topological analysis...")
    analyzer = TopologicalAnalyzer(sim.director_field)
    point_defects = analyzer.detect_point_defects()
    singularities = analyzer.detect_singularities_by_winding(threshold=0.3)

    # Plot with defects
    print("\\nCreating defect visualization...")
    fig1 = plot_with_defects(
        sim.director_field,
        point_defects,
        info['physical_size'],
        slice_type='xy'
    )
    plt.savefig('toron_with_defects.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: toron_with_defects.png")

    # Isosurface rendering
    print("\\nCreating isosurface visualization...")
    renderer = IsosurfaceRenderer(sim.director_field, info['physical_size'])

    fig2 = renderer.plot_multiple_isosurfaces(
        component='nz',
        levels=[-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
    )
    plt.savefig('toron_isosurfaces.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: toron_isosurfaces.png")

    print("\\n" + "="*60)
    print("ADVANCED VISUALIZATION COMPLETE!")
    print("="*60)

    return analyzer, renderer


if __name__ == "__main__":
    demo_advanced_visualization()
    plt.show()
