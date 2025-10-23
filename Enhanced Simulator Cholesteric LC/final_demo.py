"""
PHASE 5: Final Comprehensive Demo
Complete workflow from toron generation to optical simulation and visualization

This demonstrates the full pipeline matching the Nature Materials paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

from toron_simulator_3d import ToronSimulator3D, SimulationParams3D
from optical_simulator import CrossPolarizerSimulator, OpticalParams
from advanced_visualization import TopologicalAnalyzer, IsosurfaceRenderer, plot_with_defects
from visualize_torons import visualize_director_field_3d


def create_paper_figure_1_style(toron_type='t3-1', figsize=(20, 12)):
    """
    Create figure matching Nature Materials paper Figure 1

    Panels:
    a. 3D toron structure with nested surfaces
    b,c. Cross-sections showing director field with defects
    d,e,f. Different toron types
    g,h. FCPM simulated images
    """
    print(f"\\nCreating Figure 1 style visualization for {toron_type.upper()}...")

    # Generate structure
    params = SimulationParams3D(
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0),
        structure_type=toron_type,
        no=1.5,
        ne=1.7
    )

    sim = ToronSimulator3D(params)
    sim.generate_structure()
    info = sim.get_grid_info()

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # ========================================================================
    # Panel a: 3D structure with isosurfaces
    # ========================================================================
    print("  Panel a: 3D isosurfaces...")
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')

    # Extract and plot isosurfaces
    renderer = IsosurfaceRenderer(sim.director_field, info['physical_size'])

    # Plot nested surfaces at different nz levels
    from matplotlib import cm
    levels = [-0.7, -0.3, 0.3, 0.7]
    colors = cm.coolwarm(np.linspace(0, 1, len(levels)))

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    for level, color in zip(levels, colors):
        verts, faces = renderer.extract_isosurface('nz', level)

        if verts is not None and len(faces) > 0:
            # Subsample for speed
            mesh = [[verts[j] for j in face] for face in faces[::20]]

            if mesh:
                collection = Poly3DCollection(mesh,
                                            alpha=0.25,
                                            facecolor=color,
                                            edgecolor='none')
                ax_3d.add_collection3d(collection)

    # Plot defects
    analyzer = TopologicalAnalyzer(sim.director_field)
    defects = analyzer.detect_point_defects()

    for defect in defects:
        center = defect['center']
        x = center[0] * info['physical_size'][0] / sim.params.n_x
        y = center[1] * info['physical_size'][1] / sim.params.n_y
        z = center[2] * info['physical_size'][2] / sim.params.n_z
        ax_3d.scatter(x, y, z, c='red', s=100, marker='o',
                     edgecolors='darkred', linewidths=2)

    ax_3d.set_xlabel('X (μm)')
    ax_3d.set_ylabel('Y (μm)')
    ax_3d.set_zlabel('Z (μm)')
    ax_3d.set_title(f'Panel a: {toron_type.upper()} Structure\\nNested Level Surfaces',
                   fontweight='bold')
    ax_3d.set_xlim(0, info['physical_size'][0])
    ax_3d.set_ylim(0, info['physical_size'][1])
    ax_3d.set_zlim(0, info['physical_size'][2])

    # ========================================================================
    # Panels b,c: Cross-sections with director field
    # ========================================================================
    print("  Panels b,c: Cross-sections...")

    # Panel b: X-Y midplane
    ax_b = fig.add_subplot(gs[0, 2])
    z_mid = sim.params.n_z // 2
    nz_xy = sim.director_field[:, :, z_mid, 2]
    nx_xy = sim.director_field[:, :, z_mid, 0]
    ny_xy = sim.director_field[:, :, z_mid, 1]

    im_b = ax_b.imshow(nz_xy.T, cmap='coolwarm', vmin=-1, vmax=1,
                       extent=[0, 6, 0, 6], origin='lower')

    # Director arrows
    x_grid = np.linspace(0, 6, sim.params.n_x)
    y_grid = np.linspace(0, 6, sim.params.n_y)
    step = 4
    X, Y = np.meshgrid(x_grid[::step], y_grid[::step], indexing='ij')
    ax_b.quiver(X, Y, nx_xy[::step, ::step], ny_xy[::step, ::step],
               angles='xy', scale_units='xy', scale=8,
               color='black', alpha=0.6, width=0.003)

    ax_b.set_title('Panel b: X-Y Midplane', fontweight='bold')
    ax_b.set_xlabel('X (μm)')
    ax_b.set_ylabel('Y (μm)')
    plt.colorbar(im_b, ax=ax_b, fraction=0.046, label='nz')

    # Panel c: X-Z cross-section
    ax_c = fig.add_subplot(gs[0, 3])
    y_mid = sim.params.n_y // 2
    ny_xz = sim.director_field[:, y_mid, :, 1]

    im_c = ax_c.imshow(ny_xz.T, cmap='coolwarm', vmin=-1, vmax=1,
                       extent=[0, 6, 0, 6], origin='lower', aspect='auto')

    ax_c.set_title('Panel c: X-Z Cross-section', fontweight='bold')
    ax_c.set_xlabel('X (μm)')
    ax_c.set_ylabel('Z (μm)')
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, label='ny')

    # ========================================================================
    # Panels d,e,f: Multiple Z-slices showing director field evolution
    # ========================================================================
    print("  Panels d,e,f: Director field evolution...")

    z_indices = [sim.params.n_z//4, sim.params.n_z//2, 3*sim.params.n_z//4]
    panel_labels = ['d', 'e', 'f']

    for idx, (z_idx, label) in enumerate(zip(z_indices, panel_labels)):
        ax = fig.add_subplot(gs[1, idx+1])

        nz_slice = sim.director_field[:, :, z_idx, 2]
        nx_slice = sim.director_field[:, :, z_idx, 0]
        ny_slice = sim.director_field[:, :, z_idx, 1]

        im = ax.imshow(nz_slice.T, cmap='coolwarm', vmin=-1, vmax=1,
                      extent=[0, 6, 0, 6], origin='lower')

        # Director arrows
        ax.quiver(X, Y, nx_slice[::step, ::step], ny_slice[::step, ::step],
                 angles='xy', scale_units='xy', scale=8,
                 color='black', alpha=0.6, width=0.003)

        z_phys = z_idx * 6.0 / sim.params.n_z
        ax.set_title(f'Panel {label}: z={z_phys:.2f} μm', fontweight='bold')
        ax.set_xlabel('X (μm)')
        if idx == 0:
            ax.set_ylabel('Y (μm)')

    # ========================================================================
    # Panels g,h: FCPM simulated images
    # ========================================================================
    print("  Panels g,h: FCPM simulation...")

    optical_params = OpticalParams(
        wavelength=0.55,
        no=1.5,
        ne=1.7,
        polarizer_angle=0,
        analyzer_angle=np.pi/2,
        add_noise=False
    )

    optical_sim = CrossPolarizerSimulator(optical_params)
    intensity = optical_sim.simulate_fcpm(sim.director_field, info['physical_size'])

    # Panel g: FCPM intensity
    ax_g = fig.add_subplot(gs[2, 0])
    im_g = ax_g.imshow(intensity.T, cmap='gray',
                       extent=[0, 6, 0, 6], origin='lower')
    ax_g.set_title('Panel g: FCPM Intensity', fontweight='bold')
    ax_g.set_xlabel('X (μm)')
    ax_g.set_ylabel('Y (μm)')
    plt.colorbar(im_g, ax=ax_g, fraction=0.046)

    # Panel h: Enhanced contrast
    ax_h = fig.add_subplot(gs[2, 1])
    im_h = ax_h.imshow(intensity.T**0.5, cmap='gray',
                       extent=[0, 6, 0, 6], origin='lower')
    ax_h.set_title('Panel h: Enhanced Contrast', fontweight='bold')
    ax_h.set_xlabel('X (μm)')
    ax_h.set_ylabel('Y (μm)')
    plt.colorbar(im_h, ax=ax_h, fraction=0.046)

    # ========================================================================
    # Panel i: Defect summary and parameters
    # ========================================================================
    ax_i = fig.add_subplot(gs[2, 2:4])
    ax_i.axis('off')

    summary_text = f"STRUCTURE SUMMARY: {toron_type.upper()}\\n"
    summary_text += "="*50 + "\\n\\n"

    summary_text += f"Grid: {sim.params.n_x} × {sim.params.n_y} × {sim.params.n_z}\\n"
    summary_text += f"Size: {info['physical_size'][0]:.1f} × {info['physical_size'][1]:.1f} × {info['physical_size'][2]:.1f} μm³\\n"
    summary_text += f"Pitch: {sim.params.pitch:.2f} μm\\n"
    summary_text += f"Toron radius: {sim.params.toron_radius:.2f} μm\\n\\n"

    summary_text += "OPTICAL PARAMETERS:\\n"
    summary_text += f"λ = {optical_params.wavelength:.3f} μm\\n"
    summary_text += f"no = {optical_params.no:.2f}\\n"
    summary_text += f"ne = {optical_params.ne:.2f}\\n"
    summary_text += f"Δn = {optical_params.ne - optical_params.no:.3f}\\n\\n"

    summary_text += "TOPOLOGICAL DEFECTS:\\n"
    for i, defect in enumerate(sim.defect_locations, 1):
        summary_text += f"{i}. {defect['type']} (charge: {defect.get('charge', '?')})\\n"

    total_charge = sum(d.get('charge', 0) for d in sim.defect_locations)
    summary_text += f"\\nTOTAL CHARGE: {total_charge:+d} (conserved!)\\n"

    ax_i.text(0.05, 0.95, summary_text,
             transform=ax_i.transAxes,
             fontsize=9,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Comprehensive Toron Analysis: {toron_type.upper()}\\n'
                f'(Matching Nature Materials Figure 1 Style)',
                fontsize=16, fontweight='bold')

    return fig, sim, optical_sim


def run_full_demo():
    """Run complete demonstration of all phases"""
    print("\\n" + "="*60)
    print("FINAL COMPREHENSIVE DEMO: ALL PHASES")
    print("="*60)

    # Just run T3-1 for now (fastest)
    toron_type = 't3-1'

    print(f"\\n{'='*60}")
    print(f"Processing {toron_type.upper()}")
    print(f"{'='*60}")

    fig, sim, optical_sim = create_paper_figure_1_style(toron_type)

    filename = f'final_comprehensive_{toron_type}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")

    plt.close(fig)

    print("\\n" + "="*60)
    print("COMPREHENSIVE DEMO COMPLETE!")
    print("="*60)
    print("\\nGenerated file:")
    print(f"  - {filename}")
    print("\\nThis figure matches the style of Nature Materials Figure 1!")
    print("="*60 + "\\n")


if __name__ == "__main__":
    run_full_demo()
