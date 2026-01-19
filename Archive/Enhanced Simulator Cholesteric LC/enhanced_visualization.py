"""
Enhanced Visualization Tools for Toron Structures
Matching Nature Materials Figure 1 style with onion-like nested surfaces

Features:
- Nested toroidal isosurfaces (onion structure)
- Multiple rendering styles
- Topovec-compatible output
- Publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from skimage import measure
from typing import Tuple, List, Optional
import json


def create_onion_visualization(simulator, levels=None, figsize=(14, 12),
                               alpha=0.15, view_angles=(30, 45)):
    """
    Create onion-like nested surface visualization
    Matching Nature Materials Figure 1a-f style

    Args:
        simulator: ToronSimulator3D instance
        levels: List of nz values for isosurfaces (default: 7 levels from -0.9 to 0.9)
        figsize: Figure size
        alpha: Transparency of surfaces
        view_angles: (elevation, azimuth) viewing angles

    Returns:
        fig: Matplotlib figure
    """
    from advanced_visualization import IsosurfaceRenderer

    if levels is None:
        levels = [-0.9, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8]

    info = simulator.get_grid_info()
    renderer = IsosurfaceRenderer(simulator.director_field, info['physical_size'])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create nested surfaces with gradient coloring
    colors = cm.coolwarm(np.linspace(0, 1, len(levels)))

    print(f"Creating onion visualization with {len(levels)} nested surfaces...")

    for i, (level, color) in enumerate(zip(levels, colors)):
        print(f"  Rendering surface {i+1}/{len(levels)}: nz = {level:.2f}")

        verts, faces = renderer.extract_isosurface('nz', level)

        if verts is not None and len(faces) > 0:
            # Subsample faces for performance (every 10th face)
            subsample = max(1, len(faces) // 2000)  # Limit to ~2000 faces per surface
            mesh = [[verts[j] for j in face] for face in faces[::subsample]]

            if mesh:
                collection = Poly3DCollection(
                    mesh,
                    alpha=alpha,
                    facecolor=color,
                    edgecolor='none',
                    linewidths=0
                )
                ax.add_collection3d(collection)

    # Add defects as red spheres
    from advanced_visualization import TopologicalAnalyzer
    analyzer = TopologicalAnalyzer(simulator.director_field)
    defects = analyzer.detect_point_defects()

    for defect in defects:
        center = defect['center']
        x = center[0] * info['physical_size'][0] / simulator.params.n_x
        y = center[1] * info['physical_size'][1] / simulator.params.n_y
        z = center[2] * info['physical_size'][2] / simulator.params.n_z
        ax.scatter(x, y, z, c='red', s=200, marker='o',
                  edgecolors='darkred', linewidths=3, alpha=0.9)

    # Set viewing parameters
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_zlabel('Z (μm)', fontsize=12)
    ax.set_xlim(0, info['physical_size'][0])
    ax.set_ylim(0, info['physical_size'][1])
    ax.set_zlim(0, info['physical_size'][2])

    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    # Title with structure info
    title = f"{simulator.params.structure_type.upper()} Toron Structure\n"
    title += f"Nested Isosurfaces (nz levels)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add subtle grid
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    print("✓ Onion visualization complete")
    return fig


def plot_cross_sections_enhanced(simulator, n_slices=5, figsize=(18, 10)):
    """
    Enhanced cross-section visualization

    Top row: nz component with director field
    Bottom row: In-plane director orientation (angle map)

    This replaces the useless magnitude plot with orientation information
    """
    director = simulator.director_field
    n_x, n_y, n_z, _ = director.shape

    x = np.linspace(0, simulator.params.x_size, n_x)
    y = np.linspace(0, simulator.params.y_size, n_y)
    z = np.linspace(0, simulator.params.z_size, n_z)

    fig, axes = plt.subplots(2, n_slices, figsize=figsize)

    # Z-slices (x-y planes)
    z_indices = np.linspace(0, n_z-1, n_slices, dtype=int)

    for i, z_idx in enumerate(z_indices):
        # Top row: nz component with director arrows
        ax_top = axes[0, i]

        nx_slice = director[:, :, z_idx, 0]
        ny_slice = director[:, :, z_idx, 1]
        nz_slice = director[:, :, z_idx, 2]

        im = ax_top.imshow(nz_slice.T,
                          extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                          origin='lower', cmap='coolwarm', vmin=-1, vmax=1)

        # Overlay director arrows
        step = max(1, n_x // 8)
        X_2d, Y_2d = np.meshgrid(x[::step], y[::step], indexing='ij')
        ax_top.quiver(X_2d, Y_2d,
                     nx_slice[::step, ::step],
                     ny_slice[::step, ::step],
                     angles='xy', scale_units='xy', scale=8,
                     color='black', alpha=0.7, width=0.004)

        ax_top.set_title(f'z = {z[z_idx]:.2f} μm', fontweight='bold')
        ax_top.set_xlabel('X (μm)')
        if i == 0:
            ax_top.set_ylabel('Y (μm)')
            ax_top.text(-0.15, 0.5, 'nz component', transform=ax_top.transAxes,
                       rotation=90, va='center', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04, label='nz')

        # Bottom row: In-plane orientation angle (FIXED - was showing useless magnitude!)
        ax_bot = axes[1, i]

        # Calculate in-plane director angle
        angle = np.arctan2(ny_slice, nx_slice)

        # Mask out defect regions (NaN values)
        angle_masked = np.ma.masked_where(np.isnan(nx_slice), angle)

        im2 = ax_bot.imshow(angle_masked.T,
                           extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                           origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)

        ax_bot.set_xlabel('X (μm)')
        if i == 0:
            ax_bot.set_ylabel('Y (μm)')
            ax_bot.text(-0.15, 0.5, 'In-plane angle', transform=ax_bot.transAxes,
                       rotation=90, va='center', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04)
        cbar.set_label('φ (rad)', rotation=0, labelpad=15)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    fig.suptitle(f'Cross-sections: {simulator.params.structure_type.upper()} Structure',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def create_comprehensive_figure_fixed(simulator, optical_sim=None, config=None,
                                      figsize=(22, 14)):
    """
    Fixed comprehensive figure with no text overlap

    Relocates the summary text to avoid covering plots
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib import cm
    from skimage import measure
    from advanced_visualization import TopologicalAnalyzer, IsosurfaceRenderer

    fig = plt.figure(figsize=figsize)

    # Modified grid: more space for summary
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.05, right=0.98, top=0.94, bottom=0.05)

    info = simulator.get_grid_info()

    # ========================================================================
    # Panel 1: 3D Onion Structure (takes 2x2 grid)
    # ========================================================================
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')

    renderer = IsosurfaceRenderer(simulator.director_field, info['physical_size'])
    levels = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    colors_3d = cm.coolwarm(np.linspace(0, 1, len(levels)))

    for level, color in zip(levels, colors_3d):
        verts, faces = renderer.extract_isosurface('nz', level)
        if verts is not None and len(faces) > 0:
            mesh = [[verts[j] for j in face] for face in faces[::25]]
            if mesh:
                collection = Poly3DCollection(mesh, alpha=0.2,
                                            facecolor=color, edgecolor='none')
                ax_3d.add_collection3d(collection)

    # Add defects
    analyzer = TopologicalAnalyzer(simulator.director_field)
    defects = analyzer.detect_point_defects()

    for defect in defects:
        center = defect['center']
        x = center[0] * info['physical_size'][0] / simulator.params.n_x
        y = center[1] * info['physical_size'][1] / simulator.params.n_y
        z = center[2] * info['physical_size'][2] / simulator.params.n_z
        ax_3d.scatter(x, y, z, c='red', s=150, marker='o',
                     edgecolors='darkred', linewidths=2)

    ax_3d.set_xlabel('X (μm)', fontsize=10)
    ax_3d.set_ylabel('Y (μm)', fontsize=10)
    ax_3d.set_zlabel('Z (μm)', fontsize=10)
    ax_3d.set_title('Nested Isosurfaces\n(Onion Structure)',
                   fontweight='bold', fontsize=11)
    ax_3d.set_xlim(0, info['physical_size'][0])
    ax_3d.set_ylim(0, info['physical_size'][1])
    ax_3d.set_zlim(0, info['physical_size'][2])
    ax_3d.view_init(elev=25, azim=45)

    # ========================================================================
    # Panels 2-3: Cross-sections
    # ========================================================================
    x_grid = np.linspace(0, simulator.params.x_size, simulator.params.n_x)
    y_grid = np.linspace(0, simulator.params.y_size, simulator.params.n_y)
    step = max(1, simulator.params.n_x // 12)
    X, Y = np.meshgrid(x_grid[::step], y_grid[::step], indexing='ij')

    # X-Y midplane
    ax_xy = fig.add_subplot(gs[0, 2])
    z_mid = simulator.params.n_z // 2
    nz_xy = simulator.director_field[:, :, z_mid, 2]
    nx_xy = simulator.director_field[:, :, z_mid, 0]
    ny_xy = simulator.director_field[:, :, z_mid, 1]

    im_xy = ax_xy.imshow(nz_xy.T, cmap='coolwarm', vmin=-1, vmax=1,
                        extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                        origin='lower')
    ax_xy.quiver(X, Y, nx_xy[::step, ::step], ny_xy[::step, ::step],
                angles='xy', scale_units='xy', scale=8,
                color='black', alpha=0.6, width=0.003)
    ax_xy.set_title('X-Y Midplane', fontweight='bold', fontsize=10)
    ax_xy.set_xlabel('X (μm)', fontsize=9)
    ax_xy.set_ylabel('Y (μm)', fontsize=9)
    plt.colorbar(im_xy, ax=ax_xy, fraction=0.046, label='nz')

    # X-Z cross-section
    ax_xz = fig.add_subplot(gs[0, 3])
    y_mid = simulator.params.n_y // 2
    ny_xz = simulator.director_field[:, y_mid, :, 1]

    im_xz = ax_xz.imshow(ny_xz.T, cmap='coolwarm', vmin=-1, vmax=1,
                        extent=[0, simulator.params.x_size, 0, simulator.params.z_size],
                        origin='lower', aspect='auto')
    ax_xz.set_title('X-Z Section', fontweight='bold', fontsize=10)
    ax_xz.set_xlabel('X (μm)', fontsize=9)
    ax_xz.set_ylabel('Z (μm)', fontsize=9)
    plt.colorbar(im_xz, ax=ax_xz, fraction=0.046, label='ny')

    # ========================================================================
    # Panels 4-6: Z-slices
    # ========================================================================
    z_indices = [simulator.params.n_z//4, simulator.params.n_z//2,
                 3*simulator.params.n_z//4]
    z_coords = [idx * simulator.params.z_size / simulator.params.n_z
                for idx in z_indices]

    for idx, (z_idx, z_coord) in enumerate(zip(z_indices, z_coords)):
        ax = fig.add_subplot(gs[1, 2+idx])

        nz_slice = simulator.director_field[:, :, z_idx, 2]
        nx_slice = simulator.director_field[:, :, z_idx, 0]
        ny_slice = simulator.director_field[:, :, z_idx, 1]

        im = ax.imshow(nz_slice.T, cmap='coolwarm', vmin=-1, vmax=1,
                      extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                      origin='lower')
        ax.quiver(X, Y, nx_slice[::step, ::step], ny_slice[::step, ::step],
                 angles='xy', scale_units='xy', scale=8,
                 color='black', alpha=0.6, width=0.003)
        ax.set_title(f'z={z_coord:.2f} μm', fontweight='bold', fontsize=10)
        ax.set_xlabel('X (μm)', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Y (μm)', fontsize=9)

    # ========================================================================
    # Panels 7-8: FCPM images
    # ========================================================================
    if optical_sim is not None and hasattr(optical_sim, 'intensity_image'):
        intensity = optical_sim.intensity_image

        ax_fcpm1 = fig.add_subplot(gs[2, 0])
        im_fcpm1 = ax_fcpm1.imshow(intensity.T, cmap='gray',
                                  extent=[0, simulator.params.x_size,
                                         0, simulator.params.y_size],
                                  origin='lower')
        ax_fcpm1.set_title('FCPM Intensity', fontweight='bold', fontsize=10)
        ax_fcpm1.set_xlabel('X (μm)', fontsize=9)
        ax_fcpm1.set_ylabel('Y (μm)', fontsize=9)
        plt.colorbar(im_fcpm1, ax=ax_fcpm1, fraction=0.046)

        ax_fcpm2 = fig.add_subplot(gs[2, 1])
        im_fcpm2 = ax_fcpm2.imshow(intensity.T**0.5, cmap='gray',
                                  extent=[0, simulator.params.x_size,
                                         0, simulator.params.y_size],
                                  origin='lower')
        ax_fcpm2.set_title('Enhanced Contrast', fontweight='bold', fontsize=10)
        ax_fcpm2.set_xlabel('X (μm)', fontsize=9)
        ax_fcpm2.set_ylabel('Y (μm)', fontsize=9)
        plt.colorbar(im_fcpm2, ax=ax_fcpm2, fraction=0.046)

    # ========================================================================
    # Panel 9: Summary (FIXED - Now in right column, no overlap!)
    # ========================================================================
    ax_summary = fig.add_subplot(gs[0:2, 4])  # Right column, full height
    ax_summary.axis('off')

    # Build summary text
    summary_text = f"STRUCTURE SUMMARY\n"
    summary_text += "="*40 + "\n\n"

    if config:
        summary_text += f"Experiment:\n{config.experiment_name}\n\n"

    summary_text += f"Type: {simulator.params.structure_type.upper()}\n"
    summary_text += f"Grid: {simulator.params.n_x}×{simulator.params.n_y}×{simulator.params.n_z}\n"
    summary_text += f"Size: {info['physical_size'][0]:.1f}×{info['physical_size'][1]:.1f}×{info['physical_size'][2]:.1f} μm³\n"
    summary_text += f"Pitch: {simulator.params.pitch:.2f} μm\n\n"

    if simulator.params.structure_type.startswith('t3'):
        summary_text += f"Toron radius: {simulator.params.toron_radius:.2f} μm\n"
        summary_text += f"Center: ({simulator.params.toron_center[0]:.1f}, "
        summary_text += f"{simulator.params.toron_center[1]:.1f}, "
        summary_text += f"{simulator.params.toron_center[2]:.1f}) μm\n\n"

    if config:
        summary_text += "OPTICAL:\n"
        summary_text += f"λ = {config.wavelength:.3f} μm\n"
        summary_text += f"no = {config.no:.2f}\n"
        summary_text += f"ne = {config.ne:.2f}\n"
        summary_text += f"Δn = {config.ne - config.no:.3f}\n\n"

    if simulator.defect_locations:
        summary_text += f"DEFECTS ({len(simulator.defect_locations)}):\n"
        for i, defect in enumerate(simulator.defect_locations[:4], 1):
            summary_text += f"{i}. {defect['type']}\n"
            summary_text += f"   q={defect.get('charge', '?')}\n"

        total_charge = sum(d.get('charge', 0) for d in simulator.defect_locations)
        summary_text += f"\nTotal: q={total_charge:+d}\n"
        summary_text += "(conserved)\n\n"

    if config and config.description:
        summary_text += f"\nNOTE:\n{config.description[:80]}...\n" if len(config.description) > 80 else f"\nNOTE:\n{config.description}\n"

    ax_summary.text(0.05, 0.98, summary_text,
                   transform=ax_summary.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.8',
                            facecolor='lightblue', alpha=0.3,
                            edgecolor='navy', linewidth=1.5))

    # Overall title
    title = config.experiment_name if config else f"{simulator.params.structure_type.upper()} Toron"
    fig.suptitle(f'Comprehensive Analysis: {title}',
                fontsize=16, fontweight='bold')

    return fig


def export_for_topovec(simulator, filename):
    """
    Export director field in topovec-compatible format

    Topovec expects:
    - Director field as (nx, ny, nz, 3) array
    - Metadata with grid info
    """
    data = {
        'director_field': simulator.director_field,
        'grid_size': (simulator.params.n_x, simulator.params.n_y, simulator.params.n_z),
        'physical_size': (simulator.params.x_size, simulator.params.y_size, simulator.params.z_size),
        'pitch': simulator.params.pitch,
        'structure_type': simulator.params.structure_type
    }

    np.savez_compressed(filename, **data)
    print(f"✓ Exported to topovec-compatible format: {filename}")
    return filename


# Demonstration
if __name__ == "__main__":
    from toron_simulator_3d import ToronSimulator3D, SimulationParams3D

    print("\n" + "="*70)
    print("ENHANCED VISUALIZATION DEMO")
    print("="*70 + "\n")

    # Generate a toron
    params = SimulationParams3D(
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0),
        structure_type='t3-1'
    )

    sim = ToronSimulator3D(params)
    sim.generate_structure()

    # Create onion visualization
    print("\n1. Creating onion-like nested surface visualization...")
    fig1 = create_onion_visualization(sim)
    fig1.savefig('demo_onion_structure.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_onion_structure.png")

    # Create enhanced cross-sections
    print("\n2. Creating enhanced cross-sections (fixed bottom row)...")
    fig2 = plot_cross_sections_enhanced(sim)
    fig2.savefig('demo_cross_sections_enhanced.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: demo_cross_sections_enhanced.png")

    # Export for topovec
    print("\n3. Exporting for topovec...")
    export_for_topovec(sim, 'demo_topovec_export.npz')

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70 + "\n")

    plt.show()
