"""
Visualization script for 3D toron structures
Creates comprehensive visualizations matching Nature Materials paper Figure 1
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from toron_simulator_3d import ToronSimulator3D, SimulationParams3D


def visualize_director_field_3d(simulator, slice_indices=None, figsize=(18, 12)):
    """
    Comprehensive 3D visualization of director field

    Shows:
    - 3D quiver plot of director field
    - Cross-sections (x-y, x-z, y-z planes)
    - Director components
    - Defect locations
    """
    director = simulator.director_field
    n_x, n_y, n_z, _ = director.shape

    # Get physical coordinates
    x = np.linspace(0, simulator.params.x_size, n_x)
    y = np.linspace(0, simulator.params.y_size, n_y)
    z = np.linspace(0, simulator.params.z_size, n_z)

    fig = plt.figure(figsize=figsize)

    # ========================================================================
    # Panel 1: 3D Quiver Plot (subsampled for clarity)
    # ========================================================================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Subsample for visualization
    step = max(1, n_x // 12)
    X, Y, Z = np.meshgrid(
        x[::step], y[::step], z[::step],
        indexing='ij'
    )

    nx = director[::step, ::step, ::step, 0]
    ny = director[::step, ::step, ::step, 1]
    nz = director[::step, ::step, ::step, 2]

    # Color by nz component
    colors = nz.flatten()
    norm = plt.Normalize(vmin=-1, vmax=1)

    ax1.quiver(X, Y, Z, nx, ny, nz,
               length=0.2, normalize=True,
               cmap='coolwarm', alpha=0.6,
               arrow_length_ratio=0.3)

    # Plot defects
    for defect in simulator.defect_locations:
        if defect['type'] == 'hyperbolic_point':
            pos = defect['physical_position']
            ax1.scatter(pos[0], pos[1], pos[2],
                       c='red', s=200, marker='o',
                       edgecolors='darkred', linewidths=2,
                       label='Point defect' if defect == simulator.defect_locations[0] else '')

    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.set_title(f'3D Director Field\\n{simulator.params.structure_type.upper()}')
    if simulator.defect_locations:
        ax1.legend()

    # ========================================================================
    # Panel 2: X-Y Midplane Cross-section
    # ========================================================================
    ax2 = fig.add_subplot(2, 3, 2)

    z_mid = n_z // 2
    nx_xy = director[:, :, z_mid, 0]
    ny_xy = director[:, :, z_mid, 1]
    nz_xy = director[:, :, z_mid, 2]

    # Color map by nz component
    im2 = ax2.imshow(nz_xy.T, extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                     origin='lower', cmap='coolwarm', vmin=-1, vmax=1)

    # Overlay director arrows
    step_xy = max(1, n_x // 15)
    X_2d, Y_2d = np.meshgrid(x[::step_xy], y[::step_xy], indexing='ij')
    ax2.quiver(X_2d, Y_2d,
               nx_xy[::step_xy, ::step_xy],
               ny_xy[::step_xy, ::step_xy],
               angles='xy', scale_units='xy', scale=10,
               color='black', alpha=0.6, width=0.003)

    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title(f'X-Y Midplane (z={z[z_mid]:.2f} μm)\\nnz component')
    plt.colorbar(im2, ax=ax2, label='nz')

    # ========================================================================
    # Panel 3: X-Z Cross-section
    # ========================================================================
    ax3 = fig.add_subplot(2, 3, 3)

    y_mid = n_y // 2
    nx_xz = director[:, y_mid, :, 0]
    ny_xz = director[:, y_mid, :, 1]
    nz_xz = director[:, y_mid, :, 2]

    # Color map by ny component
    im3 = ax3.imshow(ny_xz.T, extent=[0, simulator.params.x_size, 0, simulator.params.z_size],
                     origin='lower', cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Overlay director arrows (in x-z plane)
    step_xz = max(1, n_x // 15)
    X_xz, Z_xz = np.meshgrid(x[::step_xz], z[::step_xz], indexing='ij')
    ax3.quiver(X_xz, Z_xz,
               nx_xz[::step_xz, ::step_xz],
               nz_xz[::step_xz, ::step_xz],
               angles='xy', scale_units='xy', scale=10,
               color='black', alpha=0.6, width=0.003)

    ax3.set_xlabel('X (μm)')
    ax3.set_ylabel('Z (μm)')
    ax3.set_title(f'X-Z Cross-section (y={y[y_mid]:.2f} μm)\\nny component')
    plt.colorbar(im3, ax=ax3, label='ny')

    # ========================================================================
    # Panel 4: Director Component Profiles
    # ========================================================================
    ax4 = fig.add_subplot(2, 3, 4)

    # Extract director along central axis
    center_x = n_x // 2
    center_y = n_y // 2

    nx_profile = director[center_x, center_y, :, 0]
    ny_profile = director[center_x, center_y, :, 1]
    nz_profile = director[center_x, center_y, :, 2]

    ax4.plot(z, nx_profile, 'r-', linewidth=2, label='nx', alpha=0.8)
    ax4.plot(z, ny_profile, 'g-', linewidth=2, label='ny', alpha=0.8)
    ax4.plot(z, nz_profile, 'b-', linewidth=2, label='nz', alpha=0.8)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)

    ax4.set_xlabel('Z position (μm)')
    ax4.set_ylabel('Director component')
    ax4.set_title('Director Components Along Z-axis\\n(at center x, y)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 5: nz Component Distribution
    # ========================================================================
    ax5 = fig.add_subplot(2, 3, 5)

    nz_all = director[:, :, :, 2].flatten()
    nz_valid = nz_all[~np.isnan(nz_all)]

    ax5.hist(nz_valid, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(np.mean(nz_valid), color='red', linestyle='--',
                linewidth=2, label=f'Mean = {np.mean(nz_valid):.3f}')
    ax5.set_xlabel('nz value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of nz Component')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 6: Defect Summary
    # ========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Create text summary
    summary_text = f"Structure: {simulator.params.structure_type.upper()}\\n\\n"
    summary_text += f"Grid: {n_x} × {n_y} × {n_z}\\n"
    summary_text += f"Size: {simulator.params.x_size:.1f} × {simulator.params.y_size:.1f} × {simulator.params.z_size:.1f} μm³\\n"
    summary_text += f"Pitch: {simulator.params.pitch:.2f} μm\\n\\n"

    summary_text += "TOPOLOGICAL DEFECTS:\\n"
    summary_text += "-" * 40 + "\\n"

    total_charge = 0
    for i, defect in enumerate(simulator.defect_locations, 1):
        defect_type = defect['type']
        charge = defect.get('charge', 0)
        total_charge += charge

        summary_text += f"{i}. {defect_type}\\n"
        summary_text += f"   Charge: {charge:+d}\\n"

        if 'physical_position' in defect:
            pos = defect['physical_position']
            summary_text += f"   Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\\n"
        elif 'center' in defect:
            center = defect['center']
            radius = defect.get('radius', 'N/A')
            summary_text += f"   Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\\n"
            summary_text += f"   Radius: {radius}\\n"

        summary_text += "\\n"

    summary_text += "-" * 40 + "\\n"
    summary_text += f"TOTAL CHARGE: {total_charge:+d} (conserved!)\\n"

    # ax6.text(0.1, 0.95, summary_text,
    #          transform=ax6.transAxes,
    #          fontsize=10,
    #          verticalalignment='top',
    #          fontfamily='monospace',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_multiple_cross_sections(simulator, n_slices=5, figsize=(16, 10)):
    """
    Plot multiple cross-sections showing the toron structure
    Similar to Figure 1 panels d, e, f from the paper
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

        ax_top.set_title(f'z = {z[z_idx]:.2f} μm')
        ax_top.set_xlabel('X (μm)')
        if i == 0:
            ax_top.set_ylabel('Y (μm)')
        plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)

        # Bottom row: director magnitude
        ax_bot = axes[1, i]

        magnitude = np.sqrt(nx_slice**2 + ny_slice**2 + nz_slice**2)
        im2 = ax_bot.imshow(magnitude.T,
                           extent=[0, simulator.params.x_size, 0, simulator.params.y_size],
                           origin='lower', cmap='viridis')

        ax_bot.set_xlabel('X (μm)')
        if i == 0:
            ax_bot.set_ylabel('Y (μm)')
        plt.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04)

    plt.suptitle(f'Cross-sections of {simulator.params.structure_type.upper()} Structure',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def compare_toron_types(figsize=(18, 6)):
    """
    Compare T3-1, T3-2, and T3-3 structures side-by-side
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    structures = ['t3-1', 't3-2', 't3-3']
    titles = ['T3-1\\n(2 point defects)', 'T3-2\\n(1 point + 1 ring)', 'T3-3\\n(2 rings)']

    for col, (struct_type, title) in enumerate(zip(structures, titles)):
        # Generate structure
        params = SimulationParams3D(
            n_x=48, n_y=48, n_z=48,
            x_size=6.0, y_size=6.0, z_size=6.0,
            pitch=2.0,
            toron_radius=2.0,
            toron_center=(3.0, 3.0, 3.0),
            structure_type=struct_type
        )
        sim = ToronSimulator3D(params)
        sim.generate_structure()

        # Top row: X-Y midplane
        z_mid = sim.params.n_z // 2
        nz_xy = sim.director_field[:, :, z_mid, 2]
        nx_xy = sim.director_field[:, :, z_mid, 0]
        ny_xy = sim.director_field[:, :, z_mid, 1]

        im1 = axes[0, col].imshow(nz_xy.T, cmap='coolwarm', vmin=-1, vmax=1,
                                  extent=[0, 6, 0, 6], origin='lower')

        # Add arrows
        x = np.linspace(0, 6, 48)
        y = np.linspace(0, 6, 48)
        step = 4
        X, Y = np.meshgrid(x[::step], y[::step], indexing='ij')
        axes[0, col].quiver(X, Y, nx_xy[::step, ::step], ny_xy[::step, ::step],
                           angles='xy', scale_units='xy', scale=8,
                           color='black', alpha=0.6, width=0.003)

        axes[0, col].set_title(title)
        axes[0, col].set_xlabel('X (μm)')
        if col == 0:
            axes[0, col].set_ylabel('Y (μm)')
        plt.colorbar(im1, ax=axes[0, col], fraction=0.046)

        # Bottom row: X-Z cross-section
        y_mid = sim.params.n_y // 2
        ny_xz = sim.director_field[:, y_mid, :, 1]

        im2 = axes[1, col].imshow(ny_xz.T, cmap='coolwarm', vmin=-1, vmax=1,
                                  extent=[0, 6, 0, 6], origin='lower', aspect='auto')

        axes[1, col].set_xlabel('X (μm)')
        if col == 0:
            axes[1, col].set_ylabel('Z (μm)')
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046)

    plt.suptitle('Comparison of Toron Structures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("\\n" + "="*60)
    print("TORON VISUALIZATION SUITE")
    print("="*60 + "\\n")

    # Visualize T3-1 structure
    print("Generating T3-1 structure...")
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

    print("Creating comprehensive visualization...")
    fig1 = visualize_director_field_3d(sim_t3_1)
    plt.savefig('toron_t3_1_comprehensive.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: toron_t3_1_comprehensive.png")

    print("\\nCreating cross-section visualization...")
    fig2 = plot_multiple_cross_sections(sim_t3_1, n_slices=5)
    plt.savefig('toron_t3_1_cross_sections.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: toron_t3_1_cross_sections.png")

    print("\\nCreating comparison of all toron types...")
    fig3 = compare_toron_types()
    plt.savefig('toron_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: toron_comparison.png")

    print("\\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\\nGenerated files:")
    print("  - toron_t3_1_comprehensive.png")
    print("  - toron_t3_1_cross_sections.png")
    print("  - toron_comparison.png")
    print("="*60 + "\\n")

    plt.show()
