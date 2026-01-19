#!/usr/bin/env python3
"""
Basic FCPM Simulation Example

This example demonstrates:
1. Creating a synthetic director field (cholesteric)
2. Simulating FCPM measurements
3. Visualizing the results

Run with: python -m fcpm.examples.basic_simulation
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the fcpm package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fcpm


def main():
    print("=" * 60)
    print("FCPM Basic Simulation Example")
    print("=" * 60)

    # 1. Create a cholesteric (twisted) director field
    print("\n1. Creating cholesteric director field...")
    shape = (64, 64, 32)  # (ny, nx, nz)
    pitch = 8.0  # Helical pitch in voxels

    director = fcpm.create_cholesteric_director(shape, pitch=pitch, axis='z')
    print(f"   Director field shape: {director.shape}")
    print(f"   Normalized: {director.is_normalized()}")

    # 2. Set up FCPM simulator
    print("\n2. Setting up FCPM simulator...")
    simulator = fcpm.FCPMSimulator(director)
    print(f"   Polarization angles: {np.degrees(simulator.angles).tolist()}°")

    # 3. Simulate FCPM intensities
    print("\n3. Simulating FCPM intensities...")
    I_fcpm = simulator.simulate()
    print(f"   Generated {len(I_fcpm)} intensity images")

    for angle, intensity in I_fcpm.items():
        print(f"   α = {np.degrees(angle):6.1f}°: "
              f"min={intensity.min():.4f}, max={intensity.max():.4f}")

    # 4. Visualize results
    print("\n4. Creating visualizations...")

    # Create a figure with multiple panels
    fig = plt.figure(figsize=(15, 10))

    # Panel 1: Director field (middle z-slice)
    ax1 = fig.add_subplot(2, 3, 1)
    z_mid = shape[2] // 2
    nx_slice, ny_slice, nz_slice = director.slice_z(z_mid)

    # Subsample for cleaner visualization
    step = 4
    Y, X = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
    nx_sub = nx_slice[::step, ::step]
    ny_sub = ny_slice[::step, ::step]

    colors = np.arctan2(ny_sub, nx_sub) % np.pi
    ax1.quiver(X, Y, nx_sub, ny_sub, colors, cmap='hsv',
               scale=1.2, headwidth=3, pivot='middle')
    ax1.set_aspect('equal')
    ax1.set_title(f'Director field at z={z_mid}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Panels 2-5: FCPM intensities
    angles = sorted(I_fcpm.keys())
    for i, angle in enumerate(angles[:4]):
        ax = fig.add_subplot(2, 3, i + 2)
        im = ax.imshow(I_fcpm[angle][:, :, z_mid], cmap='gray', origin='lower')
        ax.set_title(f'I(α = {np.degrees(angle):.0f}°)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 6: RGB visualization
    ax6 = fig.add_subplot(2, 3, 6)

    # Create HSV image
    angle_map = np.arctan2(ny_slice, nx_slice) % np.pi
    hue = angle_map / np.pi
    saturation = np.sqrt(nx_slice**2 + ny_slice**2)
    value = 1.0 - np.abs(nz_slice) * 0.5

    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    ax6.imshow(rgb, origin='lower')
    ax6.set_title('Director RGB encoding')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')

    plt.suptitle('FCPM Simulation of Cholesteric Liquid Crystal', fontsize=14)
    plt.tight_layout()
    plt.savefig('fcpm_simulation_example.png', dpi=150, bbox_inches='tight')
    print("   Saved figure to: fcpm_simulation_example.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
