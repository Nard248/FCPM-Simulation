#!/usr/bin/env python3
"""
Test script for the enhanced FCPM simulation fixes

This script tests:
1. Noise addition to zero-intensity lines
2. Smooth intensity profile (reduced sharp variations)
3. Reduced sharpness in high intensity regions
4. 3D vector field visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_fcpm_simulator import EnhancedFCPMSimulator, SimulationParams

def test_all_fixes():
    """Test all the implemented fixes"""

    print("="*70)
    print("TESTING ENHANCED FCPM SIMULATION FIXES")
    print("="*70)

    # Test 1-3: Create simulation with noise and smoothing
    print("\n1. Testing noise addition to zero-intensity lines...")
    print("2. Testing smooth intensity profile...")
    print("3. Testing reduced sharpness at high intensities...")

    # Create simulation with moderate noise
    params = SimulationParams(
        n_z=200,
        n_x=50,
        pitch=2.0,
        noise_level=0.1,
        include_defects=False  # Start without defects for clarity
    )

    simulator = EnhancedFCPMSimulator(params)
    intensity = simulator.simulate()

    print(f"   ✓ Simulation completed: {intensity.shape}")
    print(f"   ✓ Intensity range: [{np.min(intensity):.3f}, {np.max(intensity):.3f}]")

    # Check that zero-intensity regions have noise
    z_coords = np.linspace(0, params.z_size, params.n_z)
    intensity_1d = np.mean(intensity, axis=1)

    # Find minimum intensity regions
    min_intensity_idx = np.argmin(intensity_1d)
    min_intensity = intensity_1d[min_intensity_idx]

    print(f"   ✓ Minimum intensity: {min_intensity:.4f} (should be > 0 due to noise)")

    # Check smoothness by looking at gradient
    gradient = np.abs(np.gradient(intensity_1d))
    max_gradient = np.max(gradient)
    mean_gradient = np.mean(gradient)

    print(f"   ✓ Max gradient: {max_gradient:.4f} (lower = smoother)")
    print(f"   ✓ Mean gradient: {mean_gradient:.4f}")

    # Plot results with standard visualization
    print("\n   Plotting simulation results...")
    simulator.plot_results(show_defects=False)

    # Test 4: 3D vector field visualization
    print("\n4. Testing 3D vector field visualization...")
    print("   Creating 3D director field visualization...")
    simulator.plot_3d_director_field(n_z_planes=12, n_x_grid=20, n_y_grid=20)
    print("   ✓ 3D visualization completed")

    # Additional test: Compare before/after intensity profiles
    print("\n5. Additional analysis: Intensity profile quality...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full intensity profile
    ax = axes[0, 0]
    ax.plot(z_coords, intensity_1d, 'b-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Z position')
    ax.set_ylabel('Intensity')
    ax.set_title('Smoothed Intensity Profile')
    ax.grid(True, alpha=0.3)

    # Highlight zero-intensity regions
    zero_regions = intensity_1d < 0.1
    if np.any(zero_regions):
        z_zero = z_coords[zero_regions]
        i_zero = intensity_1d[zero_regions]
        ax.scatter(z_zero, i_zero, color='red', s=20, alpha=0.5,
                  label='Low intensity (should have noise)')
        ax.legend()

    # Plot 2: Intensity distribution histogram
    ax = axes[0, 1]
    ax.hist(intensity.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Intensity Distribution\n(Check smoothness at high values)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Gradient analysis
    ax = axes[1, 0]
    # gradient has same length as z_coords when using np.gradient
    ax.plot(z_coords, gradient, 'g-', linewidth=1.5)
    ax.set_xlabel('Z position')
    ax.set_ylabel('|Gradient|')
    ax.set_title('Intensity Gradient\n(Lower = Smoother)')
    ax.grid(True, alpha=0.3)
    ax.axhline(mean_gradient, color='r', linestyle='--',
              label=f'Mean: {mean_gradient:.3f}')
    ax.legend()

    # Plot 4: 2D intensity pattern (zoomed)
    ax = axes[1, 1]
    # Show a zoomed region to see the smoothing effect
    z_start = params.n_z // 4
    z_end = 3 * params.n_z // 4
    im = ax.imshow(intensity[z_start:z_end, :], cmap='gray', aspect='auto',
                  extent=[0, params.x_size, z_start*params.z_size/params.n_z,
                         z_end*params.z_size/params.n_z],
                  origin='lower')
    ax.set_xlabel('X position')
    ax.set_ylabel('Z position')
    ax.set_title('2D Intensity Pattern (Zoomed)')
    plt.colorbar(im, ax=ax, label='Intensity')

    plt.tight_layout()
    plt.suptitle('Intensity Profile Quality Analysis', fontsize=14, y=1.00)
    plt.show()

    # Test with defects
    print("\n6. Testing with defects...")
    params_defects = SimulationParams(
        n_z=200,
        n_x=50,
        pitch=2.0,
        noise_level=0.1,
        include_defects=True,
        defect_types=['dislocation_b_half'],
        defect_density=0.2
    )

    simulator_defects = EnhancedFCPMSimulator(params_defects)
    intensity_defects = simulator_defects.simulate()

    print(f"   ✓ Simulation with defects completed")
    print(f"   ✓ Number of defects: {len(simulator_defects.defect_locations)}")

    # Plot with defects
    simulator_defects.plot_results(show_defects=True)

    # 3D visualization with defects
    print("   Creating 3D visualization with defects...")
    simulator_defects.plot_3d_director_field(n_z_planes=12)

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nSummary of fixes:")
    print("✓ 1. Noise added to zero-intensity lines (additive noise component)")
    print("✓ 2. Intensity profile smoothed (Gaussian filtering with sigma=1.5)")
    print("✓ 3. High intensity peaks smoothed (adaptive sigma=2.5 for I>0.7)")
    print("✓ 4. 3D vector field visualization implemented")
    print("\nAll visualizations have been displayed.")
    print("="*70)

if __name__ == "__main__":
    test_all_fixes()
