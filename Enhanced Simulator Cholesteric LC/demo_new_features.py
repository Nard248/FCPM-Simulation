#!/usr/bin/env python3
"""
Quick demonstration of the new FCPM simulation features

This script showcases:
1. Improved noise model (affects zero-intensity lines)
2. Smooth intensity profile (no sharp variations)
3. Reduced sharpness at high intensities
4. 3D vector field visualization

Run this to see all the improvements in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_fcpm_simulator import EnhancedFCPMSimulator, SimulationParams

def main():
    print("="*70)
    print("FCPM SIMULATION - NEW FEATURES DEMONSTRATION")
    print("="*70)

    # Create an enhanced simulation with all fixes
    print("\nCreating enhanced FCPM simulation...")
    print("- Pitch: 2.0")
    print("- Noise level: 10%")
    print("- Resolution: 200 x 50")
    print("- Features: Smooth profile, noise in all regions, 3D visualization")

    params = SimulationParams(
        n_z=200,
        n_x=50,
        pitch=2.0,
        noise_level=0.1,
        include_defects=False,  # Start clean
        phase_offset=0.0
    )

    simulator = EnhancedFCPMSimulator(params)
    intensity = simulator.simulate()

    print(f"\n✓ Simulation complete!")
    print(f"  Shape: {intensity.shape}")
    print(f"  Intensity range: [{np.min(intensity):.4f}, {np.max(intensity):.4f}]")

    # Show that zero-intensity regions now have noise
    intensity_1d = np.mean(intensity, axis=1)
    min_intensity = np.min(intensity_1d)
    print(f"  Minimum intensity: {min_intensity:.4f} (has noise!)")

    # Show standard visualization
    print("\n[1/3] Displaying standard FCPM visualization...")
    simulator.plot_results(show_defects=False)

    # Show 3D director field
    print("\n[2/3] Displaying 3D director field visualization...")
    print("       This shows the helical structure of the cholesteric LC")
    simulator.plot_3d_director_field(
        n_z_planes=15,  # Show 15 z-planes
        n_x_grid=20,    # 20x20 grid in each plane
        n_y_grid=20
    )

    # Now try with defects
    print("\n[3/3] Creating simulation with defects...")
    params_with_defects = SimulationParams(
        n_z=200,
        n_x=50,
        pitch=2.0,
        noise_level=0.1,
        include_defects=True,
        defect_types=['dislocation_b_half', 'tau_disclination'],
        defect_density=0.15
    )

    simulator_defects = EnhancedFCPMSimulator(params_with_defects)
    intensity_defects = simulator_defects.simulate()

    print(f"  ✓ Simulation with defects complete!")
    print(f"  Number of defects: {len(simulator_defects.defect_locations)}")

    # Show results with defects
    print("  Displaying results with defects...")
    simulator_defects.plot_results(show_defects=True)

    # 3D visualization with defects
    print("  Displaying 3D director field with defects...")
    simulator_defects.plot_3d_director_field(n_z_planes=15)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nKey improvements demonstrated:")
    print("  ✓ Noise in zero-intensity regions (no pure black lines)")
    print("  ✓ Smooth intensity transitions (no sharp jumps)")
    print("  ✓ Realistic high-intensity peaks (not overly sharp)")
    print("  ✓ 3D director field showing cholesteric helix")
    print("\nAll visualizations are now displayed.")
    print("="*70)

if __name__ == "__main__":
    main()
