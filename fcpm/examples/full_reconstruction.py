#!/usr/bin/env python3
"""
Full FCPM Reconstruction Pipeline Example

This example demonstrates the complete workflow:
1. Create synthetic director field
2. Simulate FCPM measurements
3. Add realistic noise
4. Reconstruct via Q-tensor method
5. Fix sign ambiguity
6. Evaluate reconstruction quality
7. Visualize and compare results

Run with: python -m fcpm.examples.full_reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fcpm


def main():
    print("=" * 60)
    print("FCPM Full Reconstruction Pipeline")
    print("=" * 60)

    # =================================================================
    # 1. Create Ground Truth Director Field
    # =================================================================
    print("\n1. Creating ground truth director field...")

    shape = (32, 32, 16)  # Smaller for faster demonstration
    director_gt = fcpm.create_cholesteric_director(shape, pitch=8.0, axis='z')
    print(f"   Shape: {director_gt.shape}")

    # =================================================================
    # 2. Simulate FCPM Measurements
    # =================================================================
    print("\n2. Simulating FCPM measurements...")

    I_fcpm_clean = fcpm.simulate_fcpm(director_gt)
    print(f"   Generated clean FCPM for {len(I_fcpm_clean)} angles")

    # =================================================================
    # 3. Add Realistic Noise
    # =================================================================
    print("\n3. Adding realistic noise...")

    noise_level = 0.05
    I_fcpm_noisy = fcpm.add_fcpm_realistic_noise(
        I_fcpm_clean,
        noise_model='mixed',
        gaussian_sigma=noise_level,
        photon_count=5000,
        seed=42
    )

    snr = fcpm.signal_to_noise_ratio(
        list(I_fcpm_clean.values())[0],
        list(I_fcpm_clean.values())[0] - list(I_fcpm_noisy.values())[0]
    )
    print(f"   Noise level: {noise_level*100:.1f}%")
    print(f"   SNR: {snr:.1f} dB")

    # =================================================================
    # 4. Reconstruct via Q-Tensor Method
    # =================================================================
    print("\n4. Reconstructing via Q-tensor method...")

    director_recon, Q_recon, recon_info = fcpm.reconstruct_via_qtensor(I_fcpm_noisy)
    print(f"   Q-tensor reconstruction complete")
    print(f"   Mean nx²: {recon_info['nx_squared_mean']:.4f}")
    print(f"   Mean ny²: {recon_info['ny_squared_mean']:.4f}")
    print(f"   Mean nz²: {recon_info['nz_squared_mean']:.4f}")

    # =================================================================
    # 5. Fix Sign Ambiguity
    # =================================================================
    print("\n5. Fixing sign ambiguity...")

    director_fixed, opt_info = fcpm.combined_optimization(director_recon, verbose=True)
    print(f"   Converged: {opt_info['converged']}")
    print(f"   Final gradient energy: {opt_info['final_energy']:.2f}")

    # =================================================================
    # 6. Evaluate Reconstruction Quality
    # =================================================================
    print("\n6. Evaluating reconstruction quality...")

    # Angular error (nematic-aware)
    ang_error = fcpm.angular_error_nematic(director_fixed, director_gt)
    print(f"\n   Angular Error (nematic-aware):")
    print(f"     Mean:   {np.mean(ang_error):.2f}°")
    print(f"     Median: {np.median(ang_error):.2f}°")
    print(f"     Max:    {np.max(ang_error):.2f}°")
    print(f"     90th %: {np.percentile(ang_error, 90):.2f}°")

    # Intensity reconstruction (the definitive test)
    I_fcpm_recon = fcpm.simulate_fcpm(director_fixed)
    intensity_error = fcpm.intensity_reconstruction_error(I_fcpm_clean, I_fcpm_recon)
    print(f"\n   Intensity Reconstruction Error:")
    print(f"     RMSE:     {intensity_error['rmse_mean']:.2e}")
    print(f"     Relative: {intensity_error['relative_error_mean']*100:.4f}%")

    # Full summary
    metrics = fcpm.summary_metrics(director_fixed, director_gt, I_fcpm_clean, I_fcpm_recon)

    # =================================================================
    # 7. Visualize Results
    # =================================================================
    print("\n7. Creating visualizations...")

    fig = plt.figure(figsize=(16, 12))

    z_mid = shape[2] // 2

    # Row 1: Director fields
    ax1 = fig.add_subplot(3, 4, 1)
    fcpm.plot_director_slice(director_gt, z_mid, step=2, ax=ax1,
                             title='Ground Truth', show_colorbar=False)

    ax2 = fig.add_subplot(3, 4, 2)
    fcpm.plot_director_slice(director_fixed, z_mid, step=2, ax=ax2,
                             title='Reconstructed', show_colorbar=False)

    # Angular error map
    ax3 = fig.add_subplot(3, 4, 3)
    im = ax3.imshow(ang_error[:, :, z_mid], cmap='hot', origin='lower',
                    vmin=0, vmax=max(5, np.percentile(ang_error, 95)))
    ax3.set_title('Angular Error (degrees)')
    plt.colorbar(im, ax=ax3)

    # Histogram of angular errors
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.hist(ang_error.flatten(), bins=50, density=True, alpha=0.7)
    ax4.axvline(np.mean(ang_error), color='r', linestyle='--',
                label=f'Mean: {np.mean(ang_error):.2f}°')
    ax4.axvline(np.median(ang_error), color='g', linestyle='--',
                label=f'Median: {np.median(ang_error):.2f}°')
    ax4.set_xlabel('Angular Error (degrees)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.set_title('Error Distribution')

    # Row 2: FCPM intensities (noisy vs reconstructed)
    angles = sorted(I_fcpm_clean.keys())
    for i, angle in enumerate(angles[:4]):
        ax = fig.add_subplot(3, 4, 5 + i)
        diff = np.abs(I_fcpm_clean[angle][:, :, z_mid] - I_fcpm_recon[angle][:, :, z_mid])
        im = ax.imshow(diff, cmap='hot', origin='lower')
        ax.set_title(f'|I_gt - I_recon| at {np.degrees(angle):.0f}°')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Q-tensor components and convergence
    ax9 = fig.add_subplot(3, 4, 9)
    Q_gt = fcpm.director_to_qtensor(director_gt)
    im = ax9.imshow(Q_gt.Q_xy[:, :, z_mid], cmap='RdBu_r', origin='lower')
    ax9.set_title('Q_xy (Ground Truth)')
    plt.colorbar(im, ax=ax9)

    ax10 = fig.add_subplot(3, 4, 10)
    im = ax10.imshow(Q_recon.Q_xy[:, :, z_mid], cmap='RdBu_r', origin='lower')
    ax10.set_title('Q_xy (Reconstructed)')
    plt.colorbar(im, ax=ax10)

    ax11 = fig.add_subplot(3, 4, 11)
    Q_error = fcpm.qtensor_frobenius_error(Q_recon, Q_gt)
    im = ax11.imshow(Q_error[:, :, z_mid], cmap='hot', origin='lower')
    ax11.set_title('Q-tensor Frobenius Error')
    plt.colorbar(im, ax=ax11)

    # Convergence plot
    if 'history' in opt_info:
        ax12 = fig.add_subplot(3, 4, 12)
        energies = [h['energy'] for h in opt_info['history']]
        ax12.plot(energies, 'b-', linewidth=2)
        ax12.set_xlabel('Iteration')
        ax12.set_ylabel('Gradient Energy')
        ax12.set_title('Sign Optimization Convergence')
        ax12.set_yscale('log')
        ax12.grid(True, alpha=0.3)

    plt.suptitle(f'FCPM Reconstruction Results (noise={noise_level*100:.0f}%)', fontsize=14)
    plt.tight_layout()
    plt.savefig('fcpm_reconstruction_example.png', dpi=150, bbox_inches='tight')
    print("   Saved figure to: fcpm_reconstruction_example.png")
    plt.show()

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 60)
    print(f"\nNoise level: {noise_level*100:.1f}%")
    print(f"\nAngular Error:")
    print(f"  Mean:   {metrics['angular_error_mean_deg']:.2f}°")
    print(f"  Median: {metrics['angular_error_median_deg']:.2f}°")
    print(f"  95th %: {metrics['angular_error_95th_deg']:.2f}°")
    print(f"\nIntensity RMSE: {metrics.get('intensity_rmse_mean', 0):.2e}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
