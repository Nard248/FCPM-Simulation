"""
V2 Sign Optimization - Comprehensive Test and Validation.

Compares V2 layer-by-layer approach against V1 (BFS + iterative flip).

Tests:
1. Cholesteric structure (helical - natural z-layering)
2. Different noise levels
3. Different volume sizes
4. Metrics comparison: angular error, energy, computation time
"""

import sys
from pathlib import Path
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fcpm
from fcpm.reconstruction import reconstruct_via_qtensor, combined_optimization, gradient_energy
from sign_optimization_v2 import (
    combined_v2_optimization,
    layer_by_layer_optimization,
    layer_by_layer_vectorized,
    layer_then_refine,
    compute_gradient_energy
)


def run_comparison(director_gt, noise_level=0.03, verbose=True):
    """
    Run full comparison between V1 and V2 optimization.

    Returns dict with all metrics.
    """
    results = {
        'noise_level': noise_level,
        'shape': director_gt.shape,
    }

    # Simulate FCPM
    I_fcpm = fcpm.simulate_fcpm(director_gt)
    I_fcpm_noisy = fcpm.add_fcpm_realistic_noise(
        I_fcpm, noise_model='mixed', gaussian_sigma=noise_level
    )
    I_fcpm_noisy = fcpm.normalize_fcpm(I_fcpm_noisy)

    # Reconstruct Q-tensor (no sign fix)
    director_raw, Q, info = reconstruct_via_qtensor(I_fcpm_noisy)

    # Raw metrics (before sign fix)
    raw_energy = compute_gradient_energy(director_raw.to_array())
    raw_metrics = fcpm.summary_metrics(director_raw, director_gt)
    results['raw'] = {
        'energy': raw_energy,
        'angular_error_mean': raw_metrics['angular_error_mean_deg'],
        'angular_error_median': raw_metrics['angular_error_median_deg'],
    }

    if verbose:
        print(f"\nRaw (no sign fix):")
        print(f"  Energy: {raw_energy:.2f}")
        print(f"  Angular error: {raw_metrics['angular_error_mean_deg']:.2f} deg (mean)")

    # V1 optimization
    t0 = time.time()
    director_v1, info_v1 = combined_optimization(director_raw, verbose=False)
    time_v1 = time.time() - t0

    v1_metrics = fcpm.summary_metrics(director_v1, director_gt)
    results['v1'] = {
        'energy': info_v1['final_energy'],
        'angular_error_mean': v1_metrics['angular_error_mean_deg'],
        'angular_error_median': v1_metrics['angular_error_median_deg'],
        'time': time_v1,
    }

    if verbose:
        print(f"\nV1 (BFS + iterative flip):")
        print(f"  Energy: {info_v1['final_energy']:.2f}")
        print(f"  Angular error: {v1_metrics['angular_error_mean_deg']:.2f} deg (mean)")
        print(f"  Time: {time_v1:.2f}s")

    # V2 layer_then_refine (RECOMMENDED)
    t0 = time.time()
    result_v2 = layer_then_refine(director_raw, verbose=False)
    time_v2 = time.time() - t0

    v2_metrics = fcpm.summary_metrics(result_v2.director, director_gt)
    results['v2'] = {
        'energy': result_v2.final_energy,
        'angular_error_mean': v2_metrics['angular_error_mean_deg'],
        'angular_error_median': v2_metrics['angular_error_median_deg'],
        'time': time_v2,
    }

    if verbose:
        print(f"\nV2 (layer_then_refine - RECOMMENDED):")
        print(f"  Energy: {result_v2.final_energy:.2f}")
        print(f"  Angular error: {v2_metrics['angular_error_mean_deg']:.2f} deg (mean)")
        print(f"  Time: {time_v2:.2f}s")

    # V2 layer-only (no refinement, for comparison)
    t0 = time.time()
    result_v2_vec = layer_by_layer_vectorized(director_raw, verbose=False)
    time_v2_vec = time.time() - t0

    v2_vec_metrics = fcpm.summary_metrics(result_v2_vec.director, director_gt)
    results['v2_layer_only'] = {
        'energy': result_v2_vec.final_energy,
        'angular_error_mean': v2_vec_metrics['angular_error_mean_deg'],
        'angular_error_median': v2_vec_metrics['angular_error_median_deg'],
        'time': time_v2_vec,
    }

    if verbose:
        print(f"\nV2 Layer-Only (no refinement):")
        print(f"  Energy: {result_v2_vec.final_energy:.2f}")
        print(f"  Angular error: {v2_vec_metrics['angular_error_mean_deg']:.2f} deg (mean)")
        print(f"  Time: {time_v2_vec:.2f}s")

    return results, director_v1, result_v2.director, result_v2_vec.director


def test_noise_sensitivity():
    """Test how V1 and V2 compare across different noise levels."""
    print("=" * 70)
    print("NOISE SENSITIVITY TEST")
    print("=" * 70)

    # Create test director
    director_gt = fcpm.create_cholesteric_director(shape=(48, 48, 24), pitch=6.0)
    print(f"Test director: {director_gt.shape}")

    noise_levels = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    all_results = []

    for noise in noise_levels:
        print(f"\n{'='*40}")
        print(f"Noise level: {noise*100:.0f}%")
        print('='*40)

        results, _, _, _ = run_comparison(director_gt, noise_level=noise, verbose=True)
        all_results.append(results)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    noise_pct = [r['noise_level'] * 100 for r in all_results]

    # Angular error
    axes[0].plot(noise_pct, [r['raw']['angular_error_mean'] for r in all_results],
                 'k--', label='Raw (no fix)', linewidth=2)
    axes[0].plot(noise_pct, [r['v1']['angular_error_mean'] for r in all_results],
                 'b-o', label='V1 (BFS+flip)', linewidth=2, markersize=8)
    axes[0].plot(noise_pct, [r['v2']['angular_error_mean'] for r in all_results],
                 'r-s', label='V2 (layer+refine)', linewidth=2, markersize=8)
    axes[0].set_xlabel('Noise Level (%)', fontsize=12)
    axes[0].set_ylabel('Mean Angular Error (deg)', fontsize=12)
    axes[0].set_title('Angular Error vs Noise', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Energy
    axes[1].plot(noise_pct, [r['raw']['energy'] for r in all_results],
                 'k--', label='Raw', linewidth=2)
    axes[1].plot(noise_pct, [r['v1']['energy'] for r in all_results],
                 'b-o', label='V1', linewidth=2, markersize=8)
    axes[1].plot(noise_pct, [r['v2']['energy'] for r in all_results],
                 'r-s', label='V2', linewidth=2, markersize=8)
    axes[1].plot(noise_pct, [r['v2_layer_only']['energy'] for r in all_results],
                 'g--^', label='V2 layer-only', linewidth=2, markersize=6, alpha=0.7)
    axes[1].set_xlabel('Noise Level (%)', fontsize=12)
    axes[1].set_ylabel('Gradient Energy', fontsize=12)
    axes[1].set_title('Energy vs Noise', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Time
    axes[2].plot(noise_pct, [r['v1']['time'] for r in all_results],
                 'b-o', label='V1', linewidth=2, markersize=8)
    axes[2].plot(noise_pct, [r['v2']['time'] for r in all_results],
                 'r-s', label='V2', linewidth=2, markersize=8)
    axes[2].plot(noise_pct, [r['v2_layer_only']['time'] for r in all_results],
                 'g--^', label='V2 layer-only', linewidth=2, markersize=6, alpha=0.7)
    axes[2].set_xlabel('Noise Level (%)', fontsize=12)
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Computation Time', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'noise_sensitivity_comparison.png', dpi=150)
    print(f"\nSaved plot: noise_sensitivity_comparison.png")

    return all_results


def test_volume_sizes():
    """Test performance across different volume sizes."""
    print("\n" + "=" * 70)
    print("VOLUME SIZE TEST")
    print("=" * 70)

    sizes = [(32, 32, 16), (48, 48, 24), (64, 64, 32)]
    noise_level = 0.05

    all_results = []

    for shape in sizes:
        print(f"\n{'='*40}")
        print(f"Shape: {shape}")
        print('='*40)

        director_gt = fcpm.create_cholesteric_director(shape=shape, pitch=6.0)
        results, _, _, _ = run_comparison(director_gt, noise_level=noise_level, verbose=True)
        results['volume'] = np.prod(shape)
        all_results.append(results)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Shape':<20} {'V1 Error':>12} {'V2 Error':>12} {'V1 Time':>10} {'V2 Time':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{str(r['shape']):<20} "
              f"{r['v1']['angular_error_mean']:>10.2f}° "
              f"{r['v2']['angular_error_mean']:>10.2f}° "
              f"{r['v1']['time']:>9.2f}s "
              f"{r['v2']['time']:>9.2f}s")

    return all_results


def visualize_comparison(director_gt, director_v1, director_v2, noise_level):
    """Create visual comparison of reconstructions."""
    z_mid = director_gt.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Director slices
    fcpm.plot_director_slice(director_gt, z_idx=z_mid, step=2, ax=axes[0, 0],
                             title='Ground Truth')
    fcpm.plot_director_slice(director_v1, z_idx=z_mid, step=2, ax=axes[0, 1],
                             title='V1 Reconstruction')
    fcpm.plot_director_slice(director_v2, z_idx=z_mid, step=2, ax=axes[0, 2],
                             title='V2 Reconstruction')

    # Angular error maps
    from fcpm.utils.metrics import angular_error_nematic
    err_v1 = angular_error_nematic(director_v1, director_gt) * 180 / np.pi
    err_v2 = angular_error_nematic(director_v2, director_gt) * 180 / np.pi

    im1 = axes[0, 3].imshow(err_v1[:, :, z_mid], cmap='hot', vmin=0, vmax=45)
    axes[0, 3].set_title('V1 Error (deg)')
    axes[0, 3].axis('off')
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)

    im2 = axes[1, 3].imshow(err_v2[:, :, z_mid], cmap='hot', vmin=0, vmax=45)
    axes[1, 3].set_title('V2 Error (deg)')
    axes[1, 3].axis('off')
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)

    # nz components (often problematic)
    im_gt = axes[1, 0].imshow(director_gt.nz[:, :, z_mid], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('GT nz')
    axes[1, 0].axis('off')

    im_v1 = axes[1, 1].imshow(director_v1.nz[:, :, z_mid], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('V1 nz')
    axes[1, 1].axis('off')

    im_v2 = axes[1, 2].imshow(director_v2.nz[:, :, z_mid], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('V2 nz')
    axes[1, 2].axis('off')

    plt.suptitle(f'V1 vs V2 Comparison (noise={noise_level*100:.0f}%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'v1_v2_visual_comparison.png', dpi=150)
    print(f"Saved: v1_v2_visual_comparison.png")


def main():
    """Run all tests."""
    print("=" * 70)
    print("V2 SIGN OPTIMIZATION - COMPREHENSIVE VALIDATION")
    print("=" * 70)

    # Quick single comparison with visualization
    print("\n" + "=" * 70)
    print("SINGLE COMPARISON (with visualization)")
    print("=" * 70)

    director_gt = fcpm.create_cholesteric_director(shape=(48, 48, 24), pitch=6.0)
    noise_level = 0.05

    results, director_v1, director_v2, _ = run_comparison(
        director_gt, noise_level=noise_level, verbose=True
    )

    visualize_comparison(director_gt, director_v1, director_v2, noise_level)

    # Noise sensitivity
    noise_results = test_noise_sensitivity()

    # Volume size test
    size_results = test_volume_sizes()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    v1_better = sum(1 for r in noise_results
                    if r['v1']['angular_error_mean'] < r['v2']['angular_error_mean'])
    v2_better = len(noise_results) - v1_better

    print(f"\nNoise sensitivity test ({len(noise_results)} conditions):")
    print(f"  V1 better: {v1_better}")
    print(f"  V2 better: {v2_better}")

    avg_v1_time = np.mean([r['v1']['time'] for r in noise_results])
    avg_v2_time = np.mean([r['v2']['time'] for r in noise_results])
    avg_v2_layer_time = np.mean([r['v2_layer_only']['time'] for r in noise_results])

    print(f"\nAverage computation time:")
    print(f"  V1: {avg_v1_time:.2f}s")
    print(f"  V2: {avg_v2_time:.2f}s")
    print(f"  V2 layer-only: {avg_v2_layer_time:.2f}s")

    print("\nAll results saved to v2/ folder.")


if __name__ == "__main__":
    main()
