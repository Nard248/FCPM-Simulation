"""
Analysis Plots for FCPM Reconstruction.

Functions for analyzing reconstruction quality, convergence,
and error distributions.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Any
from ..core.director import DirectorField
from ..core.qtensor import QTensor
from ..utils.metrics import angular_error_nematic


def plot_error_histogram(director_recon: DirectorField,
                         director_gt: DirectorField,
                         bins: int = 50,
                         figsize: Tuple[float, float] = (10, 4)) -> plt.Figure:
    """
    Plot histogram of angular errors.

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        bins: Number of histogram bins.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    errors = angular_error_nematic(director_recon, director_gt)
    errors_flat = errors.flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Full histogram
    axes[0].hist(errors_flat, bins=bins, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(errors_flat), color='red', linestyle='--',
                    label=f'Mean: {np.mean(errors_flat):.2f}°')
    axes[0].axvline(np.median(errors_flat), color='green', linestyle='--',
                    label=f'Median: {np.median(errors_flat):.2f}°')
    axes[0].set_xlabel('Angular Error (degrees)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Angular Error Distribution')
    axes[0].legend()

    # Cumulative distribution
    sorted_errors = np.sort(errors_flat)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumulative, 'b-', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1].axhline(0.9, color='gray', linestyle=':', alpha=0.7)
    axes[1].axhline(0.95, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Angular Error (degrees)')
    axes[1].set_ylabel('Cumulative Fraction')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_by_depth(director_recon: DirectorField,
                        director_gt: DirectorField,
                        figsize: Tuple[float, float] = (8, 5)) -> plt.Figure:
    """
    Plot error statistics as a function of depth (z).

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    errors = angular_error_nematic(director_recon, director_gt)

    z_dim = errors.shape[2]
    mean_errors = [np.mean(errors[:, :, z]) for z in range(z_dim)]
    median_errors = [np.median(errors[:, :, z]) for z in range(z_dim)]
    max_errors = [np.max(errors[:, :, z]) for z in range(z_dim)]
    percentile_90 = [np.percentile(errors[:, :, z], 90) for z in range(z_dim)]

    fig, ax = plt.subplots(figsize=figsize)

    z_vals = np.arange(z_dim)
    ax.plot(z_vals, mean_errors, 'b-', label='Mean', linewidth=2)
    ax.plot(z_vals, median_errors, 'g--', label='Median', linewidth=2)
    ax.plot(z_vals, percentile_90, 'r:', label='90th percentile', linewidth=2)
    ax.fill_between(z_vals, mean_errors, alpha=0.2)

    ax.set_xlabel('Z (depth)')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title('Reconstruction Error vs. Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_intensity_reconstruction(I_original: Dict[float, np.ndarray],
                                   I_reconstructed: Dict[float, np.ndarray],
                                   z_idx: int,
                                   figsize: Tuple[float, float] = (14, 5)) -> plt.Figure:
    """
    Compare original and reconstructed FCPM intensities.

    Args:
        I_original: Original FCPM intensities.
        I_reconstructed: Reconstructed FCPM intensities.
        z_idx: Z-slice to display.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    angles = sorted(I_original.keys())

    fig, axes = plt.subplots(2, len(angles), figsize=figsize)

    for i, angle in enumerate(angles):
        I_orig = I_original[angle][:, :, z_idx]
        I_recon = I_reconstructed[angle][:, :, z_idx]

        vmin = min(I_orig.min(), I_recon.min())
        vmax = max(I_orig.max(), I_recon.max())

        im1 = axes[0, i].imshow(I_orig, cmap='gray', origin='lower',
                                vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Original α={np.degrees(angle):.0f}°')
        axes[0, i].axis('off')

        im2 = axes[1, i].imshow(I_recon, cmap='gray', origin='lower',
                                vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Reconstructed')
        axes[1, i].axis('off')

    plt.suptitle(f'FCPM Intensity Comparison at z={z_idx}')
    plt.tight_layout()

    return fig


def plot_convergence(history: List[Dict], figsize: Tuple[float, float] = (10, 4)) -> plt.Figure:
    """
    Plot convergence history from iterative optimization.

    Args:
        history: List of dicts with 'iteration', 'energy', 'flipped' keys.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    iterations = [h['iteration'] for h in history]
    energies = [h['energy'] for h in history]
    flipped = [h.get('flipped', 0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Energy plot
    axes[0].plot(iterations, energies, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gradient Energy')
    axes[0].set_title('Energy Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Flips plot
    axes[1].bar(iterations, flipped, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Voxels Flipped')
    axes[1].set_title('Sign Flips per Iteration')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_qtensor_components(Q: QTensor,
                            z_idx: int,
                            figsize: Tuple[float, float] = (15, 8)) -> plt.Figure:
    """
    Plot all Q-tensor components at a specific z-slice.

    Args:
        Q: Q-tensor field.
        z_idx: Z-slice index.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    components = [
        ('Q_xx', Q.Q_xx[:, :, z_idx]),
        ('Q_yy', Q.Q_yy[:, :, z_idx]),
        ('Q_xy', Q.Q_xy[:, :, z_idx]),
        ('Q_xz', Q.Q_xz[:, :, z_idx]),
        ('Q_yz', Q.Q_yz[:, :, z_idx]),
        ('Q_zz', Q.Q_zz[:, :, z_idx]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (name, data) in zip(axes, components):
        vmax = np.max(np.abs(data))
        im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                       vmin=-vmax, vmax=vmax)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Q-tensor components at z={z_idx}')
    plt.tight_layout()

    return fig


def plot_order_parameter(Q: QTensor,
                         z_idx: Optional[int] = None,
                         figsize: Tuple[float, float] = (8, 6)) -> plt.Figure:
    """
    Plot the scalar order parameter S distribution.

    Args:
        Q: Q-tensor field.
        z_idx: Z-slice index (None for all slices).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    S = Q.scalar_order_parameter()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if z_idx is not None:
        im = axes[0].imshow(S[:, :, z_idx], cmap='viridis', origin='lower')
        axes[0].set_title(f'Order Parameter S at z={z_idx}')
    else:
        # Show middle slice
        mid_z = S.shape[2] // 2
        im = axes[0].imshow(S[:, :, mid_z], cmap='viridis', origin='lower')
        axes[0].set_title(f'Order Parameter S at z={mid_z}')
    plt.colorbar(im, ax=axes[0])

    # Histogram
    axes[1].hist(S.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
    axes[1].axvline(np.mean(S), color='red', linestyle='--',
                    label=f'Mean: {np.mean(S):.3f}')
    axes[1].set_xlabel('S')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Order Parameter Distribution')
    axes[1].legend()

    plt.tight_layout()
    return fig


def summary_statistics(director_recon: DirectorField,
                       director_gt: DirectorField,
                       I_original: Optional[Dict[float, np.ndarray]] = None,
                       I_reconstructed: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, float]:
    """
    Compute summary statistics for reconstruction quality.

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        I_original: Original FCPM intensities.
        I_reconstructed: Reconstructed intensities.

    Returns:
        Dictionary of statistics.
    """
    errors = angular_error_nematic(director_recon, director_gt)

    stats = {
        'angular_error_mean': float(np.mean(errors)),
        'angular_error_median': float(np.median(errors)),
        'angular_error_std': float(np.std(errors)),
        'angular_error_max': float(np.max(errors)),
        'angular_error_90th': float(np.percentile(errors, 90)),
        'angular_error_95th': float(np.percentile(errors, 95)),
    }

    if I_original is not None and I_reconstructed is not None:
        intensity_errors = []
        for angle in I_original:
            if angle in I_reconstructed:
                diff = I_original[angle] - I_reconstructed[angle]
                rmse = np.sqrt(np.mean(diff**2))
                intensity_errors.append(rmse)

        stats['intensity_rmse_mean'] = float(np.mean(intensity_errors))
        stats['intensity_rmse_max'] = float(np.max(intensity_errors))

    return stats
