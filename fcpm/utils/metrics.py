"""
Error Metrics for FCPM Reconstruction.

Functions for computing reconstruction quality metrics.
Key insight: Nematic symmetry (n ≡ -n) must be accounted for in error metrics.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.director import DirectorField
from ..core.qtensor import QTensor


def angular_error_nematic(director_recon: DirectorField,
                          director_gt: DirectorField) -> np.ndarray:
    """
    Compute nematic-aware angular error.

    Since n ≡ -n in nematics, we use:
        θ = arccos(|n1 · n2|)

    This gives the minimum angle between the orientations,
    accounting for the head-tail equivalence.

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Angular error in degrees for each voxel.
    """
    n_recon = director_recon.to_array()
    n_gt = director_gt.to_array()

    # Dot product
    dot = np.sum(n_recon * n_gt, axis=-1)

    # Take absolute value for nematic symmetry
    dot = np.abs(dot)

    # Clip for numerical stability
    dot = np.clip(dot, -1, 1)

    # Angle in degrees
    angle = np.degrees(np.arccos(dot))

    return angle


def angular_error_vector(director_recon: DirectorField,
                         director_gt: DirectorField) -> np.ndarray:
    """
    Compute standard (vector) angular error.

    This does NOT account for nematic symmetry.
    Use angular_error_nematic() for proper nematic comparison.

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Angular error in degrees.
    """
    n_recon = director_recon.to_array()
    n_gt = director_gt.to_array()

    dot = np.sum(n_recon * n_gt, axis=-1)
    dot = np.clip(dot, -1, 1)

    return np.degrees(np.arccos(dot))


def euclidean_error(director_recon: DirectorField,
                    director_gt: DirectorField) -> np.ndarray:
    """
    Compute Euclidean distance between directors.

    Note: This is NOT nematic-aware.

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Euclidean distance |n1 - n2| for each voxel.
    """
    n_recon = director_recon.to_array()
    n_gt = director_gt.to_array()

    return np.linalg.norm(n_recon - n_gt, axis=-1)


def euclidean_error_nematic(director_recon: DirectorField,
                            director_gt: DirectorField) -> np.ndarray:
    """
    Compute nematic-aware Euclidean error.

    For each voxel, compute min(|n1 - n2|, |n1 + n2|).

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Minimum Euclidean distance.
    """
    n_recon = director_recon.to_array()
    n_gt = director_gt.to_array()

    dist_pos = np.linalg.norm(n_recon - n_gt, axis=-1)
    dist_neg = np.linalg.norm(n_recon + n_gt, axis=-1)

    return np.minimum(dist_pos, dist_neg)


def intensity_reconstruction_error(I_original: Dict[float, np.ndarray],
                                    I_reconstructed: Dict[float, np.ndarray]) -> Dict[str, float]:
    """
    Compute FCPM intensity reconstruction error.

    This is the DEFINITIVE test for reconstruction quality:
    If the reconstructed director produces the same FCPM signal as
    the ground truth, the reconstruction is physically correct
    (up to allowed symmetries).

    Args:
        I_original: Original FCPM intensities.
        I_reconstructed: Reconstructed intensities from recovered director.

    Returns:
        Dictionary of error metrics.
    """
    errors = []
    relative_errors = []

    for angle in I_original:
        if angle not in I_reconstructed:
            continue

        I_orig = I_original[angle]
        I_recon = I_reconstructed[angle]

        # Absolute error
        diff = I_orig - I_recon
        rmse = np.sqrt(np.mean(diff ** 2))
        errors.append(rmse)

        # Relative error
        max_val = np.max(I_orig)
        if max_val > 1e-10:
            relative_errors.append(rmse / max_val)

    return {
        'rmse_mean': float(np.mean(errors)),
        'rmse_max': float(np.max(errors)),
        'relative_error_mean': float(np.mean(relative_errors)) if relative_errors else 0.0,
        'n_angles': len(errors),
    }


def qtensor_frobenius_error(Q_recon: QTensor, Q_gt: QTensor) -> np.ndarray:
    """
    Compute Frobenius norm of Q-tensor difference.

    ||Q1 - Q2||_F = sqrt(Tr((Q1-Q2)²))

    This is sign-invariant since Q(n) = Q(-n).

    Args:
        Q_recon: Reconstructed Q-tensor.
        Q_gt: Ground truth Q-tensor.

    Returns:
        Frobenius error for each voxel.
    """
    # Compute differences for each component
    dQ_xx = Q_recon.Q_xx - Q_gt.Q_xx
    dQ_yy = Q_recon.Q_yy - Q_gt.Q_yy
    dQ_zz = Q_recon.Q_zz - Q_gt.Q_zz
    dQ_xy = Q_recon.Q_xy - Q_gt.Q_xy
    dQ_xz = Q_recon.Q_xz - Q_gt.Q_xz
    dQ_yz = Q_recon.Q_yz - Q_gt.Q_yz

    # Frobenius norm: ||dQ||² = sum of squared components
    # For symmetric matrix: diag contributes once, off-diag twice
    diff_sq = (dQ_xx**2 + dQ_yy**2 + dQ_zz**2 +
               2 * (dQ_xy**2 + dQ_xz**2 + dQ_yz**2))

    return np.sqrt(diff_sq)


def sign_accuracy(director_recon: DirectorField,
                   director_gt: DirectorField) -> float:
    """
    Compute the fraction of voxels where the sign is correct.

    A voxel has the correct sign when ``dot(n_recon, n_gt) > 0``.

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Fraction in [0, 1].  1.0 means every voxel has matching sign.
    """
    n_recon = director_recon.to_array()
    n_gt = director_gt.to_array()
    dot = np.sum(n_recon * n_gt, axis=-1)
    return float(np.mean(dot > 0))


def spatial_error_distribution(
    director_recon: DirectorField,
    director_gt: DirectorField,
) -> Dict[str, np.ndarray | float]:
    """
    Compute per-z-layer angular error statistics and full error map.

    Args:
        director_recon: Reconstructed director field.
        director_gt: Ground truth director field.

    Returns:
        Dictionary with keys:

        * ``error_map`` — full ``(ny, nx, nz)`` angular error array (degrees)
        * ``layer_mean`` — ``(nz,)`` mean angular error per z-layer
        * ``layer_median`` — ``(nz,)`` median angular error per z-layer
        * ``layer_max`` — ``(nz,)`` max angular error per z-layer
    """
    ang_err = angular_error_nematic(director_recon, director_gt)
    nz = ang_err.shape[2]

    layer_mean = np.array([np.mean(ang_err[:, :, z]) for z in range(nz)])
    layer_median = np.array([np.median(ang_err[:, :, z]) for z in range(nz)])
    layer_max = np.array([np.max(ang_err[:, :, z]) for z in range(nz)])

    return {
        'error_map': ang_err,
        'layer_mean': layer_mean,
        'layer_median': layer_median,
        'layer_max': layer_max,
    }


def summary_metrics(director_recon: DirectorField,
                    director_gt: DirectorField,
                    I_original: Optional[Dict[float, np.ndarray]] = None,
                    I_reconstructed: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, float]:
    """
    Compute comprehensive summary metrics.

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        I_original: Original FCPM intensities.
        I_reconstructed: Reconstructed intensities.

    Returns:
        Dictionary of all metrics.
    """
    # Angular error (nematic-aware)
    ang_err = angular_error_nematic(director_recon, director_gt)

    metrics = {
        'angular_error_mean_deg': float(np.mean(ang_err)),
        'angular_error_median_deg': float(np.median(ang_err)),
        'angular_error_std_deg': float(np.std(ang_err)),
        'angular_error_max_deg': float(np.max(ang_err)),
        'angular_error_90th_deg': float(np.percentile(ang_err, 90)),
        'angular_error_95th_deg': float(np.percentile(ang_err, 95)),
        'angular_error_99th_deg': float(np.percentile(ang_err, 99)),
    }

    # Euclidean error
    euc_err = euclidean_error_nematic(director_recon, director_gt)
    metrics['euclidean_error_mean'] = float(np.mean(euc_err))
    metrics['euclidean_error_max'] = float(np.max(euc_err))

    # Sign accuracy
    metrics['sign_accuracy'] = sign_accuracy(director_recon, director_gt)

    # Intensity reconstruction (if available)
    if I_original is not None and I_reconstructed is not None:
        intensity_metrics = intensity_reconstruction_error(I_original, I_reconstructed)
        metrics.update({f'intensity_{k}': v for k, v in intensity_metrics.items()})

    return metrics


def perfect_reconstruction_test(director_recon: DirectorField,
                                 director_gt: DirectorField,
                                 tolerance: float = 0.1) -> Tuple[bool, Dict]:
    """
    Test if reconstruction is perfect (within tolerance).

    A perfect reconstruction has either:
    - Very small angular error, OR
    - Perfect FCPM intensity reconstruction

    Due to nematic symmetry, angular error can be large (up to 180°)
    even for perfect reconstructions if signs are flipped.

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        tolerance: Angular tolerance in degrees.

    Returns:
        Tuple of (is_perfect: bool, details: dict).
    """
    # Compute nematic angular error
    ang_err = angular_error_nematic(director_recon, director_gt)

    max_err = np.max(ang_err)
    mean_err = np.mean(ang_err)

    is_perfect = max_err < tolerance

    details = {
        'max_angular_error': float(max_err),
        'mean_angular_error': float(mean_err),
        'tolerance': tolerance,
        'is_perfect': is_perfect,
    }

    return is_perfect, details
