"""
Systematic distortions for FCPM experimental simulation.

Models common experimental artifacts that degrade reconstruction quality.
Each distortion function takes clean FCPM data and returns corrupted data,
enabling systematic robustness testing.

Distortion types:
    - Polarization offset: misaligned polarizer
    - Intensity drift: photobleaching or lamp instability
    - Slice misregistration: sample drift between angle acquisitions
    - PSF blur: optical point spread function
    - Background gradient: non-uniform illumination
    - Saturation: detector clipping at high intensity
    - Bleaching: progressive signal loss during acquisition
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


def apply_polarization_offset(
    I_fcpm: Dict[float, np.ndarray],
    offset_rad: float,
) -> Dict[float, np.ndarray]:
    """Simulate a misaligned polarizer by shifting all angles.

    In practice, the actual measurement angles differ from the nominal
    values by a constant offset (misalignment of the polarizer or the
    sample orientation).

    Args:
        I_fcpm: Clean FCPM intensities keyed by nominal angle (radians).
        offset_rad: Polarizer offset in radians.

    Returns:
        FCPM intensities that would be measured at the shifted angles,
        keyed by the original (nominal) angles.
    """
    # We need the director to re-simulate, but we only have intensities.
    # Instead, we shift which angle produces which measurement.
    # This is equivalent to rotating the sample by -offset_rad.
    # For a linear approximation, interpolate between neighboring angles.
    angles = sorted(I_fcpm.keys())
    I_shifted = {}

    for nominal_angle in angles:
        true_angle = nominal_angle + offset_rad

        # Find bracketing angles for interpolation
        below = max([a for a in angles if a <= true_angle], default=angles[0])
        above = min([a for a in angles if a >= true_angle], default=angles[-1])

        if below == above:
            I_shifted[nominal_angle] = I_fcpm[below].copy()
        else:
            t = (true_angle - below) / (above - below)
            I_shifted[nominal_angle] = (1 - t) * I_fcpm[below] + t * I_fcpm[above]

    return I_shifted


def apply_intensity_drift(
    I_fcpm: Dict[float, np.ndarray],
    drift_rate: float,
    axis: str = 'z',
) -> Dict[float, np.ndarray]:
    """Simulate photobleaching or lamp drift along an axis.

    The signal decays exponentially along the specified axis:
        I_corrupted(z) = I_clean(z) * exp(-drift_rate * z / nz)

    Args:
        I_fcpm: Clean FCPM intensities.
        drift_rate: Total fractional decay. E.g., 0.2 means 20% signal
            loss from first to last slice.
        axis: Axis along which drift occurs ('z', 'y', or 'x').

    Returns:
        FCPM intensities with drift applied.
    """
    I_drifted = {}
    axis_idx = {'y': 0, 'x': 1, 'z': 2}[axis]

    for angle, I in I_fcpm.items():
        n_slices = I.shape[axis_idx]
        decay = np.exp(-drift_rate * np.arange(n_slices) / n_slices)

        # Reshape decay to broadcast along the right axis
        shape = [1, 1, 1]
        shape[axis_idx] = n_slices
        decay = decay.reshape(shape)

        I_drifted[angle] = I * decay

    return I_drifted


def apply_slice_misregistration(
    I_fcpm: Dict[float, np.ndarray],
    shift_yx: Tuple[float, float] = (0.0, 0.0),
    per_angle: bool = False,
    seed: Optional[int] = None,
) -> Dict[float, np.ndarray]:
    """Simulate sample drift between angle acquisitions.

    In real microscopy, the sample may drift between consecutive
    polarization measurements. This manifests as a spatial shift
    between the images at different angles.

    Args:
        I_fcpm: Clean FCPM intensities.
        shift_yx: Maximum shift in (y, x) voxels. If per_angle=False,
            applied uniformly. If per_angle=True, each angle gets a
            random shift up to this magnitude.
        per_angle: If True, apply independent random shifts per angle.
        seed: Random seed (only used when per_angle=True).

    Returns:
        FCPM intensities with spatial shifts applied.
    """
    rng = np.random.default_rng(seed)
    I_shifted = {}

    for angle, I in I_fcpm.items():
        if per_angle:
            sy = rng.uniform(-shift_yx[0], shift_yx[0])
            sx = rng.uniform(-shift_yx[1], shift_yx[1])
        else:
            sy, sx = shift_yx

        if abs(sy) < 1e-6 and abs(sx) < 1e-6:
            I_shifted[angle] = I.copy()
        else:
            I_shifted[angle] = ndimage.shift(I, [sy, sx, 0], order=1, mode='nearest')

    return I_shifted


def apply_psf_blur(
    I_fcpm: Dict[float, np.ndarray],
    sigma_xy: float = 1.0,
    sigma_z: float = 2.0,
) -> Dict[float, np.ndarray]:
    """Simulate optical point spread function blurring.

    In confocal microscopy, the PSF is typically anisotropic:
    wider in z (axial) than in xy (lateral).

    Args:
        I_fcpm: Clean FCPM intensities.
        sigma_xy: Lateral PSF width in voxels.
        sigma_z: Axial PSF width in voxels.

    Returns:
        FCPM intensities with PSF blur applied.
    """
    I_blurred = {}
    sigma = (sigma_xy, sigma_xy, sigma_z)  # (y, x, z)

    for angle, I in I_fcpm.items():
        I_blurred[angle] = ndimage.gaussian_filter(I, sigma=sigma)

    return I_blurred


def apply_background_gradient(
    I_fcpm: Dict[float, np.ndarray],
    direction: str = 'x',
    magnitude: float = 0.1,
) -> Dict[float, np.ndarray]:
    """Simulate non-uniform illumination (background gradient).

    Adds a linear background gradient across the field of view.

    Args:
        I_fcpm: Clean FCPM intensities.
        direction: Gradient direction ('x', 'y', or 'z').
        magnitude: Maximum background added (relative to max intensity).

    Returns:
        FCPM intensities with gradient background.
    """
    I_grad = {}
    axis_idx = {'y': 0, 'x': 1, 'z': 2}[direction]

    for angle, I in I_fcpm.items():
        n = I.shape[axis_idx]
        ramp = np.linspace(0, magnitude * np.max(I), n)

        shape = [1, 1, 1]
        shape[axis_idx] = n
        ramp = ramp.reshape(shape)

        I_grad[angle] = I + ramp

    return I_grad


def apply_saturation(
    I_fcpm: Dict[float, np.ndarray],
    saturation_level: float = 1.0,
) -> Dict[float, np.ndarray]:
    """Simulate detector saturation (clipping at maximum).

    Args:
        I_fcpm: FCPM intensities.
        saturation_level: Maximum detectable intensity. Pixels above
            this are clipped.

    Returns:
        FCPM intensities with saturation applied.
    """
    return {angle: np.minimum(I, saturation_level) for angle, I in I_fcpm.items()}


def apply_bleaching(
    I_fcpm: Dict[float, np.ndarray],
    bleach_rate: float = 0.05,
) -> Dict[float, np.ndarray]:
    """Simulate progressive photobleaching during sequential acquisition.

    Each subsequent angle measurement receives progressively less signal
    because the fluorophores are being destroyed by the excitation light.

    Args:
        I_fcpm: Clean FCPM intensities.
        bleach_rate: Fractional signal loss per acquisition step.

    Returns:
        FCPM intensities with progressive bleaching.
    """
    I_bleached = {}
    angles_sorted = sorted(I_fcpm.keys())

    for i, angle in enumerate(angles_sorted):
        decay = (1 - bleach_rate) ** i
        I_bleached[angle] = I_fcpm[angle] * decay

    return I_bleached
