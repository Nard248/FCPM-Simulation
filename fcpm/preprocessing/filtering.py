"""
Filtering and smoothing utilities for FCPM data.

Functions for denoising and preprocessing FCPM intensity images.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Union, Tuple
from ..core.director import DirectorField, DTYPE


def gaussian_filter_fcpm(I_fcpm: Dict[float, np.ndarray],
                         sigma: Union[float, Tuple[float, float, float]] = 1.0) -> Dict[float, np.ndarray]:
    """
    Apply Gaussian smoothing to FCPM intensity data.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        sigma: Standard deviation for Gaussian kernel. Float for isotropic,
               tuple for anisotropic (sigma_y, sigma_x, sigma_z).

    Returns:
        Smoothed FCPM intensities.

    Note:
        Requires scipy.ndimage.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError("scipy required for Gaussian filtering. "
                         "Install with: pip install scipy")

    I_filtered = {}
    for angle, intensity in I_fcpm.items():
        I_filtered[angle] = gaussian_filter(intensity, sigma=sigma).astype(DTYPE)

    return I_filtered


def median_filter_fcpm(I_fcpm: Dict[float, np.ndarray],
                       size: Union[int, Tuple[int, int, int]] = 3) -> Dict[float, np.ndarray]:
    """
    Apply median filtering to FCPM data (good for salt-and-pepper noise).

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        size: Size of the median filter window.

    Returns:
        Median-filtered FCPM intensities.
    """
    try:
        from scipy.ndimage import median_filter
    except ImportError:
        raise ImportError("scipy required for median filtering.")

    I_filtered = {}
    for angle, intensity in I_fcpm.items():
        I_filtered[angle] = median_filter(intensity, size=size).astype(DTYPE)

    return I_filtered


def bilateral_filter_fcpm(I_fcpm: Dict[float, np.ndarray],
                          sigma_spatial: float = 2.0,
                          sigma_intensity: float = 0.1) -> Dict[float, np.ndarray]:
    """
    Apply bilateral filtering (edge-preserving smoothing).

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        sigma_spatial: Spatial smoothing parameter.
        sigma_intensity: Intensity-dependent smoothing parameter.

    Returns:
        Bilateral-filtered FCPM intensities.

    Note:
        This is a simplified 3D bilateral filter implementation.
        For production use, consider OpenCV or skimage implementations.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError("scipy required for bilateral filtering.")

    I_filtered = {}

    for angle, intensity in I_fcpm.items():
        # Simplified bilateral: spatial Gaussian weighted by intensity similarity
        spatial_smooth = gaussian_filter(intensity, sigma=sigma_spatial)

        # Compute weights based on intensity difference
        intensity_diff = np.abs(intensity - spatial_smooth)
        weights = np.exp(-(intensity_diff ** 2) / (2 * sigma_intensity ** 2))

        # Weighted combination
        filtered = weights * intensity + (1 - weights) * spatial_smooth
        I_filtered[angle] = filtered.astype(DTYPE)

    return I_filtered


def remove_background_fcpm(I_fcpm: Dict[float, np.ndarray],
                           method: str = 'percentile',
                           percentile: float = 5.0,
                           sigma: float = 50.0) -> Dict[float, np.ndarray]:
    """
    Remove background from FCPM intensity data.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        method: 'percentile' (subtract low percentile),
                'rolling_ball' (morphological background estimation),
                'gaussian' (subtract heavily smoothed version).
        percentile: Percentile value for 'percentile' method.
        sigma: Smoothing parameter for 'gaussian' method.

    Returns:
        Background-subtracted FCPM intensities.
    """
    I_corrected = {}

    for angle, intensity in I_fcpm.items():
        if method == 'percentile':
            # Subtract a low percentile as background estimate
            background = np.percentile(intensity, percentile)
            corrected = intensity - background

        elif method == 'gaussian':
            try:
                from scipy.ndimage import gaussian_filter
            except ImportError:
                raise ImportError("scipy required for Gaussian background removal.")

            # Estimate background as heavily smoothed image
            background = gaussian_filter(intensity, sigma=sigma)
            corrected = intensity - background + np.mean(background)

        elif method == 'rolling_ball':
            try:
                from scipy.ndimage import grey_opening
            except ImportError:
                raise ImportError("scipy required for rolling ball method.")

            # Morphological opening as background estimate
            size = int(sigma)
            background = grey_opening(intensity, size=size)
            corrected = intensity - background

        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure non-negative
        I_corrected[angle] = np.maximum(corrected, 0).astype(DTYPE)

    return I_corrected


def normalize_fcpm(I_fcpm: Dict[float, np.ndarray],
                   method: str = 'max',
                   target: float = 1.0) -> Dict[float, np.ndarray]:
    """
    Normalize FCPM intensity data.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        method: 'max' (divide by max), 'sum' (divide by sum),
                'per_angle' (normalize each angle independently),
                'global' (normalize all angles together).
        target: Target value for normalization.

    Returns:
        Normalized FCPM intensities.
    """
    I_normalized = {}

    if method == 'per_angle':
        for angle, intensity in I_fcpm.items():
            max_val = np.max(intensity)
            if max_val > 0:
                I_normalized[angle] = (intensity / max_val * target).astype(DTYPE)
            else:
                I_normalized[angle] = intensity.copy()

    elif method in ['max', 'global']:
        # Find global max across all angles
        global_max = max(np.max(I) for I in I_fcpm.values())
        if global_max > 0:
            for angle, intensity in I_fcpm.items():
                I_normalized[angle] = (intensity / global_max * target).astype(DTYPE)
        else:
            I_normalized = {a: I.copy() for a, I in I_fcpm.items()}

    elif method == 'sum':
        for angle, intensity in I_fcpm.items():
            total = np.sum(intensity)
            if total > 0:
                I_normalized[angle] = (intensity / total * target * intensity.size).astype(DTYPE)
            else:
                I_normalized[angle] = intensity.copy()

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return I_normalized


def clip_fcpm(I_fcpm: Dict[float, np.ndarray],
              vmin: Optional[float] = None,
              vmax: Optional[float] = None,
              percentile_range: Optional[Tuple[float, float]] = None) -> Dict[float, np.ndarray]:
    """
    Clip FCPM intensity values to a specified range.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        vmin, vmax: Explicit min/max values. If None, no clipping.
        percentile_range: (low_percentile, high_percentile) to compute bounds.

    Returns:
        Clipped FCPM intensities.
    """
    I_clipped = {}

    # Compute bounds from percentiles if specified
    if percentile_range is not None:
        all_values = np.concatenate([I.flatten() for I in I_fcpm.values()])
        if vmin is None:
            vmin = np.percentile(all_values, percentile_range[0])
        if vmax is None:
            vmax = np.percentile(all_values, percentile_range[1])

    for angle, intensity in I_fcpm.items():
        clipped = intensity.copy()
        if vmin is not None:
            clipped = np.maximum(clipped, vmin)
        if vmax is not None:
            clipped = np.minimum(clipped, vmax)
        I_clipped[angle] = clipped.astype(DTYPE)

    return I_clipped


def smooth_director(director: DirectorField,
                    sigma: float = 1.0,
                    preserve_norm: bool = True) -> DirectorField:
    """
    Smooth a director field using Gaussian filtering.

    Smooths each component independently, then renormalizes.

    Args:
        director: Input director field.
        sigma: Smoothing parameter.
        preserve_norm: If True, renormalize after smoothing.

    Returns:
        Smoothed DirectorField.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError("scipy required for smoothing.")

    nx_smooth = gaussian_filter(director.nx, sigma=sigma)
    ny_smooth = gaussian_filter(director.ny, sigma=sigma)
    nz_smooth = gaussian_filter(director.nz, sigma=sigma)

    result = DirectorField(
        nx=nx_smooth, ny=ny_smooth, nz=nz_smooth,
        metadata={**director.metadata, 'smoothed': True, 'smooth_sigma': sigma}
    )

    if preserve_norm:
        result = result.normalize()

    return result
