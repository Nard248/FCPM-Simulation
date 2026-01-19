"""
Noise Utilities for FCPM Simulation.

Functions for adding realistic noise to simulated data and
for noise characterization/removal.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from ..core.director import DTYPE


def add_gaussian_noise(data: np.ndarray,
                       sigma: float = 0.05,
                       relative: bool = True,
                       seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to data.

    Args:
        data: Input array.
        sigma: Noise standard deviation.
        relative: If True, sigma is relative to data max.
        seed: Random seed.

    Returns:
        Noisy data.
    """
    if seed is not None:
        np.random.seed(seed)

    if relative:
        sigma = sigma * np.max(np.abs(data))

    noise = np.random.normal(0, sigma, data.shape)
    return (data + noise).astype(DTYPE)


def add_poisson_noise(data: np.ndarray,
                      photon_count: float = 1000.0,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Add Poisson (shot) noise to simulate photon counting.

    Args:
        data: Input array (interpreted as expected photon rate).
        photon_count: Scale factor for photon counts.
        seed: Random seed.

    Returns:
        Noisy data with Poisson statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Scale to photon counts
    scaled = data * photon_count
    scaled = np.maximum(scaled, 0)

    # Sample from Poisson distribution
    noisy = np.random.poisson(scaled)

    # Scale back
    return (noisy / photon_count).astype(DTYPE)


def add_salt_pepper_noise(data: np.ndarray,
                          fraction: float = 0.01,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Add salt and pepper noise (random extreme values).

    Args:
        data: Input array.
        fraction: Fraction of pixels to affect.
        seed: Random seed.

    Returns:
        Noisy data.
    """
    if seed is not None:
        np.random.seed(seed)

    output = data.copy()
    n_salt = int(fraction * data.size / 2)
    n_pepper = int(fraction * data.size / 2)

    # Salt (max values)
    salt_coords = tuple(np.random.randint(0, dim, n_salt)
                        for dim in data.shape)
    output[salt_coords] = np.max(data)

    # Pepper (zero values)
    pepper_coords = tuple(np.random.randint(0, dim, n_pepper)
                          for dim in data.shape)
    output[pepper_coords] = 0

    return output.astype(DTYPE)


def add_fcpm_realistic_noise(I_fcpm: Dict[float, np.ndarray],
                             noise_model: str = 'mixed',
                             gaussian_sigma: float = 0.02,
                             photon_count: float = 10000.0,
                             seed: Optional[int] = None) -> Dict[float, np.ndarray]:
    """
    Add realistic noise to FCPM intensity data.

    Realistic noise includes:
    - Shot noise (Poisson) from photon counting
    - Read noise (Gaussian) from detector
    - Background noise

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        noise_model: 'gaussian', 'poisson', or 'mixed'.
        gaussian_sigma: Gaussian noise level.
        photon_count: Expected photon count for Poisson.
        seed: Random seed.

    Returns:
        Noisy FCPM intensities.
    """
    if seed is not None:
        np.random.seed(seed)

    I_noisy = {}

    for angle, intensity in I_fcpm.items():
        if noise_model == 'gaussian':
            noisy = add_gaussian_noise(intensity, gaussian_sigma, relative=True)
        elif noise_model == 'poisson':
            noisy = add_poisson_noise(intensity, photon_count)
        elif noise_model == 'mixed':
            # Shot noise + read noise
            noisy = add_poisson_noise(intensity, photon_count)
            noisy = add_gaussian_noise(noisy, gaussian_sigma, relative=True)
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")

        # Clip to valid range
        I_noisy[angle] = np.clip(noisy, 0, None).astype(DTYPE)

    return I_noisy


def estimate_noise_level(data: np.ndarray,
                         method: str = 'mad') -> float:
    """
    Estimate noise level in data.

    Args:
        data: Input data.
        method: 'std' (standard deviation), 'mad' (median absolute deviation).

    Returns:
        Estimated noise level.
    """
    if method == 'std':
        return float(np.std(data))
    elif method == 'mad':
        # Robust estimate using MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        # Scale factor to convert MAD to sigma for Gaussian
        return float(1.4826 * mad)
    else:
        raise ValueError(f"Unknown method: {method}")


def signal_to_noise_ratio(signal: np.ndarray,
                          noise: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio (SNR).

    Args:
        signal: Signal array.
        noise: Noise array (or noisy data minus signal).

    Returns:
        SNR in dB.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-20:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)
