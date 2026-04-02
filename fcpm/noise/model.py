"""
Structured noise model for FCPM detector physics.

Models the complete signal chain:
    photons -> detector electrons -> ADU counts -> digital image

Noise sources:
    - Shot noise (Poisson): sqrt(signal) -- fundamental
    - Read noise (Gaussian): constant per pixel -- electronics
    - Dark current: proportional to exposure -- thermal
    - Background: constant offset -- stray light
    - Saturation: hard clipping at detector maximum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class NoiseModel:
    """Physically motivated noise model for FCPM detector.

    Default values approximate a typical scientific CCD/CMOS camera
    used in confocal microscopy.

    Example::

        model = NoiseModel(read_noise_std=2.0, background_level=100.0)
        I_noisy = model.apply(I_clean, seed=42)
        ll = model.log_likelihood(I_noisy, I_clean)
    """

    shot_noise: bool = True
    read_noise_std: float = 0.0
    background_level: float = 0.0
    dark_current: float = 0.0
    gain: float = 1.0
    saturation: float = float('inf')

    def apply(
        self,
        I_clean: Dict[float, np.ndarray],
        seed: Optional[int] = None,
    ) -> Dict[float, np.ndarray]:
        """Apply realistic noise to clean FCPM intensities.

        Signal model for each pixel:
            signal_electrons = gain * I_clean + dark_current
            if shot_noise: signal_electrons ~ Poisson(signal_electrons)
            measured = signal_electrons + read_noise
            measured += background_level
            measured = clip(measured, 0, saturation)

        Args:
            I_clean: Clean FCPM intensities keyed by angle (radians).
            seed: Random seed for reproducibility.

        Returns:
            Noisy FCPM intensities with same keys.
        """
        rng = np.random.default_rng(seed)
        I_noisy = {}

        for angle, I in I_clean.items():
            signal = self.gain * np.maximum(I, 0) + self.dark_current

            if self.shot_noise and np.any(signal > 0):
                # Poisson noise on the signal (in electron counts)
                signal = rng.poisson(np.maximum(signal, 0).astype(np.float64)).astype(np.float64)

            # Add read noise (Gaussian)
            if self.read_noise_std > 0:
                signal = signal + rng.normal(0, self.read_noise_std, size=signal.shape)

            # Add background
            signal = signal + self.background_level

            # Apply saturation
            signal = np.clip(signal, 0, self.saturation)

            I_noisy[angle] = signal

        return I_noisy

    def log_likelihood(
        self,
        I_observed: Dict[float, np.ndarray],
        I_model: Dict[float, np.ndarray],
    ) -> float:
        """Compute log-likelihood of observed data given the model intensities.

        For the mixed Poisson-Gaussian model, uses the saddle-point
        approximation: the effective variance at each pixel is
            var = gain * I_model + read_noise_std^2
        and the log-likelihood is approximately Gaussian:
            log L ~ -0.5 * sum((I_obs - I_model - bg)^2 / var + log(var))

        Args:
            I_observed: Measured FCPM intensities.
            I_model: Model-predicted clean intensities.

        Returns:
            Total log-likelihood (scalar, higher = better fit).
        """
        total_ll = 0.0

        for angle in I_observed:
            if angle not in I_model:
                continue

            obs = I_observed[angle]
            model = self.gain * np.maximum(I_model[angle], 0) + self.dark_current + self.background_level

            # Effective variance: shot noise + read noise
            var = np.maximum(model, 1e-10) + self.read_noise_std ** 2
            var = np.maximum(var, 1e-10)

            residual = obs - model
            ll = -0.5 * np.sum(residual ** 2 / var + np.log(var))
            total_ll += ll

        return float(total_ll)

    def estimate_from_data(
        self,
        I_repeated: Dict[float, list],
    ) -> 'NoiseModel':
        """Estimate noise parameters from repeated measurements.

        Given multiple acquisitions at each angle, estimates the
        background level and read noise from the pixel-wise statistics.

        Args:
            I_repeated: Dict mapping angle to a LIST of repeated
                measurements (arrays), e.g. {0.0: [I1, I2, I3, ...]}.

        Returns:
            A new NoiseModel with estimated parameters.
        """
        all_means = []
        all_vars = []

        for angle, repeats in I_repeated.items():
            stack = np.array(repeats)  # shape: (n_repeats, ...)
            pixel_mean = np.mean(stack, axis=0)
            pixel_var = np.var(stack, axis=0, ddof=1)
            all_means.append(pixel_mean.ravel())
            all_vars.append(pixel_var.ravel())

        means = np.concatenate(all_means)
        variances = np.concatenate(all_vars)

        # For Poisson+Gaussian: var = gain * mean + read_noise^2
        # Linear fit: var = a * mean + b
        valid = means > 0
        if np.sum(valid) < 10:
            return NoiseModel()

        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(means[valid], variances[valid], deg=1)
        gain_est = max(coeffs[1], 0.1)
        read_var = max(coeffs[0], 0.0)

        # Background: estimate from lowest-signal pixels
        bg_est = float(np.percentile(means, 1))

        return NoiseModel(
            shot_noise=True,
            read_noise_std=float(np.sqrt(read_var)),
            background_level=bg_est,
            gain=float(gain_est),
        )
