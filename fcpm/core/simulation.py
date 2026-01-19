"""
FCPM (Fluorescence Confocal Polarizing Microscopy) simulation.

Physics: Two-photon excitation with polarized light.
The fluorescence intensity depends on the angle between the polarization
direction and the local director orientation:

    I(α) ∝ cos⁴(β) = [nx·cos(α) + ny·sin(α)]⁴

where:
    - α is the polarization angle
    - β is the angle between polarization and in-plane director
    - (nx, ny) are the in-plane director components

Key insight: I depends only on the IN-PLANE projection of the director,
not the full 3D orientation. The out-of-plane component nz affects only
the overall intensity scale (through the normalization |n_xy|²).

Functions:
    simulate_fcpm: Generate FCPM intensity for a director field.
    simulate_fcpm_with_noise: Add realistic noise to simulated FCPM.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .director import DirectorField, DTYPE


def simulate_fcpm(director: DirectorField,
                  angles: Optional[List[float]] = None,
                  normalize_intensity: bool = True) -> Dict[float, np.ndarray]:
    """
    Simulate FCPM intensity images for a director field.

    Physics: I(α) = [nx·cos(α) + ny·sin(α)]⁴

    This follows from two-photon absorption where the absorption probability
    is proportional to cos⁴(β), with β being the angle between the excitation
    polarization and the transition dipole moment (parallel to director).

    Args:
        director: Input director field.
        angles: List of polarization angles in radians.
                Default is [0, π/4, π/2, 3π/4] (4 standard angles).
        normalize_intensity: If True, normalize max intensity to 1.0.

    Returns:
        Dictionary mapping angle -> intensity array of shape (ny, nx, nz).
    """
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    nx, ny = director.nx, director.ny
    I_fcpm = {}

    for alpha in angles:
        # Projection onto polarization direction
        projection = nx * np.cos(alpha) + ny * np.sin(alpha)

        # Two-photon absorption: I ∝ cos⁴(β) = projection⁴
        intensity = projection ** 4

        if normalize_intensity:
            max_val = np.max(intensity)
            if max_val > 0:
                intensity = intensity / max_val

        I_fcpm[alpha] = intensity.astype(DTYPE)

    return I_fcpm


def simulate_fcpm_extended(director: DirectorField,
                           angles: Optional[List[float]] = None,
                           include_z_contribution: bool = False) -> Dict[float, np.ndarray]:
    """
    Extended FCPM simulation with optional z-component effects.

    Standard FCPM: I(α) = [nx·cos(α) + ny·sin(α)]⁴

    With z-contribution (models partial transmission of z-polarized light):
    I(α) ∝ (nx² + ny²)² * cos⁴(effective_angle)

    In practice, the standard model is usually sufficient.

    Args:
        director: Input director field.
        angles: List of polarization angles in radians.
        include_z_contribution: Whether to modulate by (1 - nz²).

    Returns:
        Dictionary mapping angle -> intensity array.
    """
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    nx, ny, nz = director.nx, director.ny, director.nz
    I_fcpm = {}

    for alpha in angles:
        projection = nx * np.cos(alpha) + ny * np.sin(alpha)
        intensity = projection ** 4

        if include_z_contribution:
            # Modulate by in-plane magnitude squared
            in_plane_mag_sq = nx**2 + ny**2
            intensity = intensity * in_plane_mag_sq

        I_fcpm[alpha] = intensity.astype(DTYPE)

    return I_fcpm


def add_gaussian_noise(I_fcpm: Dict[float, np.ndarray],
                       noise_level: float = 0.05,
                       seed: Optional[int] = None) -> Dict[float, np.ndarray]:
    """
    Add Gaussian noise to FCPM intensity images.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        noise_level: Standard deviation of noise relative to max intensity.
        seed: Random seed for reproducibility.

    Returns:
        Noisy FCPM intensities (clipped to [0, ∞)).
    """
    if seed is not None:
        np.random.seed(seed)

    I_noisy = {}
    for alpha, intensity in I_fcpm.items():
        max_val = np.max(intensity)
        noise = np.random.normal(0, noise_level * max_val, intensity.shape)
        I_noisy[alpha] = np.maximum(intensity + noise, 0).astype(DTYPE)

    return I_noisy


def add_poisson_noise(I_fcpm: Dict[float, np.ndarray],
                      photon_count: float = 1000.0,
                      seed: Optional[int] = None) -> Dict[float, np.ndarray]:
    """
    Add Poisson (shot) noise to FCPM intensity images.

    This models the photon counting statistics in real microscopy.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        photon_count: Expected photon count at max intensity.
        seed: Random seed for reproducibility.

    Returns:
        Noisy FCPM intensities with Poisson statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    I_noisy = {}
    for alpha, intensity in I_fcpm.items():
        # Scale to photon counts
        scaled = intensity * photon_count
        # Sample from Poisson distribution
        noisy = np.random.poisson(np.maximum(scaled, 0))
        # Scale back
        I_noisy[alpha] = (noisy / photon_count).astype(DTYPE)

    return I_noisy


def compute_fcpm_observables(I_fcpm: Dict[float, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute observable quantities from FCPM intensity images.

    From 4 polarization angles, we can extract:
    - I_sum: Total intensity (related to in-plane director magnitude)
    - I_diff: Intensity difference revealing director angle

    For standard angles [0, π/4, π/2, 3π/4]:
    - nx² ∝ √I(0) - √I(π/2)
    - ny² ∝ √I(π/2) - √I(0)  (or use I(0) + I(π/2))
    - nx·ny ∝ √I(π/4) - √I(3π/4)

    Args:
        I_fcpm: Dictionary of FCPM intensities for standard angles.

    Returns:
        Dictionary of observable quantities.
    """
    # Extract standard angles
    angles = sorted(I_fcpm.keys())

    if len(angles) < 4:
        raise ValueError("Need at least 4 angles for full analysis")

    # Assume standard angles
    I_0 = I_fcpm.get(0, I_fcpm.get(0.0))
    I_45 = I_fcpm.get(np.pi/4, None)
    I_90 = I_fcpm.get(np.pi/2, None)
    I_135 = I_fcpm.get(3*np.pi/4, None)

    if any(I is None for I in [I_0, I_45, I_90, I_135]):
        # Try to find closest angles
        raise ValueError("Standard angles [0, π/4, π/2, 3π/4] required")

    # Take fourth root to get effective projections
    sqrt4_I_0 = np.power(I_0, 0.25)
    sqrt4_I_45 = np.power(I_45, 0.25)
    sqrt4_I_90 = np.power(I_90, 0.25)
    sqrt4_I_135 = np.power(I_135, 0.25)

    observables = {
        'I_sum': I_0 + I_45 + I_90 + I_135,
        'nx_squared': (sqrt4_I_0**2 + sqrt4_I_90**2) / 2,  # Approximate
        'nx_ny_product': (sqrt4_I_45 - sqrt4_I_135) / 2,   # Signed!
        'in_plane_anisotropy': I_0 - I_90,
    }

    return observables


class FCPMSimulator:
    """
    High-level FCPM simulation interface.

    Provides a convenient object-oriented interface for simulating
    FCPM measurements with various options.

    Attributes:
        director: The director field to simulate.
        angles: Polarization angles used.
        noise_type: Type of noise ('gaussian', 'poisson', or None).
        noise_level: Noise magnitude.
    """

    def __init__(self,
                 director: DirectorField,
                 angles: Optional[List[float]] = None,
                 noise_type: Optional[str] = None,
                 noise_level: float = 0.05):
        """
        Initialize FCPM simulator.

        Args:
            director: Director field to simulate.
            angles: Polarization angles in radians.
            noise_type: 'gaussian', 'poisson', or None.
            noise_level: Noise magnitude (std for Gaussian, photon count for Poisson).
        """
        self.director = director
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.noise_type = noise_type
        self.noise_level = noise_level

        self._I_fcpm: Optional[Dict[float, np.ndarray]] = None

    def simulate(self, seed: Optional[int] = None) -> Dict[float, np.ndarray]:
        """
        Run FCPM simulation.

        Args:
            seed: Random seed for noise generation.

        Returns:
            Dictionary of simulated FCPM intensities.
        """
        # Generate clean signal
        I_fcpm = simulate_fcpm(self.director, self.angles)

        # Add noise if requested
        if self.noise_type == 'gaussian':
            I_fcpm = add_gaussian_noise(I_fcpm, self.noise_level, seed)
        elif self.noise_type == 'poisson':
            I_fcpm = add_poisson_noise(I_fcpm, self.noise_level, seed)

        self._I_fcpm = I_fcpm
        return I_fcpm

    @property
    def intensities(self) -> Optional[Dict[float, np.ndarray]]:
        """Return last simulated intensities."""
        return self._I_fcpm

    def to_array(self) -> np.ndarray:
        """
        Stack intensities into a 4D array.

        Returns:
            Array of shape (n_angles, ny, nx, nz).
        """
        if self._I_fcpm is None:
            raise ValueError("No simulation data. Call simulate() first.")

        sorted_angles = sorted(self._I_fcpm.keys())
        return np.stack([self._I_fcpm[a] for a in sorted_angles], axis=0)
