"""
Direct Inversion Methods for FCPM Reconstruction.

These methods directly invert the FCPM intensity formula to recover
the director field components without optimization.

Theory:
    I(α) = [nx·cos(α) + ny·sin(α)]⁴

From 4 standard angles [0, π/4, π/2, 3π/4], we can extract:
    - nx² and ny² from I(0) and I(π/2)
    - nx·ny (with sign!) from I(π/4) and I(3π/4)
    - nz² from unit constraint: nz² = 1 - nx² - ny²

Key insight: The sign of nx·ny is DETERMINABLE from FCPM.
The individual signs of nx, ny, nz require additional information.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.director import DirectorField, DTYPE
from ..core.qtensor import QTensor


def extract_magnitudes(I_fcpm: Dict[float, np.ndarray],
                       epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract squared director components from FCPM intensities.

    From the FCPM formula I(α) = (nx·cos(α) + ny·sin(α))⁴:
    - I^(1/4)(0) = |nx|
    - I^(1/4)(π/2) = |ny|
    - nz² = 1 - nx² - ny²

    Args:
        I_fcpm: Dictionary of FCPM intensities for angles [0, π/4, π/2, 3π/4].
        epsilon: Small value to avoid numerical issues.

    Returns:
        Tuple of (nx_squared, ny_squared, nz_squared).
    """
    # Get intensities at key angles
    I_0 = I_fcpm.get(0.0, I_fcpm.get(0))
    I_90 = I_fcpm.get(np.pi/2, None)

    if I_0 is None or I_90 is None:
        raise ValueError("Need intensities at α=0 and α=π/2")

    # Fourth root to get |projection|
    nx_abs = np.power(np.maximum(I_0, 0), 0.25)
    ny_abs = np.power(np.maximum(I_90, 0), 0.25)

    nx_squared = nx_abs ** 2
    ny_squared = ny_abs ** 2

    # Enforce unit constraint
    sum_sq = nx_squared + ny_squared
    # Clip to ensure nz² >= 0
    sum_sq = np.minimum(sum_sq, 1.0)
    nz_squared = 1.0 - sum_sq

    return nx_squared.astype(DTYPE), ny_squared.astype(DTYPE), nz_squared.astype(DTYPE)


def extract_cross_term(I_fcpm: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Extract the signed nx·ny product from FCPM intensities.

    This is the KEY quantity that FCPM can determine with correct sign!

    At α = π/4:  I = (nx/√2 + ny/√2)⁴ = ((nx + ny)/√2)⁴ = (nx + ny)⁴/4
    At α = 3π/4: I = (-nx/√2 + ny/√2)⁴ = ((-nx + ny)/√2)⁴ = (ny - nx)⁴/4

    Therefore:
        I^(1/4)(π/4) - I^(1/4)(3π/4) = |nx + ny|/√2 - |ny - nx|/√2

    For small angle regime (continuous director):
        nx·ny ≈ (√I(π/4) - √I(3π/4)) / 2

    Args:
        I_fcpm: Dictionary of FCPM intensities.

    Returns:
        Signed nx·ny product.
    """
    I_45 = I_fcpm.get(np.pi/4, None)
    I_135 = I_fcpm.get(3*np.pi/4, None)

    if I_45 is None or I_135 is None:
        raise ValueError("Need intensities at α=π/4 and α=3π/4")

    # Fourth root
    sqrt4_I_45 = np.power(np.maximum(I_45, 0), 0.25)
    sqrt4_I_135 = np.power(np.maximum(I_135, 0), 0.25)

    # The difference gives signed information
    nx_ny = (sqrt4_I_45 - sqrt4_I_135) / np.sqrt(2)

    return nx_ny.astype(DTYPE)


def reconstruct_director_direct(I_fcpm: Dict[float, np.ndarray],
                                 sign_nz_positive: bool = True) -> DirectorField:
    """
    Reconstruct director field using direct inversion.

    This method uses:
    1. Magnitude extraction from I(0), I(π/2)
    2. Sign determination for nx·ny from I(π/4), I(3π/4)
    3. Unit constraint for nz magnitude
    4. Assumed positive nz (can be flipped later)

    Limitations:
    - Individual signs of nx, ny are ambiguous (but product nx·ny is known)
    - Sign of nz is assumed (typically positive for upward-facing samples)

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        sign_nz_positive: Assume nz >= 0 everywhere.

    Returns:
        Reconstructed DirectorField.
    """
    # Extract magnitudes
    nx_sq, ny_sq, nz_sq = extract_magnitudes(I_fcpm)

    # Get signed cross term
    nx_ny = extract_cross_term(I_fcpm)

    # Determine relative signs of nx and ny from the cross term
    # If nx·ny > 0, same sign. If nx·ny < 0, opposite signs.

    # We need to choose a sign convention. Let's use:
    # - nx >= 0 (arbitrary choice for one component)
    # - ny sign determined by nx·ny
    nx = np.sqrt(np.maximum(nx_sq, 0))
    ny = np.sign(nx_ny) * np.sqrt(np.maximum(ny_sq, 0))

    # Handle zero case for nx (when nx_ny would be unreliable)
    # In this case, ny sign is arbitrary
    zero_nx = nx_sq < 1e-10
    ny = np.where(zero_nx, np.sqrt(np.maximum(ny_sq, 0)), ny)

    # nz magnitude
    nz = np.sqrt(np.maximum(nz_sq, 0))
    if not sign_nz_positive:
        nz = -nz

    # Normalize
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    mag = np.where(mag > 1e-10, mag, 1.0)
    nx = nx / mag
    ny = ny / mag
    nz = nz / mag

    return DirectorField(
        nx=nx, ny=ny, nz=nz,
        metadata={'method': 'direct_inversion'}
    )


def reconstruct_director_all_angles(I_fcpm: Dict[float, np.ndarray]) -> DirectorField:
    """
    Reconstruct director using all available angles via least-squares.

    This method fits the model I(α) = [nx·cos(α) + ny·sin(α)]⁴
    to all available angle measurements.

    Args:
        I_fcpm: Dictionary of FCPM intensities (any number of angles).

    Returns:
        Reconstructed DirectorField.
    """
    angles = np.array(sorted(I_fcpm.keys()))
    n_angles = len(angles)

    if n_angles < 2:
        raise ValueError("Need at least 2 angles for reconstruction")

    # Get shape from first intensity
    shape = list(I_fcpm.values())[0].shape

    # Stack intensities
    I_stack = np.stack([I_fcpm[a] for a in angles], axis=-1)  # shape + (n_angles,)

    # Take fourth root
    sqrt4_I = np.power(np.maximum(I_stack, 0), 0.25)

    # For each voxel, solve for nx, ny using least squares
    # Model: sqrt4_I(α) = |nx·cos(α) + ny·sin(α)|

    # This is a linear problem in (nx, ny) if we ignore the absolute value
    # For continuous fields, we can assume the projection doesn't change sign
    # across angles, so: sqrt4_I(α) ≈ nx·cos(α) + ny·sin(α)

    # Design matrix
    A = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_angles, 2)

    # Solve per voxel using pseudo-inverse
    # sqrt4_I has shape (*shape, n_angles), we need to solve A @ [nx, ny] = sqrt4_I
    sqrt4_I_flat = sqrt4_I.reshape(-1, n_angles)  # (N, n_angles)

    # Least squares solution: x = (A^T A)^-1 A^T b
    AtA_inv_At = np.linalg.lstsq(A, np.eye(n_angles), rcond=None)[0].T  # (2, n_angles)
    n_xy = sqrt4_I_flat @ AtA_inv_At.T  # (N, 2)

    nx = n_xy[:, 0].reshape(shape)
    ny = n_xy[:, 1].reshape(shape)

    # Compute nz from unit constraint
    nx_sq = nx ** 2
    ny_sq = ny ** 2
    sum_sq = np.minimum(nx_sq + ny_sq, 1.0)
    nz = np.sqrt(1.0 - sum_sq)

    # Normalize
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    mag = np.where(mag > 1e-10, mag, 1.0)

    return DirectorField(
        nx=(nx / mag).astype(DTYPE),
        ny=(ny / mag).astype(DTYPE),
        nz=(nz / mag).astype(DTYPE),
        metadata={'method': 'all_angles_lstsq', 'n_angles': n_angles}
    )
