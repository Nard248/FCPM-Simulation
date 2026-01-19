"""
Q-Tensor Based Reconstruction for FCPM.

The Q-tensor approach eliminates the sign ambiguity problem because:
    Q(n) = Q(-n)

From FCPM intensities, we can uniquely determine:
    - Q_xx = S(nx² - 1/3)
    - Q_yy = S(ny² - 1/3)
    - Q_xy = S·nx·ny (with correct sign!)
    - Q_zz = -Q_xx - Q_yy (from tracelessness)

We CANNOT determine from FCPM alone:
    - Q_xz = S·nx·nz (only |nx·nz|)
    - Q_yz = S·ny·nz (only |ny·nz|)

The director is then extracted via eigendecomposition of Q.
The sign ambiguity (eigenvector ±n) is resolved by spatial propagation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.director import DirectorField, DTYPE
from ..core.qtensor import QTensor, director_to_qtensor


def qtensor_from_fcpm(I_fcpm: Dict[float, np.ndarray],
                      S: float = 1.0) -> Tuple[QTensor, Dict]:
    """
    Reconstruct Q-tensor components from FCPM intensities.

    This determines Q_xx, Q_yy, Q_xy EXACTLY from the measurements.
    Q_xz and Q_yz are set based on the unit constraint but have
    sign ambiguity that cannot be resolved from FCPM alone.

    Args:
        I_fcpm: FCPM intensities at standard angles [0, π/4, π/2, 3π/4].
        S: Assumed scalar order parameter.

    Returns:
        Tuple of (QTensor, info_dict with reconstruction details).
    """
    # Extract component magnitudes
    I_0 = I_fcpm.get(0.0, I_fcpm.get(0))
    I_45 = I_fcpm.get(np.pi/4, None)
    I_90 = I_fcpm.get(np.pi/2, None)
    I_135 = I_fcpm.get(3*np.pi/4, None)

    if any(I is None for I in [I_0, I_45, I_90, I_135]):
        raise ValueError("Need intensities at [0, π/4, π/2, 3π/4]")

    # Fourth roots
    sqrt4_0 = np.power(np.maximum(I_0, 0), 0.25)
    sqrt4_45 = np.power(np.maximum(I_45, 0), 0.25)
    sqrt4_90 = np.power(np.maximum(I_90, 0), 0.25)
    sqrt4_135 = np.power(np.maximum(I_135, 0), 0.25)

    # Director component magnitudes squared
    nx_sq = sqrt4_0 ** 2
    ny_sq = sqrt4_90 ** 2

    # Signed cross term: nx·ny
    # I(π/4)^(1/4) = |nx + ny|/√2, I(3π/4)^(1/4) = |ny - nx|/√2
    nx_ny = (sqrt4_45 - sqrt4_135) / np.sqrt(2)

    # Q-tensor components (assuming S=1)
    # Q_ij = S(n_i·n_j - δ_ij/3)
    Q_xx = S * (nx_sq - 1.0/3.0)
    Q_yy = S * (ny_sq - 1.0/3.0)
    Q_xy = S * nx_ny

    # For Q_xz, Q_yz: we know magnitudes but not signs
    # nz² = 1 - nx² - ny²
    sum_sq = np.minimum(nx_sq + ny_sq, 1.0)
    nz_sq = np.maximum(1.0 - sum_sq, 0)

    # |nx·nz| = |nx|·|nz|, |ny·nz| = |ny|·|nz|
    # We choose positive signs (will be resolved by propagation later)
    nx_abs = np.sqrt(np.maximum(nx_sq, 0))
    ny_abs = np.sqrt(np.maximum(ny_sq, 0))
    nz_abs = np.sqrt(nz_sq)

    Q_xz = S * nx_abs * nz_abs  # Sign ambiguous!
    Q_yz = S * ny_abs * nz_abs  # Sign ambiguous!

    info = {
        'method': 'qtensor_from_fcpm',
        'S': S,
        'nx_squared_mean': float(np.mean(nx_sq)),
        'ny_squared_mean': float(np.mean(ny_sq)),
        'nz_squared_mean': float(np.mean(nz_sq)),
        'qxz_qyz_sign_ambiguous': True,
    }

    return QTensor(Q_xx=Q_xx, Q_yy=Q_yy, Q_xy=Q_xy, Q_xz=Q_xz, Q_yz=Q_yz,
                   metadata={'source': 'fcpm_reconstruction'}), info


def qtensor_from_fcpm_exact(I_fcpm: Dict[float, np.ndarray],
                            S: float = 1.0) -> Tuple[QTensor, Dict]:
    """
    Reconstruct ONLY the exactly determinable Q-tensor components.

    Sets Q_xz = Q_yz = 0 since these cannot be determined from FCPM.
    The director extraction will work but may have errors in nz.

    Args:
        I_fcpm: FCPM intensities.
        S: Scalar order parameter.

    Returns:
        Tuple of (QTensor with Q_xz=Q_yz=0, info_dict).
    """
    Q_full, info = qtensor_from_fcpm(I_fcpm, S)

    # Zero out the ambiguous components
    Q_exact = QTensor(
        Q_xx=Q_full.Q_xx,
        Q_yy=Q_full.Q_yy,
        Q_xy=Q_full.Q_xy,
        Q_xz=np.zeros_like(Q_full.Q_xz),
        Q_yz=np.zeros_like(Q_full.Q_yz),
        metadata={'source': 'fcpm_reconstruction_exact'}
    )

    info['method'] = 'qtensor_from_fcpm_exact'
    info['qxz_qyz_zeroed'] = True

    return Q_exact, info


def reconstruct_via_qtensor(I_fcpm: Dict[float, np.ndarray],
                            S: float = 1.0,
                            vectorized: bool = True) -> Tuple[DirectorField, QTensor, Dict]:
    """
    Full reconstruction pipeline using Q-tensor approach.

    Steps:
    1. Reconstruct Q-tensor components from FCPM
    2. Extract director via eigendecomposition
    3. Director has arbitrary sign (±n) at each point

    Note: Sign consistency must be enforced separately using
    the sign_optimization module.

    Args:
        I_fcpm: FCPM intensities.
        S: Scalar order parameter.
        vectorized: Use vectorized eigendecomposition (faster).

    Returns:
        Tuple of (DirectorField, QTensor, info_dict).
    """
    # Step 1: Q-tensor from FCPM
    Q, info = qtensor_from_fcpm(I_fcpm, S)

    # Step 2: Director from eigendecomposition
    if vectorized:
        director = Q.to_director_vectorized()
    else:
        director = Q.to_director()

    info['eigendecomposition'] = 'vectorized' if vectorized else 'loop'

    return director, Q, info


def compute_qtensor_error(Q_recon: QTensor, Q_gt: QTensor,
                          components: str = 'all') -> Dict[str, float]:
    """
    Compute reconstruction error metrics for Q-tensor.

    Args:
        Q_recon: Reconstructed Q-tensor.
        Q_gt: Ground truth Q-tensor.
        components: 'all', 'in_plane' (xx, yy, xy only), or 'exact' (same as in_plane).

    Returns:
        Dictionary of error metrics.
    """
    if components in ['in_plane', 'exact']:
        # Only compare the exactly determinable components
        diff_xx = Q_recon.Q_xx - Q_gt.Q_xx
        diff_yy = Q_recon.Q_yy - Q_gt.Q_yy
        diff_xy = Q_recon.Q_xy - Q_gt.Q_xy

        rmse = np.sqrt(np.mean(diff_xx**2 + diff_yy**2 + 2*diff_xy**2))
        max_error = np.max(np.abs(np.stack([diff_xx, diff_yy, diff_xy])))

    else:  # 'all'
        from ..core.qtensor import qtensor_difference
        diff = qtensor_difference(Q_recon, Q_gt)
        rmse = np.sqrt(np.mean(diff**2))
        max_error = np.max(diff)

    return {
        'rmse': float(rmse),
        'max_error': float(max_error),
        'components': components,
    }
