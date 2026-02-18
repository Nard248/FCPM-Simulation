"""
Energy functions for director field optimization.

Provides the canonical single-constant gradient energy used throughout the
optimization pipeline, plus an anisotropic Frank elastic energy decomposition
(splay, twist, bend) for detailed physical analysis.

The single-constant approximation:
    E = (K/2) * integral |grad n|^2 dV  ~  (K/2) * sum |n_i - n_j|^2

The full anisotropic Frank energy:
    f = (K1/2)(div n)^2 + (K2/2)(n . curl n + q0)^2 + (K3/2)|n x curl n|^2

where q0 = 2*pi / pitch is the equilibrium twist wave-vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from ..core.director import DirectorField, DTYPE


# ---------------------------------------------------------------------------
# Single-constant gradient energy (canonical implementation)
# ---------------------------------------------------------------------------

def compute_gradient_energy(directors: Union[np.ndarray, DirectorField]) -> float:
    """Compute the total gradient energy (squared-difference approximation).

    E = sum_{axes} sum |n_i - n_{i+1}|^2

    This is the standard cost function minimised by all sign optimizers.
    Lower energy indicates a smoother, more physically realistic field.

    Args:
        directors: Either a raw ``(ny, nx, nz, 3)`` array or a
            :class:`DirectorField`.

    Returns:
        Total gradient energy (scalar).
    """
    if isinstance(directors, DirectorField):
        n = directors.to_array()
    else:
        n = np.asarray(directors)

    energy = 0.0
    for axis in range(3):
        n_shifted = np.roll(n, -1, axis=axis)
        energy += np.sum((n - n_shifted) ** 2)
    return float(energy)


# ---------------------------------------------------------------------------
# Anisotropic Frank elastic energy
# ---------------------------------------------------------------------------

@dataclass
class FrankConstants:
    """Frank elastic constants for a nematic/cholesteric liquid crystal.

    Default values are typical for 5CB-like nematics (in pN).
    For cholesterics, set *pitch* to a finite value; for nematics, leave
    ``pitch=None`` (equivalent to infinite pitch, i.e. q0 = 0).

    References:
        de Gennes & Prost, *The Physics of Liquid Crystals*, 1993.
    """

    K1: float = 10.3   # splay  (pN)
    K2: float = 7.4    # twist  (pN)
    K3: float = 16.48  # bend   (pN)
    pitch: Optional[float] = None  # cholesteric pitch in voxels

    @property
    def q0(self) -> float:
        """Equilibrium twist wave-vector 2*pi / pitch (rad / voxel)."""
        if self.pitch is None or self.pitch == 0:
            return 0.0
        return 2.0 * np.pi / self.pitch


def compute_frank_energy_anisotropic(
    directors: Union[np.ndarray, DirectorField],
    constants: Optional[FrankConstants] = None,
    voxel_size: float = 1.0,
) -> Dict[str, np.ndarray | float]:
    """Compute the full anisotropic Frank elastic energy.

    Decomposes into splay, twist, and bend contributions using
    central finite differences for the spatial derivatives.

    Args:
        directors: Director field as ``(ny, nx, nz, 3)`` array or
            :class:`DirectorField`.
        constants: Frank elastic constants.  Uses defaults if ``None``.
        voxel_size: Physical size of a voxel (isotropic assumed).

    Returns:
        Dictionary with keys:

        * ``splay`` — per-voxel splay energy density array
        * ``twist`` — per-voxel twist energy density array
        * ``bend``  — per-voxel bend energy density array
        * ``total`` — per-voxel total energy density array
        * ``total_integrated`` — scalar integral over volume
        * ``splay_integrated``, ``twist_integrated``, ``bend_integrated``
    """
    if constants is None:
        constants = FrankConstants()

    if isinstance(directors, DirectorField):
        n = directors.to_array().astype(DTYPE)
    else:
        n = np.asarray(directors, dtype=DTYPE)

    nx_c = n[..., 0]
    ny_c = n[..., 1]
    nz_c = n[..., 2]

    h = voxel_size

    # Central finite differences  dn_i / dx_j  (axes: 0=y, 1=x, 2=z)
    def _grad(component: np.ndarray, axis: int) -> np.ndarray:
        return (np.roll(component, -1, axis=axis)
                - np.roll(component, 1, axis=axis)) / (2.0 * h)

    # dn_x/dx, dn_x/dy, dn_x/dz  etc.
    dnx_dx = _grad(nx_c, 1)
    dnx_dy = _grad(nx_c, 0)
    dnx_dz = _grad(nx_c, 2)

    dny_dx = _grad(ny_c, 1)
    dny_dy = _grad(ny_c, 0)
    dny_dz = _grad(ny_c, 2)

    dnz_dx = _grad(nz_c, 1)
    dnz_dy = _grad(nz_c, 0)
    dnz_dz = _grad(nz_c, 2)

    # div(n) = dn_x/dx + dn_y/dy + dn_z/dz
    div_n = dnx_dx + dny_dy + dnz_dz

    # curl(n) = ( dn_z/dy - dn_y/dz,
    #             dn_x/dz - dn_z/dx,
    #             dn_y/dx - dn_x/dy )
    curl_x = dnz_dy - dny_dz
    curl_y = dnx_dz - dnz_dx
    curl_z = dny_dx - dnx_dy

    # n . curl(n)
    n_dot_curl = nx_c * curl_x + ny_c * curl_y + nz_c * curl_z

    # n x curl(n)
    ncurl_x = ny_c * curl_z - nz_c * curl_y
    ncurl_y = nz_c * curl_x - nx_c * curl_z
    ncurl_z = nx_c * curl_y - ny_c * curl_x

    ncurl_sq = ncurl_x**2 + ncurl_y**2 + ncurl_z**2

    # Energy densities  (per voxel)
    q0 = constants.q0
    splay = 0.5 * constants.K1 * div_n**2
    twist = 0.5 * constants.K2 * (n_dot_curl + q0)**2
    bend = 0.5 * constants.K3 * ncurl_sq
    total = splay + twist + bend

    dV = h**3

    return {
        "splay": splay,
        "twist": twist,
        "bend": bend,
        "total": total,
        "splay_integrated": float(np.sum(splay) * dV),
        "twist_integrated": float(np.sum(twist) * dV),
        "bend_integrated": float(np.sum(bend) * dV),
        "total_integrated": float(np.sum(total) * dV),
    }


__all__ = [
    "compute_gradient_energy",
    "FrankConstants",
    "compute_frank_energy_anisotropic",
]
