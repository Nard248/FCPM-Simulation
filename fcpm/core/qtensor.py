"""
Q-Tensor representation for liquid crystal systems.

The Q-tensor is a symmetric, traceless tensor that describes nematic order:
    Q_ij = S * (n_i * n_j - δ_ij / 3)

Key property: Q(n) = Q(-n), eliminating the sign ambiguity inherent in the director.

Classes:
    QTensor: Container for Q-tensor field components.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from .director import DirectorField, DTYPE


@dataclass
class QTensor:
    """
    Represents a 3D Q-tensor field for liquid crystal systems.

    The Q-tensor is a symmetric, traceless 3x3 tensor at each point.
    Due to symmetry and tracelessness, only 5 independent components exist:
    - Q_xx, Q_yy (Q_zz = -Q_xx - Q_yy from tracelessness)
    - Q_xy = Q_yx
    - Q_xz = Q_zx
    - Q_yz = Q_zy

    Attributes:
        Q_xx: xx-component, shape (ny, nx, nz)
        Q_yy: yy-component, shape (ny, nx, nz)
        Q_xy: xy-component, shape (ny, nx, nz)
        Q_xz: xz-component, shape (ny, nx, nz)
        Q_yz: yz-component, shape (ny, nx, nz)
        metadata: Optional dictionary for additional information

    Note:
        Q_zz is not stored but computed as Q_zz = -Q_xx - Q_yy.
    """

    Q_xx: np.ndarray
    Q_yy: np.ndarray
    Q_xy: np.ndarray
    Q_xz: np.ndarray
    Q_yz: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and convert arrays to correct dtype."""
        self.Q_xx = np.asarray(self.Q_xx, dtype=DTYPE)
        self.Q_yy = np.asarray(self.Q_yy, dtype=DTYPE)
        self.Q_xy = np.asarray(self.Q_xy, dtype=DTYPE)
        self.Q_xz = np.asarray(self.Q_xz, dtype=DTYPE)
        self.Q_yz = np.asarray(self.Q_yz, dtype=DTYPE)

        shapes = [self.Q_xx.shape, self.Q_yy.shape, self.Q_xy.shape,
                  self.Q_xz.shape, self.Q_yz.shape]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"All component shapes must match: {shapes}")

        if self.Q_xx.ndim != 3:
            raise ValueError(f"Expected 3D arrays, got {self.Q_xx.ndim}D")

    @property
    def Q_zz(self) -> np.ndarray:
        """Compute Q_zz from tracelessness: Q_zz = -Q_xx - Q_yy."""
        return -self.Q_xx - self.Q_yy

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the Q-tensor field (ny, nx, nz)."""
        return self.Q_xx.shape

    @property
    def size(self) -> int:
        """Return total number of voxels."""
        return self.Q_xx.size

    def to_matrix(self, y: int, x: int, z: int) -> np.ndarray:
        """
        Get the full 3x3 Q-tensor matrix at a specific point.

        Args:
            y, x, z: Voxel indices.

        Returns:
            3x3 symmetric traceless matrix.
        """
        Q = np.array([
            [self.Q_xx[y, x, z], self.Q_xy[y, x, z], self.Q_xz[y, x, z]],
            [self.Q_xy[y, x, z], self.Q_yy[y, x, z], self.Q_yz[y, x, z]],
            [self.Q_xz[y, x, z], self.Q_yz[y, x, z], self.Q_zz[y, x, z]]
        ], dtype=DTYPE)
        return Q

    def to_matrices(self) -> np.ndarray:
        """
        Get all Q-tensor matrices as a 5D array.

        Returns:
            Array of shape (ny, nx, nz, 3, 3).
        """
        shape = self.shape
        Q = np.zeros(shape + (3, 3), dtype=DTYPE)

        Q[..., 0, 0] = self.Q_xx
        Q[..., 0, 1] = self.Q_xy
        Q[..., 0, 2] = self.Q_xz
        Q[..., 1, 0] = self.Q_xy
        Q[..., 1, 1] = self.Q_yy
        Q[..., 1, 2] = self.Q_yz
        Q[..., 2, 0] = self.Q_xz
        Q[..., 2, 1] = self.Q_yz
        Q[..., 2, 2] = self.Q_zz

        return Q

    def scalar_order_parameter(self) -> np.ndarray:
        """
        Compute the scalar order parameter S from the Q-tensor.

        The largest eigenvalue of Q is λ_max = 2S/3.
        This method uses the Frobenius norm for efficiency:
        S = sqrt(3/2 * Tr(Q²))

        Returns:
            Scalar order parameter field S(r).
        """
        # Tr(Q²) = Q_xx² + Q_yy² + Q_zz² + 2*(Q_xy² + Q_xz² + Q_yz²)
        Q_zz = self.Q_zz
        trace_Q2 = (self.Q_xx**2 + self.Q_yy**2 + Q_zz**2 +
                    2 * (self.Q_xy**2 + self.Q_xz**2 + self.Q_yz**2))
        S = np.sqrt(1.5 * trace_Q2)
        return S

    def to_director(self) -> DirectorField:
        """
        Extract the director field from Q-tensor via eigendecomposition.

        The director is the eigenvector corresponding to the largest eigenvalue.
        Note: Due to Q(n) = Q(-n), the sign of the director is arbitrary.

        Returns:
            DirectorField extracted from Q-tensor.
        """
        Q_matrices = self.to_matrices()
        shape = self.shape

        nx = np.zeros(shape, dtype=DTYPE)
        ny = np.zeros(shape, dtype=DTYPE)
        nz = np.zeros(shape, dtype=DTYPE)

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                for iz in range(shape[2]):
                    Q = Q_matrices[iy, ix, iz]
                    eigenvalues, eigenvectors = np.linalg.eigh(Q)
                    # Largest eigenvalue is last (eigh returns sorted)
                    n = eigenvectors[:, -1]
                    nx[iy, ix, iz] = n[0]
                    ny[iy, ix, iz] = n[1]
                    nz[iy, ix, iz] = n[2]

        return DirectorField(
            nx=nx, ny=ny, nz=nz,
            metadata={'source': 'qtensor_eigendecomposition'}
        )

    def to_director_vectorized(self) -> DirectorField:
        """
        Extract director field using vectorized eigendecomposition.

        Faster than to_director() for large fields.

        Returns:
            DirectorField extracted from Q-tensor.
        """
        Q_matrices = self.to_matrices()
        original_shape = self.shape

        # Reshape to (N, 3, 3) for batch eigendecomposition
        Q_flat = Q_matrices.reshape(-1, 3, 3)

        # Batch eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Q_flat)

        # Director is eigenvector of largest eigenvalue (index -1)
        n_flat = eigenvectors[..., -1]  # Shape (N, 3)

        # Reshape back
        n = n_flat.reshape(original_shape + (3,))

        return DirectorField(
            nx=n[..., 0],
            ny=n[..., 1],
            nz=n[..., 2],
            metadata={'source': 'qtensor_eigendecomposition_vectorized'}
        )

    def frobenius_norm(self) -> np.ndarray:
        """
        Compute the Frobenius norm of Q at each point.

        ||Q||_F = sqrt(Tr(Q²))

        Returns:
            Frobenius norm field.
        """
        Q_zz = self.Q_zz
        trace_Q2 = (self.Q_xx**2 + self.Q_yy**2 + Q_zz**2 +
                    2 * (self.Q_xy**2 + self.Q_xz**2 + self.Q_yz**2))
        return np.sqrt(trace_Q2)

    def copy(self) -> 'QTensor':
        """Create a deep copy of this Q-tensor field."""
        return QTensor(
            Q_xx=self.Q_xx.copy(),
            Q_yy=self.Q_yy.copy(),
            Q_xy=self.Q_xy.copy(),
            Q_xz=self.Q_xz.copy(),
            Q_yz=self.Q_yz.copy(),
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        S_mean = np.mean(self.scalar_order_parameter())
        return f"QTensor(shape={self.shape}, mean_S={S_mean:.3f})"


def director_to_qtensor(director: DirectorField, S: float = 1.0) -> QTensor:
    """
    Convert a director field to Q-tensor representation.

    Q_ij = S * (n_i * n_j - δ_ij / 3)

    Args:
        director: Input director field.
        S: Scalar order parameter (default 1.0 for perfect order).

    Returns:
        QTensor representation of the director field.
    """
    nx, ny, nz = director.nx, director.ny, director.nz

    Q_xx = S * (nx * nx - 1.0 / 3.0)
    Q_yy = S * (ny * ny - 1.0 / 3.0)
    Q_xy = S * (nx * ny)
    Q_xz = S * (nx * nz)
    Q_yz = S * (ny * nz)

    return QTensor(
        Q_xx=Q_xx,
        Q_yy=Q_yy,
        Q_xy=Q_xy,
        Q_xz=Q_xz,
        Q_yz=Q_yz,
        metadata={'source': 'director_conversion', 'S': S}
    )


def qtensor_difference(Q1: QTensor, Q2: QTensor) -> np.ndarray:
    """
    Compute the Frobenius norm of the difference between two Q-tensors.

    This is a sign-invariant error metric: ||Q1 - Q2||_F

    Args:
        Q1, Q2: Q-tensor fields to compare.

    Returns:
        Per-voxel Frobenius norm of difference.
    """
    dQ_xx = Q1.Q_xx - Q2.Q_xx
    dQ_yy = Q1.Q_yy - Q2.Q_yy
    dQ_zz = Q1.Q_zz - Q2.Q_zz
    dQ_xy = Q1.Q_xy - Q2.Q_xy
    dQ_xz = Q1.Q_xz - Q2.Q_xz
    dQ_yz = Q1.Q_yz - Q2.Q_yz

    diff_sq = (dQ_xx**2 + dQ_yy**2 + dQ_zz**2 +
               2 * (dQ_xy**2 + dQ_xz**2 + dQ_yz**2))

    return np.sqrt(diff_sq)
