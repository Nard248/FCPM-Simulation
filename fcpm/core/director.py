"""
Director Field representation for liquid crystal systems.

The director field n(r) describes the average molecular orientation at each point.
In nematic liquid crystals, n ≡ -n (head-tail symmetry).

Classes:
    DirectorField: Container for 3D director field with validation and utilities.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path


DTYPE = np.float64


@dataclass
class DirectorField:
    """
    Represents a 3D director field for liquid crystal systems.

    The director n = (nx, ny, nz) is a unit vector field defined on a 3D grid.
    Due to nematic symmetry, n and -n represent the same physical state.

    Attributes:
        nx: x-component of director field, shape (ny, nx, nz)
        ny: y-component of director field, shape (ny, nx, nz)
        nz: z-component of director field, shape (ny, nx, nz)
        metadata: Optional dictionary for additional information

    Note:
        Array indexing convention: (y_index, x_index, z_index) to match
        typical image conventions where first axis is vertical (y).
    """

    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and convert arrays to correct dtype."""
        self.nx = np.asarray(self.nx, dtype=DTYPE)
        self.ny = np.asarray(self.ny, dtype=DTYPE)
        self.nz = np.asarray(self.nz, dtype=DTYPE)

        if not (self.nx.shape == self.ny.shape == self.nz.shape):
            raise ValueError(
                f"Component shapes must match: "
                f"nx={self.nx.shape}, ny={self.ny.shape}, nz={self.nz.shape}"
            )

        if self.nx.ndim != 3:
            raise ValueError(f"Expected 3D arrays, got {self.nx.ndim}D")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the director field (ny, nx, nz)."""
        return self.nx.shape

    @property
    def size(self) -> int:
        """Return total number of voxels."""
        return self.nx.size

    @property
    def ndim(self) -> int:
        """Return number of dimensions (always 3)."""
        return 3

    def magnitude(self) -> np.ndarray:
        """
        Compute the magnitude |n| at each point.

        For a properly normalized director field, this should be 1.0 everywhere.

        Returns:
            Array of magnitudes with same shape as components.
        """
        return np.sqrt(self.nx**2 + self.ny**2 + self.nz**2)

    def normalize(self, inplace: bool = False) -> 'DirectorField':
        """
        Normalize the director field to unit length.

        Args:
            inplace: If True, modify this field. If False, return new field.

        Returns:
            Normalized DirectorField (self if inplace=True).
        """
        mag = self.magnitude()
        mag = np.where(mag > 1e-10, mag, 1.0)  # Avoid division by zero

        if inplace:
            self.nx /= mag
            self.ny /= mag
            self.nz /= mag
            return self
        else:
            return DirectorField(
                nx=self.nx / mag,
                ny=self.ny / mag,
                nz=self.nz / mag,
                metadata=self.metadata.copy()
            )

    def is_normalized(self, tol: float = 1e-6) -> bool:
        """
        Check if the director field is normalized to unit length.

        Args:
            tol: Tolerance for deviation from unit length.

        Returns:
            True if |n| ≈ 1 everywhere within tolerance.
        """
        mag = self.magnitude()
        return np.allclose(mag, 1.0, atol=tol)

    def to_array(self) -> np.ndarray:
        """
        Convert to a 4D array with shape (ny, nx, nz, 3).

        Returns:
            4D numpy array with components stacked along last axis.
        """
        return np.stack([self.nx, self.ny, self.nz], axis=-1)

    @classmethod
    def from_array(cls, arr: np.ndarray,
                   metadata: Optional[Dict[str, Any]] = None) -> 'DirectorField':
        """
        Create DirectorField from a 4D array.

        Args:
            arr: Array with shape (..., 3) where last axis is (nx, ny, nz).
            metadata: Optional metadata dictionary.

        Returns:
            New DirectorField instance.
        """
        if arr.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3, got {arr.shape[-1]}")

        return cls(
            nx=arr[..., 0],
            ny=arr[..., 1],
            nz=arr[..., 2],
            metadata=metadata or {}
        )

    def flip_signs(self, mask: np.ndarray) -> 'DirectorField':
        """
        Flip the sign of director at specified locations.

        Due to nematic symmetry (n ≡ -n), this doesn't change the physics.
        Useful for ensuring consistent orientation across the field.

        Args:
            mask: Boolean array where True indicates flip locations.

        Returns:
            New DirectorField with flipped signs.
        """
        factor = np.where(mask, -1.0, 1.0)
        return DirectorField(
            nx=self.nx * factor,
            ny=self.ny * factor,
            nz=self.nz * factor,
            metadata=self.metadata.copy()
        )

    def slice_z(self, z_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a z-slice of the director field.

        Args:
            z_idx: Index of z-slice to extract.

        Returns:
            Tuple of (nx, ny, nz) 2D arrays for the slice.
        """
        return self.nx[:, :, z_idx], self.ny[:, :, z_idx], self.nz[:, :, z_idx]

    def copy(self) -> 'DirectorField':
        """Create a deep copy of this director field."""
        return DirectorField(
            nx=self.nx.copy(),
            ny=self.ny.copy(),
            nz=self.nz.copy(),
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        return f"DirectorField(shape={self.shape}, normalized={self.is_normalized()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectorField):
            return NotImplemented
        return (np.allclose(self.nx, other.nx) and
                np.allclose(self.ny, other.ny) and
                np.allclose(self.nz, other.nz))


def create_uniform_director(shape: Tuple[int, int, int],
                            direction: Tuple[float, float, float] = (0, 0, 1)
                            ) -> DirectorField:
    """
    Create a uniform director field pointing in a fixed direction.

    Args:
        shape: Shape of the field (ny, nx, nz).
        direction: Direction vector (will be normalized).

    Returns:
        DirectorField with uniform orientation.
    """
    direction = np.array(direction, dtype=DTYPE)
    direction = direction / np.linalg.norm(direction)

    nx = np.full(shape, direction[0], dtype=DTYPE)
    ny = np.full(shape, direction[1], dtype=DTYPE)
    nz = np.full(shape, direction[2], dtype=DTYPE)

    return DirectorField(nx=nx, ny=ny, nz=nz,
                         metadata={'type': 'uniform', 'direction': tuple(direction)})


def create_cholesteric_director(shape: Tuple[int, int, int],
                                 pitch: float = 10.0,
                                 axis: str = 'z') -> DirectorField:
    """
    Create a cholesteric (twisted nematic) director field.

    The director rotates helically along the specified axis.

    Args:
        shape: Shape of the field (ny, nx, nz).
        pitch: Helical pitch (full 360° rotation distance in grid units).
        axis: Helix axis ('x', 'y', or 'z').

    Returns:
        DirectorField with cholesteric structure.
    """
    ny_dim, nx_dim, nz_dim = shape

    # Create coordinate arrays
    y, x, z = np.meshgrid(
        np.arange(ny_dim),
        np.arange(nx_dim),
        np.arange(nz_dim),
        indexing='ij'
    )

    # Calculate twist angle based on position along helix axis
    if axis == 'z':
        theta = 2 * np.pi * z / pitch
        nx = np.cos(theta)
        ny = np.sin(theta)
        nz = np.zeros_like(theta)
    elif axis == 'y':
        theta = 2 * np.pi * y / pitch
        nx = np.cos(theta)
        ny = np.zeros_like(theta)
        nz = np.sin(theta)
    elif axis == 'x':
        theta = 2 * np.pi * x / pitch
        nx = np.zeros_like(theta)
        ny = np.cos(theta)
        nz = np.sin(theta)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    return DirectorField(
        nx=nx.astype(DTYPE),
        ny=ny.astype(DTYPE),
        nz=nz.astype(DTYPE),
        metadata={'type': 'cholesteric', 'pitch': pitch, 'axis': axis}
    )


def create_radial_director(shape: Tuple[int, int, int],
                           center: Optional[Tuple[float, float, float]] = None
                           ) -> DirectorField:
    """
    Create a radial (hedgehog) director field.

    Director points radially outward from a center point.

    Args:
        shape: Shape of the field (ny, nx, nz).
        center: Center point (y, x, z). If None, uses grid center.

    Returns:
        DirectorField with radial structure.
    """
    ny_dim, nx_dim, nz_dim = shape

    if center is None:
        center = (ny_dim / 2, nx_dim / 2, nz_dim / 2)

    y, x, z = np.meshgrid(
        np.arange(ny_dim),
        np.arange(nx_dim),
        np.arange(nz_dim),
        indexing='ij'
    )

    # Displacement from center
    dy = y - center[0]
    dx = x - center[1]
    dz = z - center[2]

    # Radial distance
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.where(r > 1e-10, r, 1.0)  # Avoid division by zero at center

    # Normalize to get unit vectors
    nx = (dx / r).astype(DTYPE)
    ny = (dy / r).astype(DTYPE)
    nz = (dz / r).astype(DTYPE)

    # Handle center point
    center_idx = tuple(int(c) for c in center)
    if all(0 <= c < s for c, s in zip(center_idx, shape)):
        nx[center_idx] = 0.0
        ny[center_idx] = 0.0
        nz[center_idx] = 1.0  # Arbitrary direction at singularity

    return DirectorField(
        nx=nx, ny=ny, nz=nz,
        metadata={'type': 'radial', 'center': center}
    )
