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


def create_disclination_director(
    shape: Tuple[int, int, int],
    charge: float = 1.0,
    axis: str = 'z',
    center: Optional[Tuple[float, float]] = None,
) -> DirectorField:
    """Create a director field with a line disclination of given topological charge.

    The disclination line runs along the specified axis. In the plane
    perpendicular to the axis, the director angle varies as:
        phi = charge * arctan2(dy, dx) + phase_offset

    For charge = +1: radial/azimuthal pattern
    For charge = -1: saddle pattern
    For charge = +1/2 or -1/2: half-integer disclination (no smooth
        director exists — requires branch cuts)

    Args:
        shape: (ny, nx, nz) grid dimensions.
        charge: Topological charge. Common values: +1, -1, +0.5, -0.5.
        axis: Direction of the disclination line ('x', 'y', or 'z').
        center: Center of the disclination in the perpendicular plane.
            If None, uses the grid center.

    Returns:
        DirectorField. For half-integer charges, the field will have a
        branch cut (discontinuity) — this is topologically unavoidable.
    """
    ny_dim, nx_dim, nz_dim = shape

    y, x, z = np.meshgrid(
        np.arange(ny_dim, dtype=DTYPE),
        np.arange(nx_dim, dtype=DTYPE),
        np.arange(nz_dim, dtype=DTYPE),
        indexing='ij',
    )

    if axis == 'z':
        cy = center[0] if center else ny_dim / 2.0
        cx = center[1] if center else nx_dim / 2.0
        theta = charge * np.arctan2(y - cy, x - cx)
        nx_arr = np.cos(theta)
        ny_arr = np.sin(theta)
        nz_arr = np.zeros_like(theta)
    elif axis == 'y':
        cx = center[0] if center else nx_dim / 2.0
        cz = center[1] if center else nz_dim / 2.0
        theta = charge * np.arctan2(z - cz, x - cx)
        nx_arr = np.cos(theta)
        ny_arr = np.zeros_like(theta)
        nz_arr = np.sin(theta)
    elif axis == 'x':
        cy = center[0] if center else ny_dim / 2.0
        cz = center[1] if center else nz_dim / 2.0
        theta = charge * np.arctan2(z - cz, y - cy)
        nx_arr = np.zeros_like(theta)
        ny_arr = np.cos(theta)
        nz_arr = np.sin(theta)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    return DirectorField(
        nx=nx_arr.astype(DTYPE),
        ny=ny_arr.astype(DTYPE),
        nz=nz_arr.astype(DTYPE),
        metadata={
            'type': 'disclination',
            'charge': charge,
            'axis': axis,
            'center': center,
            'has_branch_cut': abs(charge) % 1 != 0,
        },
    )


def create_skyrmion_director(
    shape: Tuple[int, int, int],
    radius: float = 10.0,
    helicity: float = 0.0,
    center: Optional[Tuple[float, float]] = None,
) -> DirectorField:
    """Create a 2D baby-Skyrmion director field.

    The director tilts from vertical (nz=1) at the center to vertical
    (nz=-1) far away, passing through in-plane at r = radius. The
    in-plane angle has a helicity twist.

    Args:
        shape: (ny, nx, nz) grid dimensions.
        radius: Characteristic radius of the Skyrmion (voxels).
        helicity: In-plane angle offset (0 = Neel, pi/2 = Bloch).
        center: Center in the xy-plane. If None, uses grid center.

    Returns:
        DirectorField with Skyrmion structure.
    """
    ny_dim, nx_dim, nz_dim = shape
    cy = center[0] if center else ny_dim / 2.0
    cx = center[1] if center else nx_dim / 2.0

    y, x, z = np.meshgrid(
        np.arange(ny_dim, dtype=DTYPE),
        np.arange(nx_dim, dtype=DTYPE),
        np.arange(nz_dim, dtype=DTYPE),
        indexing='ij',
    )

    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)

    # Tilt angle: 0 at center (nz=1), pi at r >> radius (nz=-1)
    tilt = np.pi * (1 - np.exp(-(r / radius) ** 2))

    # Director components
    nx_arr = np.sin(tilt) * np.cos(phi + helicity)
    ny_arr = np.sin(tilt) * np.sin(phi + helicity)
    nz_arr = np.cos(tilt)

    return DirectorField(
        nx=nx_arr.astype(DTYPE),
        ny=ny_arr.astype(DTYPE),
        nz=nz_arr.astype(DTYPE),
        metadata={
            'type': 'skyrmion',
            'radius': radius,
            'helicity': helicity,
            'center': (cy, cx),
        },
    )


def create_toron_director(
    shape: Tuple[int, int, int],
    pitch: float = 16.0,
    position: Optional[Tuple[float, float, float]] = None,
) -> DirectorField:
    """Create an approximate toron director field.

    A toron is a localized structure in a cholesteric: a double-twist
    cylinder capped by two point defects. This creates an approximate
    version using a localized twist modulation on a cholesteric background.

    Args:
        shape: (ny, nx, nz) grid dimensions.
        pitch: Cholesteric pitch (voxels).
        position: Center of the toron (y, x, z). If None, uses grid center.

    Returns:
        DirectorField with toron-like structure.
    """
    ny_dim, nx_dim, nz_dim = shape
    if position is None:
        position = (ny_dim / 2.0, nx_dim / 2.0, nz_dim / 2.0)

    y, x, z = np.meshgrid(
        np.arange(ny_dim, dtype=DTYPE),
        np.arange(nx_dim, dtype=DTYPE),
        np.arange(nz_dim, dtype=DTYPE),
        indexing='ij',
    )

    # Distance from toron center
    dy = y - position[0]
    dx = x - position[1]
    dz = z - position[2]
    r_xy = np.sqrt(dx**2 + dy**2)
    r_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    # Background cholesteric twist
    q0 = 2 * np.pi / pitch
    theta_bg = q0 * z

    # Localized double-twist modulation
    # The twist angle increases toward the center
    toron_radius = pitch / 2
    modulation = np.exp(-(r_3d / toron_radius) ** 2)

    # Double twist: in-plane angle depends on azimuthal position
    phi_xy = np.arctan2(dy, dx)
    theta_local = theta_bg + modulation * phi_xy

    # Tilt from vertical at the toron core
    tilt = np.pi / 2 * (1 - modulation * 0.5)

    nx_arr = np.sin(tilt) * np.cos(theta_local)
    ny_arr = np.sin(tilt) * np.sin(theta_local)
    nz_arr = np.cos(tilt)

    # Normalize
    mag = np.sqrt(nx_arr**2 + ny_arr**2 + nz_arr**2)
    mag = np.where(mag > 1e-10, mag, 1.0)
    nx_arr /= mag
    ny_arr /= mag
    nz_arr /= mag

    return DirectorField(
        nx=nx_arr.astype(DTYPE),
        ny=ny_arr.astype(DTYPE),
        nz=nz_arr.astype(DTYPE),
        metadata={
            'type': 'toron',
            'pitch': pitch,
            'position': position,
        },
    )
