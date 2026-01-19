"""
Cropping and Region of Interest (ROI) utilities for FCPM data.

Functions for extracting subregions from director fields and FCPM data.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from ..core.director import DirectorField, DTYPE
from ..core.qtensor import QTensor


def crop_director(director: DirectorField,
                  y_range: Optional[Tuple[int, int]] = None,
                  x_range: Optional[Tuple[int, int]] = None,
                  z_range: Optional[Tuple[int, int]] = None) -> DirectorField:
    """
    Crop a director field to a specified region.

    Args:
        director: Input director field.
        y_range: (start, end) indices for y-axis. None means full range.
        x_range: (start, end) indices for x-axis.
        z_range: (start, end) indices for z-axis.

    Returns:
        Cropped DirectorField.

    Example:
        >>> cropped = crop_director(director, y_range=(10, 50), x_range=(10, 50))
    """
    shape = director.shape

    # Default to full range if not specified
    y_start, y_end = y_range if y_range else (0, shape[0])
    x_start, x_end = x_range if x_range else (0, shape[1])
    z_start, z_end = z_range if z_range else (0, shape[2])

    # Validate ranges
    y_start = max(0, y_start)
    y_end = min(shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(shape[1], x_end)
    z_start = max(0, z_start)
    z_end = min(shape[2], z_end)

    metadata = director.metadata.copy()
    metadata['cropped'] = True
    metadata['original_shape'] = shape
    metadata['crop_region'] = {
        'y': (y_start, y_end),
        'x': (x_start, x_end),
        'z': (z_start, z_end)
    }

    return DirectorField(
        nx=director.nx[y_start:y_end, x_start:x_end, z_start:z_end].copy(),
        ny=director.ny[y_start:y_end, x_start:x_end, z_start:z_end].copy(),
        nz=director.nz[y_start:y_end, x_start:x_end, z_start:z_end].copy(),
        metadata=metadata
    )


def crop_director_center(director: DirectorField,
                         size: Union[int, Tuple[int, int, int]]) -> DirectorField:
    """
    Crop a director field to a centered region of specified size.

    Args:
        director: Input director field.
        size: Size of cropped region. If int, same size for all dimensions.
              If tuple, (ny, nx, nz) sizes.

    Returns:
        Center-cropped DirectorField.
    """
    shape = director.shape

    if isinstance(size, int):
        size = (size, size, size)

    # Calculate center crop ranges
    y_size = min(size[0], shape[0])
    x_size = min(size[1], shape[1])
    z_size = min(size[2], shape[2])

    y_start = (shape[0] - y_size) // 2
    x_start = (shape[1] - x_size) // 2
    z_start = (shape[2] - z_size) // 2

    return crop_director(
        director,
        y_range=(y_start, y_start + y_size),
        x_range=(x_start, x_start + x_size),
        z_range=(z_start, z_start + z_size)
    )


def crop_fcpm(I_fcpm: Dict[float, np.ndarray],
              y_range: Optional[Tuple[int, int]] = None,
              x_range: Optional[Tuple[int, int]] = None,
              z_range: Optional[Tuple[int, int]] = None) -> Dict[float, np.ndarray]:
    """
    Crop FCPM intensity data to a specified region.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        y_range, x_range, z_range: Crop ranges for each axis.

    Returns:
        Cropped FCPM intensity dictionary.
    """
    # Get shape from first image
    first_key = next(iter(I_fcpm))
    shape = I_fcpm[first_key].shape

    # Default to full range
    y_start, y_end = y_range if y_range else (0, shape[0])
    x_start, x_end = x_range if x_range else (0, shape[1])
    z_start, z_end = z_range if z_range else (0, shape[2])

    # Validate
    y_start, y_end = max(0, y_start), min(shape[0], y_end)
    x_start, x_end = max(0, x_start), min(shape[1], x_end)
    z_start, z_end = max(0, z_start), min(shape[2], z_end)

    I_cropped = {}
    for angle, intensity in I_fcpm.items():
        I_cropped[angle] = intensity[y_start:y_end, x_start:x_end, z_start:z_end].copy()

    return I_cropped


def crop_fcpm_center(I_fcpm: Dict[float, np.ndarray],
                     size: Union[int, Tuple[int, int, int]]) -> Dict[float, np.ndarray]:
    """
    Crop FCPM data to a centered region.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        size: Size of cropped region.

    Returns:
        Center-cropped FCPM dictionary.
    """
    first_key = next(iter(I_fcpm))
    shape = I_fcpm[first_key].shape

    if isinstance(size, int):
        size = (size, size, size)

    y_size = min(size[0], shape[0])
    x_size = min(size[1], shape[1])
    z_size = min(size[2], shape[2])

    y_start = (shape[0] - y_size) // 2
    x_start = (shape[1] - x_size) // 2
    z_start = (shape[2] - z_size) // 2

    return crop_fcpm(
        I_fcpm,
        y_range=(y_start, y_start + y_size),
        x_range=(x_start, x_start + x_size),
        z_range=(z_start, z_start + z_size)
    )


def crop_qtensor(Q: QTensor,
                 y_range: Optional[Tuple[int, int]] = None,
                 x_range: Optional[Tuple[int, int]] = None,
                 z_range: Optional[Tuple[int, int]] = None) -> QTensor:
    """
    Crop a Q-tensor field to a specified region.

    Args:
        Q: Input Q-tensor field.
        y_range, x_range, z_range: Crop ranges.

    Returns:
        Cropped QTensor.
    """
    shape = Q.shape

    y_start, y_end = y_range if y_range else (0, shape[0])
    x_start, x_end = x_range if x_range else (0, shape[1])
    z_start, z_end = z_range if z_range else (0, shape[2])

    y_start, y_end = max(0, y_start), min(shape[0], y_end)
    x_start, x_end = max(0, x_start), min(shape[1], x_end)
    z_start, z_end = max(0, z_start), min(shape[2], z_end)

    slc = (slice(y_start, y_end), slice(x_start, x_end), slice(z_start, z_end))

    metadata = Q.metadata.copy()
    metadata['cropped'] = True
    metadata['original_shape'] = shape

    return QTensor(
        Q_xx=Q.Q_xx[slc].copy(),
        Q_yy=Q.Q_yy[slc].copy(),
        Q_xy=Q.Q_xy[slc].copy(),
        Q_xz=Q.Q_xz[slc].copy(),
        Q_yz=Q.Q_yz[slc].copy(),
        metadata=metadata
    )


def pad_director(director: DirectorField,
                 pad_width: Union[int, Tuple[Tuple[int, int], ...]],
                 mode: str = 'edge') -> DirectorField:
    """
    Pad a director field.

    Args:
        director: Input director field.
        pad_width: Padding width. Int for uniform padding, or
                   ((y_before, y_after), (x_before, x_after), (z_before, z_after)).
        mode: Padding mode ('edge', 'constant', 'reflect', 'wrap').

    Returns:
        Padded DirectorField.
    """
    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width),) * 3

    nx_padded = np.pad(director.nx, pad_width, mode=mode)
    ny_padded = np.pad(director.ny, pad_width, mode=mode)
    nz_padded = np.pad(director.nz, pad_width, mode=mode)

    metadata = director.metadata.copy()
    metadata['padded'] = True
    metadata['pad_width'] = pad_width

    result = DirectorField(nx=nx_padded, ny=ny_padded, nz=nz_padded, metadata=metadata)

    # Renormalize after padding (especially important for 'constant' mode)
    return result.normalize()


def extract_slice(director: DirectorField,
                  axis: str,
                  index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2D slice from a director field.

    Args:
        director: Input director field.
        axis: Axis perpendicular to slice ('x', 'y', or 'z').
        index: Slice index.

    Returns:
        Tuple of (nx, ny, nz) 2D arrays for the slice.
    """
    if axis == 'z':
        return director.slice_z(index)
    elif axis == 'y':
        return (director.nx[index, :, :],
                director.ny[index, :, :],
                director.nz[index, :, :])
    elif axis == 'x':
        return (director.nx[:, index, :],
                director.ny[:, index, :],
                director.nz[:, index, :])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")


def subsample_director(director: DirectorField,
                       factor: Union[int, Tuple[int, int, int]]) -> DirectorField:
    """
    Subsample a director field by a given factor.

    Args:
        director: Input director field.
        factor: Subsampling factor. Int for uniform, tuple for per-axis.

    Returns:
        Subsampled DirectorField.
    """
    if isinstance(factor, int):
        factor = (factor, factor, factor)

    slc = (slice(None, None, factor[0]),
           slice(None, None, factor[1]),
           slice(None, None, factor[2]))

    metadata = director.metadata.copy()
    metadata['subsampled'] = True
    metadata['subsample_factor'] = factor

    return DirectorField(
        nx=director.nx[slc].copy(),
        ny=director.ny[slc].copy(),
        nz=director.nz[slc].copy(),
        metadata=metadata
    )


def subsample_fcpm(I_fcpm: Dict[float, np.ndarray],
                   factor: Union[int, Tuple[int, int, int]]) -> Dict[float, np.ndarray]:
    """
    Subsample FCPM data by a given factor.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        factor: Subsampling factor.

    Returns:
        Subsampled FCPM dictionary.
    """
    if isinstance(factor, int):
        factor = (factor, factor, factor)

    slc = (slice(None, None, factor[0]),
           slice(None, None, factor[1]),
           slice(None, None, factor[2]))

    return {angle: intensity[slc].copy() for angle, intensity in I_fcpm.items()}
