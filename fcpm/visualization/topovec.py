"""
TopoVec Integration for Director Field Visualization.

TopoVec is a specialized visualization tool for liquid crystal director fields.
This module provides integration with TopoVec if available, or fallback
matplotlib-based 3D visualization.

Note: TopoVec must be installed separately. This module provides a wrapper
interface that works with or without TopoVec installed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import warnings
from ..core.director import DirectorField

# Try to import TopoVec
try:
    from topovec import topovec_field
    TOPOVEC_AVAILABLE = True
except ImportError:
    TOPOVEC_AVAILABLE = False


def check_topovec_available() -> bool:
    """Check if TopoVec is installed and available."""
    return TOPOVEC_AVAILABLE


def visualize_topovec(director: DirectorField,
                      color_by: str = 'nz',
                      scale: float = 1.0,
                      subsample: int = 1,
                      title: Optional[str] = None,
                      **kwargs) -> Any:
    """
    Visualize director field using TopoVec.

    Args:
        director: Director field to visualize.
        color_by: Color coding ('nz', 'angle', 'uniform').
        scale: Glyph scale factor.
        subsample: Subsampling factor.
        title: Visualization title.
        **kwargs: Additional TopoVec parameters.

    Returns:
        TopoVec visualization object.

    Raises:
        ImportError: If TopoVec is not installed.
    """
    if not TOPOVEC_AVAILABLE:
        raise ImportError(
            "TopoVec is not installed. Install with: pip install topovec\n"
            "Or use the matplotlib-based visualization functions instead."
        )

    # Prepare data for TopoVec
    n = director.to_array()

    # Subsample if requested
    if subsample > 1:
        n = n[::subsample, ::subsample, ::subsample]

    # Create coordinate arrays
    shape = n.shape[:3]
    coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    coords = coords.transpose(1, 2, 3, 0).reshape(-1, 3)

    # Flatten director data
    n_flat = n.reshape(-1, 3)

    # Color data
    if color_by == 'nz':
        colors = np.abs(n_flat[:, 2])
    elif color_by == 'angle':
        colors = np.arctan2(n_flat[:, 1], n_flat[:, 0]) % np.pi
    else:
        colors = np.ones(len(n_flat))

    # Call TopoVec
    return topovec_field(coords, n_flat, colors=colors, scale=scale,
                         title=title or "Director Field", **kwargs)


def prepare_topovec_data(director: DirectorField,
                         subsample: int = 1,
                         normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Prepare director field data for TopoVec visualization.

    Returns arrays in the format expected by TopoVec.

    Args:
        director: Director field.
        subsample: Subsampling factor.
        normalize: Ensure unit length.

    Returns:
        Dictionary with 'positions', 'directions', 'nz' arrays.
    """
    n = director.to_array()

    if subsample > 1:
        n = n[::subsample, ::subsample, ::subsample]

    if normalize:
        mag = np.linalg.norm(n, axis=-1, keepdims=True)
        mag = np.where(mag > 1e-10, mag, 1.0)
        n = n / mag

    shape = n.shape[:3]

    # Create position grid
    y, x, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    positions = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    # Flatten directions
    directions = n.reshape(-1, 3).astype(np.float32)

    # nz values for coloring
    nz = np.abs(n[..., 2]).reshape(-1).astype(np.float32)

    return {
        'positions': positions,
        'directions': directions,
        'nz': nz,
        'shape': shape,
    }


def visualize_3d_matplotlib(director: DirectorField,
                            subsample: int = 2,
                            color_by: str = 'nz',
                            figsize: Tuple[float, float] = (10, 8),
                            elevation: float = 20,
                            azimuth: float = 45) -> Any:
    """
    3D visualization using matplotlib (fallback when TopoVec unavailable).

    Args:
        director: Director field.
        subsample: Subsampling factor.
        color_by: Color coding.
        figsize: Figure size.
        elevation: Viewing elevation angle.
        azimuth: Viewing azimuth angle.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n = director.to_array()

    if subsample > 1:
        n = n[::subsample, ::subsample, ::subsample]

    shape = n.shape[:3]
    y, x, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Flatten
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    nx_flat = n[..., 0].flatten()
    ny_flat = n[..., 1].flatten()
    nz_flat = n[..., 2].flatten()

    # Colors
    if color_by == 'nz':
        colors = np.abs(nz_flat)
    elif color_by == 'angle':
        colors = np.arctan2(ny_flat, nx_flat) % np.pi
        colors = colors / np.pi
    else:
        colors = np.ones_like(nz_flat)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Use quiver for 3D arrows
    ax.quiver(x_flat, y_flat, z_flat,
              nx_flat, ny_flat, nz_flat,
              length=0.5 * subsample,
              normalize=True,
              cmap='viridis',
              alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elevation, azim=azimuth)

    return fig


def export_for_paraview(director: DirectorField,
                        filename: Union[str, Path],
                        format: str = 'vtk') -> Path:
    """
    Export director field for visualization in ParaView.

    Args:
        director: Director field to export.
        filename: Output filename (without extension).
        format: Output format ('vtk' or 'vtu').

    Returns:
        Path to saved file.

    Note:
        Requires the 'vtk' package for VTK format export.
    """
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        raise ImportError("VTK package required for ParaView export. "
                         "Install with: pip install vtk")

    filepath = Path(filename)
    n = director.to_array()
    shape = director.shape

    # Create structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(shape[1], shape[0], shape[2])

    # Create points
    points = vtk.vtkPoints()
    for iz in range(shape[2]):
        for iy in range(shape[0]):
            for ix in range(shape[1]):
                points.InsertNextPoint(ix, iy, iz)
    grid.SetPoints(points)

    # Add director as vector field
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("director")

    for iz in range(shape[2]):
        for iy in range(shape[0]):
            for ix in range(shape[1]):
                vectors.InsertNextTuple3(n[iy, ix, iz, 0],
                                        n[iy, ix, iz, 1],
                                        n[iy, ix, iz, 2])
    grid.GetPointData().SetVectors(vectors)

    # Write file
    if format == 'vtu':
        writer = vtk.vtkXMLStructuredGridWriter()
        filepath = filepath.with_suffix('.vts')
    else:
        writer = vtk.vtkStructuredGridWriter()
        filepath = filepath.with_suffix('.vtk')

    writer.SetFileName(str(filepath))
    writer.SetInputData(grid)
    writer.Write()

    return filepath
