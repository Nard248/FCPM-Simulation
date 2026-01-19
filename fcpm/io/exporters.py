"""
Data Exporters for FCPM Simulation.

Functions for saving director fields and FCPM data to various formats.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Any
import json
from ..core.director import DirectorField
from ..core.qtensor import QTensor


def save_director_npz(director: DirectorField,
                      filepath: Union[str, Path],
                      compressed: bool = True) -> Path:
    """
    Save director field to NumPy .npz file.

    Args:
        director: Director field to save.
        filepath: Output path.
        compressed: Use compression.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath).with_suffix('.npz')

    save_func = np.savez_compressed if compressed else np.savez
    save_func(
        filepath,
        nx=director.nx,
        ny=director.ny,
        nz=director.nz,
        metadata=director.metadata
    )

    return filepath


def save_director_npy(director: DirectorField,
                      filepath: Union[str, Path]) -> Path:
    """
    Save director field to single .npy file.

    Output shape: (ny, nx, nz, 3) with order [nx, ny, nz] in last axis.

    Args:
        director: Director field to save.
        filepath: Output path.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath).with_suffix('.npy')
    data = director.to_array()
    np.save(filepath, data)
    return filepath


def save_fcpm_npz(I_fcpm: Dict[float, np.ndarray],
                  filepath: Union[str, Path],
                  compressed: bool = True) -> Path:
    """
    Save FCPM intensity data to .npz file.

    Args:
        I_fcpm: Dictionary mapping angle to intensity.
        filepath: Output path.
        compressed: Use compression.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath).with_suffix('.npz')

    # Convert angle keys to strings for numpy
    data_dict = {str(angle): intensity for angle, intensity in I_fcpm.items()}

    save_func = np.savez_compressed if compressed else np.savez
    save_func(filepath, **data_dict)

    return filepath


def save_fcpm_tiff(I_fcpm: Dict[float, np.ndarray],
                   filepath: Union[str, Path],
                   z_idx: Optional[int] = None) -> Path:
    """
    Save FCPM intensity data to TIFF file(s).

    Args:
        I_fcpm: Dictionary mapping angle to intensity.
        filepath: Output path (base name).
        z_idx: Specific z-slice to save (None = all slices).

    Returns:
        Path to saved file(s).

    Note:
        Requires tifffile package.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile package required. Install with: pip install tifffile")

    filepath = Path(filepath)

    angles = sorted(I_fcpm.keys())

    if z_idx is not None:
        # Save single slice
        stack = np.stack([I_fcpm[a][:, :, z_idx] for a in angles], axis=0)
        out_path = filepath.with_name(f"{filepath.stem}_z{z_idx}.tiff")
        tifffile.imwrite(str(out_path), stack.astype(np.float32))
        return out_path
    else:
        # Save full 3D stack
        stack = np.stack([I_fcpm[a] for a in angles], axis=0)
        out_path = filepath.with_suffix('.tiff')
        tifffile.imwrite(str(out_path), stack.astype(np.float32))
        return out_path


def save_qtensor_npz(Q: QTensor,
                     filepath: Union[str, Path],
                     compressed: bool = True) -> Path:
    """
    Save Q-tensor field to .npz file.

    Args:
        Q: Q-tensor to save.
        filepath: Output path.
        compressed: Use compression.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath).with_suffix('.npz')

    save_func = np.savez_compressed if compressed else np.savez
    save_func(
        filepath,
        Q_xx=Q.Q_xx,
        Q_yy=Q.Q_yy,
        Q_xy=Q.Q_xy,
        Q_xz=Q.Q_xz,
        Q_yz=Q.Q_yz,
        metadata=Q.metadata
    )

    return filepath


def save_simulation_results(results: Dict[str, Any],
                            dirpath: Union[str, Path],
                            overwrite: bool = False) -> Path:
    """
    Save complete simulation results to a directory.

    Args:
        results: Dictionary with 'director', 'fcpm', 'qtensor', 'metadata'.
        dirpath: Output directory.
        overwrite: Overwrite existing files.

    Returns:
        Path to output directory.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    if 'director' in results:
        out_path = dirpath / 'director.npz'
        if overwrite or not out_path.exists():
            save_director_npz(results['director'], out_path)

    if 'fcpm' in results:
        out_path = dirpath / 'fcpm_intensities.npz'
        if overwrite or not out_path.exists():
            save_fcpm_npz(results['fcpm'], out_path)

    if 'qtensor' in results:
        out_path = dirpath / 'qtensor.npz'
        if overwrite or not out_path.exists():
            save_qtensor_npz(results['qtensor'], out_path)

    if 'metadata' in results:
        out_path = dirpath / 'metadata.json'
        if overwrite or not out_path.exists():
            with open(out_path, 'w') as f:
                json.dump(results['metadata'], f, indent=2)

    return dirpath


def export_for_matlab(director: DirectorField,
                      filepath: Union[str, Path]) -> Path:
    """
    Export director field for MATLAB.

    Args:
        director: Director field to export.
        filepath: Output path.

    Returns:
        Path to saved file.

    Note:
        Requires scipy package.
    """
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy package required. Install with: pip install scipy")

    filepath = Path(filepath).with_suffix('.mat')

    data = {
        'nx': director.nx,
        'ny': director.ny,
        'nz': director.nz,
        'shape': np.array(director.shape),
    }

    savemat(str(filepath), data)
    return filepath


def export_for_vtk(director: DirectorField,
                   filepath: Union[str, Path]) -> Path:
    """
    Export director field as VTK file for ParaView/VisIt.

    Args:
        director: Director field to export.
        filepath: Output path.

    Returns:
        Path to saved file.

    Note:
        Creates a structured points dataset with vector field.
    """
    filepath = Path(filepath).with_suffix('.vtk')

    shape = director.shape
    n = director.to_array()

    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Director field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {shape[1]} {shape[0]} {shape[2]}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {shape[0] * shape[1] * shape[2]}\n")
        f.write("VECTORS director float\n")

        for iz in range(shape[2]):
            for iy in range(shape[0]):
                for ix in range(shape[1]):
                    f.write(f"{n[iy, ix, iz, 0]:.6f} "
                           f"{n[iy, ix, iz, 1]:.6f} "
                           f"{n[iy, ix, iz, 2]:.6f}\n")

    return filepath
