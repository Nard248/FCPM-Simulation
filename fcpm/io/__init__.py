"""
FCPM I/O Module

Functions for loading and saving director fields, FCPM data, and Q-tensors.

Supported formats:
    - NumPy .npz (recommended for Python workflows)
    - NumPy .npy (single array)
    - TIFF (for image data)
    - MATLAB .mat (for MATLAB interoperability)
    - VTK (for ParaView/VisIt visualization)

Auto-detection:
    Use load_director() and load_fcpm() for automatic format detection.
"""

from .loaders import (
    # Auto-detection loaders (recommended)
    load_director,
    load_fcpm,
    # Format-specific loaders
    load_director_npz,
    load_director_npy,
    load_director_mat,
    load_director_tiff,
    load_director_hdf5,
    load_fcpm_npz,
    load_fcpm_tiff_stack,
    load_fcpm_mat,
    load_fcpm_from_directory,
    load_fcpm_image_sequence,
    load_qtensor_npz,
    load_simulation_results,
    load_lcsim_npz,
)

from .exporters import (
    save_director_npz,
    save_director_npy,
    save_director_hdf5,
    save_fcpm_npz,
    save_fcpm_tiff,
    save_qtensor_npz,
    save_simulation_results,
    export_for_matlab,
    export_for_vtk,
)

__all__ = [
    # Auto-detection loaders (recommended)
    'load_director',
    'load_fcpm',
    # Format-specific loaders
    'load_director_npz',
    'load_director_npy',
    'load_director_mat',
    'load_director_tiff',
    'load_director_hdf5',
    'load_fcpm_npz',
    'load_fcpm_tiff_stack',
    'load_fcpm_mat',
    'load_fcpm_from_directory',
    'load_fcpm_image_sequence',
    'load_qtensor_npz',
    'load_simulation_results',
    'load_lcsim_npz',
    # Exporters
    'save_director_npz',
    'save_director_npy',
    'save_director_hdf5',
    'save_fcpm_npz',
    'save_fcpm_tiff',
    'save_qtensor_npz',
    'save_simulation_results',
    'export_for_matlab',
    'export_for_vtk',
]
