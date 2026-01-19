"""
Data Loaders for FCPM Simulation.

Functions for loading director fields and FCPM data from various formats.

Supported formats:
    - NumPy .npz (recommended for Python)
    - NumPy .npy (single array)
    - TIFF stacks (microscopy images)
    - MATLAB .mat files
    - Image sequences (PNG, JPG, etc.)

Auto-detection:
    Use load_director() or load_fcpm() for automatic format detection.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, Any, List
import json
import glob as glob_module
from ..core.director import DirectorField, DTYPE
from ..core.qtensor import QTensor


# =============================================================================
# Auto-detection loaders
# =============================================================================

def load_director(source: Union[str, Path, np.ndarray, Dict[str, np.ndarray]],
                  **kwargs) -> DirectorField:
    """
    Load director field with automatic format detection.

    This is the recommended entry point for loading director fields.
    Automatically detects the input type and calls the appropriate loader.

    Args:
        source: Can be:
            - Path to .npz file (with nx, ny, nz keys)
            - Path to .npy file (4D array with shape (..., 3))
            - Path to .mat file (MATLAB)
            - NumPy array (4D with shape (..., 3) or dict-like)
            - Dictionary with 'nx', 'ny', 'nz' keys
        **kwargs: Additional arguments passed to specific loaders.

    Returns:
        DirectorField loaded from source.

    Examples:
        >>> # From file (auto-detect format)
        >>> director = fcpm.load_director('director.npz')
        >>> director = fcpm.load_director('data/director.mat')

        >>> # From numpy array
        >>> director = fcpm.load_director(my_array)

        >>> # From dictionary
        >>> director = fcpm.load_director({'nx': nx, 'ny': ny, 'nz': nz})
    """
    # Handle numpy array input
    if isinstance(source, np.ndarray):
        if source.ndim == 4 and source.shape[-1] == 3:
            return DirectorField.from_array(source)
        else:
            raise ValueError(f"Array must be 4D with shape (..., 3), got {source.shape}")

    # Handle dictionary input
    if isinstance(source, dict):
        if 'nx' in source and 'ny' in source and 'nz' in source:
            return DirectorField(
                nx=source['nx'],
                ny=source['ny'],
                nz=source['nz'],
                metadata=source.get('metadata', {})
            )
        else:
            raise ValueError("Dictionary must contain 'nx', 'ny', 'nz' keys")

    # Handle file path
    filepath = Path(source)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.npz':
        return load_director_npz(filepath, **kwargs)
    elif suffix == '.npy':
        return load_director_npy(filepath, **kwargs)
    elif suffix == '.mat':
        return load_director_mat(filepath, **kwargs)
    elif suffix in ['.tif', '.tiff']:
        return load_director_tiff(filepath, **kwargs)
    else:
        # Try npz format as fallback
        try:
            return load_director_npz(filepath, **kwargs)
        except Exception as e:
            raise ValueError(f"Unknown file format: {suffix}. "
                           f"Supported: .npz, .npy, .mat, .tif. Error: {e}")


def load_fcpm(source: Union[str, Path, Dict[float, np.ndarray], np.ndarray],
              angles: Optional[List[float]] = None,
              **kwargs) -> Dict[float, np.ndarray]:
    """
    Load FCPM intensity data with automatic format detection.

    Args:
        source: Can be:
            - Path to .npz file
            - Path to .tif/.tiff file
            - Path to directory containing image files
            - Dictionary mapping angle -> intensity (returns as-is)
            - 4D numpy array (angles, y, x, z)
        angles: Polarization angles in radians (required for some formats).
                Default: [0, π/4, π/2, 3π/4]
        **kwargs: Additional arguments passed to specific loaders.

    Returns:
        Dictionary mapping angle (float) to intensity array.

    Examples:
        >>> I_fcpm = fcpm.load_fcpm('intensities.npz')
        >>> I_fcpm = fcpm.load_fcpm('data/fcpm_stack.tif', angles=[0, np.pi/4, np.pi/2])
        >>> I_fcpm = fcpm.load_fcpm('data/images/', angles=[0, np.pi/4])
    """
    # Default angles
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Handle dictionary input (pass through)
    if isinstance(source, dict):
        return {float(k): v.astype(DTYPE) for k, v in source.items()}

    # Handle numpy array input
    if isinstance(source, np.ndarray):
        if source.ndim != 4:
            raise ValueError(f"Array must be 4D (angles, y, x, z), got {source.ndim}D")
        if source.shape[0] != len(angles):
            raise ValueError(f"First dimension ({source.shape[0]}) must match "
                           f"number of angles ({len(angles)})")
        return {angle: source[i].astype(DTYPE) for i, angle in enumerate(angles)}

    # Handle file path
    filepath = Path(source)

    if filepath.is_dir():
        return load_fcpm_from_directory(filepath, angles=angles, **kwargs)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.npz':
        return load_fcpm_npz(filepath)
    elif suffix in ['.tif', '.tiff']:
        return load_fcpm_tiff_stack(filepath, angles=angles, **kwargs)
    elif suffix == '.mat':
        return load_fcpm_mat(filepath, angles=angles, **kwargs)
    else:
        try:
            return load_fcpm_npz(filepath)
        except Exception:
            raise ValueError(f"Unknown file format: {suffix}")


# =============================================================================
# MATLAB format support
# =============================================================================

def load_director_mat(filepath: Union[str, Path],
                      nx_key: str = 'nx',
                      ny_key: str = 'ny',
                      nz_key: str = 'nz') -> DirectorField:
    """
    Load director field from MATLAB .mat file.

    Args:
        filepath: Path to .mat file.
        nx_key, ny_key, nz_key: Variable names in the .mat file.

    Returns:
        DirectorField loaded from file.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy required for MATLAB file loading. "
                         "Install with: pip install scipy")

    filepath = Path(filepath)
    data = loadmat(str(filepath))

    # Try to find director components
    nx = None
    ny = None
    nz = None

    # Direct key lookup
    if nx_key in data:
        nx = data[nx_key]
    if ny_key in data:
        ny = data[ny_key]
    if nz_key in data:
        nz = data[nz_key]

    # Alternative: look for 'director' or 'n' array
    if nx is None:
        for key in ['director', 'n', 'Director', 'N']:
            if key in data:
                arr = data[key]
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    nx, ny, nz = arr[..., 0], arr[..., 1], arr[..., 2]
                    break

    if nx is None or ny is None or nz is None:
        available = [k for k in data.keys() if not k.startswith('_')]
        raise ValueError(f"Could not find director components. "
                        f"Available keys: {available}")

    return DirectorField(
        nx=nx.astype(DTYPE),
        ny=ny.astype(DTYPE),
        nz=nz.astype(DTYPE),
        metadata={'source_file': str(filepath), 'format': 'matlab'}
    )


def load_fcpm_mat(filepath: Union[str, Path],
                  angles: Optional[List[float]] = None,
                  intensity_key: Optional[str] = None) -> Dict[float, np.ndarray]:
    """
    Load FCPM intensities from MATLAB .mat file.

    Args:
        filepath: Path to .mat file.
        angles: Polarization angles in radians.
        intensity_key: Variable name for intensity data. If None, auto-detect.

    Returns:
        Dictionary mapping angle to intensity.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy required for MATLAB file loading.")

    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    filepath = Path(filepath)
    data = loadmat(str(filepath))

    # Try to find intensity data
    intensity_data = None

    if intensity_key and intensity_key in data:
        intensity_data = data[intensity_key]
    else:
        # Auto-detect: look for 4D array or dict-like structure
        for key in ['I', 'intensity', 'I_fcpm', 'fcpm', 'images']:
            if key in data:
                arr = data[key]
                if arr.ndim == 4:
                    intensity_data = arr
                    break

    if intensity_data is None:
        available = [k for k in data.keys() if not k.startswith('_')]
        raise ValueError(f"Could not find intensity data. Available: {available}")

    # Assume first dimension is angles
    if intensity_data.shape[0] != len(angles):
        raise ValueError(f"First dimension ({intensity_data.shape[0]}) must match "
                        f"number of angles ({len(angles)})")

    return {angle: intensity_data[i].astype(DTYPE) for i, angle in enumerate(angles)}


# =============================================================================
# TIFF format support
# =============================================================================

def load_director_tiff(filepath: Union[str, Path],
                       component_axis: int = 0) -> DirectorField:
    """
    Load director field from TIFF file.

    Expects a 4D TIFF with components stacked along one axis.

    Args:
        filepath: Path to TIFF file.
        component_axis: Axis along which nx, ny, nz are stacked.

    Returns:
        DirectorField loaded from file.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile required. Install with: pip install tifffile")

    filepath = Path(filepath)
    data = tifffile.imread(str(filepath))

    if data.shape[component_axis] != 3:
        raise ValueError(f"Expected 3 components along axis {component_axis}, "
                        f"got {data.shape[component_axis]}")

    # Move component axis to last position
    data = np.moveaxis(data, component_axis, -1)

    return DirectorField(
        nx=data[..., 0].astype(DTYPE),
        ny=data[..., 1].astype(DTYPE),
        nz=data[..., 2].astype(DTYPE),
        metadata={'source_file': str(filepath), 'format': 'tiff'}
    )


def load_fcpm_from_directory(dirpath: Union[str, Path],
                             angles: List[float],
                             pattern: str = '*.tif',
                             sort_key: Optional[callable] = None) -> Dict[float, np.ndarray]:
    """
    Load FCPM data from a directory of image files.

    Each file should contain a 3D stack for one polarization angle.

    Args:
        dirpath: Path to directory containing image files.
        angles: Polarization angles in radians.
        pattern: Glob pattern to match files.
        sort_key: Function to sort filenames. Default is alphabetical.

    Returns:
        Dictionary mapping angle to intensity.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile required for image loading.")

    dirpath = Path(dirpath)
    files = sorted(dirpath.glob(pattern), key=sort_key)

    if len(files) != len(angles):
        raise ValueError(f"Found {len(files)} files but {len(angles)} angles specified")

    I_fcpm = {}
    for filepath, angle in zip(files, angles):
        data = tifffile.imread(str(filepath))
        I_fcpm[angle] = data.astype(DTYPE)

    return I_fcpm


def load_fcpm_image_sequence(filepaths: List[Union[str, Path]],
                             angles: List[float]) -> Dict[float, np.ndarray]:
    """
    Load FCPM data from a list of image files.

    Args:
        filepaths: List of paths to image files.
        angles: Corresponding polarization angles.

    Returns:
        Dictionary mapping angle to intensity.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile required for image loading.")

    if len(filepaths) != len(angles):
        raise ValueError(f"Number of files ({len(filepaths)}) must match "
                        f"number of angles ({len(angles)})")

    I_fcpm = {}
    for filepath, angle in zip(filepaths, angles):
        data = tifffile.imread(str(filepath))
        I_fcpm[angle] = data.astype(DTYPE)

    return I_fcpm


# =============================================================================
# Original loaders (kept for backwards compatibility)
# =============================================================================


def load_director_npz(filepath: Union[str, Path]) -> DirectorField:
    """
    Load director field from NumPy .npz file.

    Supports multiple formats:
    1. Keys 'nx', 'ny', 'nz' (standard format)
    2. Key 'PATH' with shape (..., 3) (experimental format)
    3. Key 'director' or 'n' with shape (..., 3)

    Args:
        filepath: Path to .npz file.

    Returns:
        DirectorField loaded from file.
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    metadata = {}
    if 'metadata' in data:
        md = data['metadata']
        metadata = md.item() if hasattr(md, 'item') else dict(md)

    metadata['source_file'] = str(filepath)

    # Format 1: Standard nx, ny, nz keys
    if 'nx' in data and 'ny' in data and 'nz' in data:
        return DirectorField(
            nx=data['nx'],
            ny=data['ny'],
            nz=data['nz'],
            metadata=metadata
        )

    # Format 2: Experimental 'PATH' format (from simulations)
    if 'PATH' in data:
        path_data = data['PATH']
        # Squeeze out singleton dimensions
        path_squeezed = np.squeeze(path_data)

        if path_squeezed.ndim == 4 and path_squeezed.shape[-1] == 3:
            # Shape is (y, x, z, 3)
            metadata['format'] = 'experimental_PATH'
            if 'settings' in data:
                settings_str = str(data['settings'])
                metadata['settings'] = settings_str
            return DirectorField(
                nx=path_squeezed[..., 0].astype(DTYPE),
                ny=path_squeezed[..., 1].astype(DTYPE),
                nz=path_squeezed[..., 2].astype(DTYPE),
                metadata=metadata
            )

    # Format 3: Generic 'director' or 'n' array
    for key in ['director', 'n', 'Director', 'N']:
        if key in data:
            arr = np.squeeze(data[key])
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                metadata['format'] = f'generic_{key}'
                return DirectorField(
                    nx=arr[..., 0].astype(DTYPE),
                    ny=arr[..., 1].astype(DTYPE),
                    nz=arr[..., 2].astype(DTYPE),
                    metadata=metadata
                )

    # If we get here, couldn't parse the file
    available = [k for k in data.keys() if not k.startswith('_')]
    raise ValueError(f"Could not find director data in file. "
                    f"Available keys: {available}. "
                    f"Expected 'nx'/'ny'/'nz', 'PATH', or 'director'.")


def load_director_npy(filepath: Union[str, Path],
                      component_order: str = 'xyz') -> DirectorField:
    """
    Load director field from single .npy file.

    The file should contain a 4D array with shape (..., 3).

    Args:
        filepath: Path to .npy file.
        component_order: Order of components in last axis ('xyz' or 'zyx').

    Returns:
        DirectorField loaded from file.
    """
    filepath = Path(filepath)
    data = np.load(filepath)

    if data.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3, got {data.shape[-1]}")

    if component_order == 'xyz':
        return DirectorField(
            nx=data[..., 0],
            ny=data[..., 1],
            nz=data[..., 2],
            metadata={'source_file': str(filepath)}
        )
    elif component_order == 'zyx':
        return DirectorField(
            nx=data[..., 2],
            ny=data[..., 1],
            nz=data[..., 0],
            metadata={'source_file': str(filepath)}
        )
    else:
        raise ValueError(f"Unknown component_order: {component_order}")


def load_fcpm_npz(filepath: Union[str, Path]) -> Dict[float, np.ndarray]:
    """
    Load FCPM intensity data from .npz file.

    Expected format: keys are angle values (as strings), values are intensity arrays.

    Args:
        filepath: Path to .npz file.

    Returns:
        Dictionary mapping angle (radians) to intensity array.
    """
    filepath = Path(filepath)
    data = np.load(filepath)

    I_fcpm = {}
    for key in data.keys():
        try:
            angle = float(key)
            I_fcpm[angle] = data[key].astype(DTYPE)
        except ValueError:
            # Skip non-numeric keys (might be metadata)
            continue

    if not I_fcpm:
        raise ValueError(f"No valid angle data found in {filepath}")

    return I_fcpm


def load_fcpm_tiff_stack(filepath: Union[str, Path],
                         angles: list,
                         axis: int = 0) -> Dict[float, np.ndarray]:
    """
    Load FCPM data from TIFF stack.

    Args:
        filepath: Path to TIFF file.
        angles: List of polarization angles in radians.
        axis: Axis along which angles are stacked.

    Returns:
        Dictionary mapping angle to intensity.

    Note:
        Requires tifffile package.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile package required. Install with: pip install tifffile")

    filepath = Path(filepath)
    data = tifffile.imread(str(filepath))

    if data.shape[axis] != len(angles):
        raise ValueError(f"Number of images ({data.shape[axis]}) doesn't match "
                        f"number of angles ({len(angles)})")

    I_fcpm = {}
    for i, angle in enumerate(angles):
        slices = [slice(None)] * data.ndim
        slices[axis] = i
        I_fcpm[angle] = data[tuple(slices)].astype(DTYPE)

    return I_fcpm


def load_qtensor_npz(filepath: Union[str, Path]) -> QTensor:
    """
    Load Q-tensor field from .npz file.

    Args:
        filepath: Path to .npz file.

    Returns:
        QTensor loaded from file.
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    required = ['Q_xx', 'Q_yy', 'Q_xy', 'Q_xz', 'Q_yz']
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    metadata = {}
    if 'metadata' in data:
        metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else dict(data['metadata'])

    metadata['source_file'] = str(filepath)

    return QTensor(
        Q_xx=data['Q_xx'],
        Q_yy=data['Q_yy'],
        Q_xy=data['Q_xy'],
        Q_xz=data['Q_xz'],
        Q_yz=data['Q_yz'],
        metadata=metadata
    )


def load_simulation_results(dirpath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load complete simulation results from a directory.

    Expected structure:
        dirpath/
            director.npz
            fcpm_intensities.npz
            qtensor.npz (optional)
            metadata.json (optional)

    Args:
        dirpath: Path to results directory.

    Returns:
        Dictionary with loaded data.
    """
    dirpath = Path(dirpath)

    results = {}

    # Load director
    director_path = dirpath / 'director.npz'
    if director_path.exists():
        results['director'] = load_director_npz(director_path)

    # Load FCPM
    fcpm_path = dirpath / 'fcpm_intensities.npz'
    if fcpm_path.exists():
        results['fcpm'] = load_fcpm_npz(fcpm_path)

    # Load Q-tensor
    qtensor_path = dirpath / 'qtensor.npz'
    if qtensor_path.exists():
        results['qtensor'] = load_qtensor_npz(qtensor_path)

    # Load metadata
    metadata_path = dirpath / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)

    return results
