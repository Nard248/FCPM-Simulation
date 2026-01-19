"""
Director Field Visualization.

Functions for visualizing director fields in 2D slices and 3D volumes.
Uses matplotlib for 2D plots and optional mayavi/plotly for 3D.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Union, Dict, Any
from ..core.director import DirectorField


def plot_director_slice(director: DirectorField,
                        z_idx: int,
                        step: int = 1,
                        scale: float = 0.8,
                        color_by: str = 'angle',
                        cmap: str = 'hsv',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (10, 8),
                        title: Optional[str] = None,
                        show_colorbar: bool = True) -> plt.Figure:
    """
    Plot a 2D slice of the director field as arrows.

    Args:
        director: Director field to visualize.
        z_idx: Index of z-slice to plot.
        step: Subsampling step (1 = all points, 2 = every other, etc.).
        scale: Arrow length scale factor.
        color_by: Color arrows by 'angle' (in-plane), 'nz' (out-of-plane),
                  'magnitude', or None (uniform color).
        cmap: Colormap name.
        ax: Matplotlib axes (created if None).
        figsize: Figure size if creating new figure.
        title: Plot title.
        show_colorbar: Whether to show colorbar.

    Returns:
        Matplotlib figure.
    """
    nx_slice, ny_slice, nz_slice = director.slice_z(z_idx)

    # Subsample
    nx_sub = nx_slice[::step, ::step]
    ny_sub = ny_slice[::step, ::step]
    nz_sub = nz_slice[::step, ::step]

    # Create coordinate grids
    Y, X = np.mgrid[0:nx_sub.shape[0], 0:nx_sub.shape[1]]
    X = X * step
    Y = Y * step

    # Determine colors
    if color_by == 'angle':
        # In-plane angle (0 to π due to nematic symmetry)
        angles = np.arctan2(ny_sub, nx_sub) % np.pi
        colors = angles / np.pi
        clabel = 'In-plane angle (rad/π)'
    elif color_by == 'nz':
        colors = np.abs(nz_sub)
        clabel = '|nz|'
    elif color_by == 'magnitude':
        colors = np.sqrt(nx_sub**2 + ny_sub**2)
        clabel = 'In-plane magnitude'
    else:
        colors = None
        clabel = None

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot quiver
    if colors is not None:
        q = ax.quiver(X, Y, nx_sub, ny_sub,
                      colors, cmap=cmap,
                      scale=1/scale, scale_units='xy',
                      headwidth=3, headlength=4,
                      pivot='middle')
        if show_colorbar:
            cbar = plt.colorbar(q, ax=ax, label=clabel)
    else:
        ax.quiver(X, Y, nx_sub, ny_sub,
                  scale=1/scale, scale_units='xy',
                  headwidth=3, headlength=4,
                  pivot='middle')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Director field at z={z_idx}')

    return fig


def plot_director_streamlines(director: DirectorField,
                               z_idx: int,
                               density: float = 1.0,
                               color_by: str = 'angle',
                               cmap: str = 'twilight',
                               ax: Optional[plt.Axes] = None,
                               figsize: Tuple[float, float] = (10, 8),
                               title: Optional[str] = None) -> plt.Figure:
    """
    Plot director field as streamlines (integral curves).

    Note: Due to nematic symmetry, streamlines may have discontinuities.

    Args:
        director: Director field to visualize.
        z_idx: Index of z-slice.
        density: Streamline density.
        color_by: Color by 'angle', 'nz', or 'speed'.
        cmap: Colormap.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    nx_slice, ny_slice, nz_slice = director.slice_z(z_idx)

    Y, X = np.mgrid[0:nx_slice.shape[0], 0:nx_slice.shape[1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if color_by == 'angle':
        colors = np.arctan2(ny_slice, nx_slice) % np.pi
    elif color_by == 'nz':
        colors = np.abs(nz_slice)
    else:
        colors = np.sqrt(nx_slice**2 + ny_slice**2)

    strm = ax.streamplot(X, Y, nx_slice, ny_slice,
                         color=colors, cmap=cmap,
                         density=density, linewidth=1)
    plt.colorbar(strm.lines, ax=ax)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or f'Director streamlines at z={z_idx}')

    return fig


def plot_director_rgb(director: DirectorField,
                      z_idx: int,
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[float, float] = (10, 8),
                      title: Optional[str] = None) -> plt.Figure:
    """
    Plot director orientation as RGB color image.

    Color mapping:
    - Hue: in-plane angle (0 to π mapped to 0 to 1)
    - Saturation: in-plane magnitude
    - Value: 1 - |nz| (bright = in-plane, dark = out-of-plane)

    Args:
        director: Director field.
        z_idx: Z-slice index.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    nx_slice, ny_slice, nz_slice = director.slice_z(z_idx)

    # Hue from in-plane angle
    angle = np.arctan2(ny_slice, nx_slice) % np.pi
    hue = angle / np.pi

    # Saturation from in-plane magnitude
    in_plane_mag = np.sqrt(nx_slice**2 + ny_slice**2)
    saturation = np.clip(in_plane_mag, 0, 1)

    # Value from nz
    value = 1.0 - np.abs(nz_slice) * 0.5  # Keep some brightness

    # Convert HSV to RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(rgb, origin='lower')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or f'Director RGB at z={z_idx}')

    return fig


def plot_fcpm_intensities(I_fcpm: Dict[float, np.ndarray],
                          z_idx: int,
                          figsize: Tuple[float, float] = (12, 3),
                          cmap: str = 'gray') -> plt.Figure:
    """
    Plot FCPM intensity images for all angles.

    Args:
        I_fcpm: Dictionary of FCPM intensities.
        z_idx: Z-slice index.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Matplotlib figure.
    """
    angles = sorted(I_fcpm.keys())
    n_angles = len(angles)

    fig, axes = plt.subplots(1, n_angles, figsize=figsize)
    if n_angles == 1:
        axes = [axes]

    for ax, angle in zip(axes, angles):
        I = I_fcpm[angle][:, :, z_idx]
        im = ax.imshow(I, cmap=cmap, origin='lower')
        ax.set_title(f'α = {np.degrees(angle):.0f}°')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'FCPM intensities at z={z_idx}')
    plt.tight_layout()

    return fig


def compare_directors(director1: DirectorField,
                      director2: DirectorField,
                      z_idx: int,
                      step: int = 2,
                      labels: Tuple[str, str] = ('Ground Truth', 'Reconstructed'),
                      figsize: Tuple[float, float] = (14, 5)) -> plt.Figure:
    """
    Side-by-side comparison of two director fields.

    Args:
        director1, director2: Director fields to compare.
        z_idx: Z-slice index.
        step: Subsampling step.
        labels: Labels for the two fields.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_director_slice(director1, z_idx, step=step, ax=axes[0],
                        title=labels[0], show_colorbar=True)
    plot_director_slice(director2, z_idx, step=step, ax=axes[1],
                        title=labels[1], show_colorbar=True)

    plt.tight_layout()
    return fig


def plot_error_map(director_recon: DirectorField,
                   director_gt: DirectorField,
                   z_idx: int,
                   error_type: str = 'angular',
                   ax: Optional[plt.Axes] = None,
                   figsize: Tuple[float, float] = (8, 6),
                   cmap: str = 'hot') -> plt.Figure:
    """
    Plot spatial map of reconstruction error.

    Args:
        director_recon: Reconstructed director.
        director_gt: Ground truth director.
        z_idx: Z-slice index.
        error_type: 'angular' (nematic-aware angle) or 'euclidean'.
        ax: Matplotlib axes.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Matplotlib figure.
    """
    n_recon = director_recon.to_array()[:, :, z_idx]
    n_gt = director_gt.to_array()[:, :, z_idx]

    if error_type == 'angular':
        # Nematic-aware angle: min(angle(n1, n2), angle(n1, -n2))
        dot = np.abs(np.sum(n_recon * n_gt, axis=-1))
        dot = np.clip(dot, -1, 1)
        error = np.degrees(np.arccos(dot))
    else:
        # Euclidean (not nematic-aware)
        error = np.linalg.norm(n_recon - n_gt, axis=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(error, cmap=cmap, origin='lower')
    plt.colorbar(im, ax=ax, label=f'{error_type.capitalize()} error')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{error_type.capitalize()} error at z={z_idx}')

    return fig
