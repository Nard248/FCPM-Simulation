"""
FCPM Visualization Module

Tools for visualizing director fields, FCPM intensities, and analysis results.

Submodules:
    director_plot: 2D slice visualization of director fields
    topovec: TopoVec integration for 3D visualization
    analysis: Error analysis and statistics plots
"""

from .director_plot import (
    plot_director_slice,
    plot_director_streamlines,
    plot_director_rgb,
    plot_fcpm_intensities,
    compare_directors,
    plot_error_map,
)

from .topovec import (
    check_topovec_available,
    visualize_topovec,
    prepare_topovec_data,
    visualize_3d_matplotlib,
    export_for_paraview,
)

from .analysis import (
    plot_error_histogram,
    plot_error_by_depth,
    plot_intensity_reconstruction,
    plot_convergence,
    plot_qtensor_components,
    plot_order_parameter,
    summary_statistics,
)

__all__ = [
    # Director plots
    'plot_director_slice',
    'plot_director_streamlines',
    'plot_director_rgb',
    'plot_fcpm_intensities',
    'compare_directors',
    'plot_error_map',
    # TopoVec
    'check_topovec_available',
    'visualize_topovec',
    'prepare_topovec_data',
    'visualize_3d_matplotlib',
    'export_for_paraview',
    # Analysis
    'plot_error_histogram',
    'plot_error_by_depth',
    'plot_intensity_reconstruction',
    'plot_convergence',
    'plot_qtensor_components',
    'plot_order_parameter',
    'summary_statistics',
]
