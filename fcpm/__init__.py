"""
FCPM - Fluorescence Confocal Polarizing Microscopy Simulation and Reconstruction

A Python library for simulating and reconstructing liquid crystal director fields
from Fluorescence Confocal Polarizing Microscopy (FCPM) data.

Main Classes:
    DirectorField: 3D director field representation
    QTensor: Q-tensor field representation (sign-invariant)
    FCPMSimulator: FCPM intensity simulation

Quick Start:
    >>> import fcpm
    >>>
    >>> # Create a cholesteric director field
    >>> director = fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)
    >>>
    >>> # Simulate FCPM measurements
    >>> simulator = fcpm.FCPMSimulator(director)
    >>> I_fcpm = simulator.simulate()
    >>>
    >>> # Reconstruct director from FCPM
    >>> director_recon, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)
    >>>
    >>> # Fix sign ambiguity
    >>> director_fixed, _ = fcpm.combined_optimization(director_recon)
    >>>
    >>> # Visualize
    >>> fcpm.plot_director_slice(director_fixed, z_idx=16)

Physics Background:
    FCPM uses polarized two-photon excitation to probe the local molecular
    orientation in liquid crystals. The fluorescence intensity depends on
    the angle between the polarization direction and the director:

        I(α) ∝ [nx·cos(α) + ny·sin(α)]⁴

    where α is the polarization angle and (nx, ny, nz) is the director.

    The Q-tensor approach eliminates sign ambiguity because Q(n) = Q(-n).

Modules:
    fcpm.core: Core data structures and simulation
    fcpm.reconstruction: Director field reconstruction methods
    fcpm.visualization: Visualization tools
    fcpm.io: Data loading and saving
    fcpm.preprocessing: Cropping, filtering, normalization
    fcpm.utils: Utilities (noise, metrics)
    fcpm.pipeline: High-level processing pipeline

Command Line Interface:
    python -m fcpm reconstruct input.npz -o output/
    python -m fcpm simulate director.npz -o fcpm.npz
    python -m fcpm info data.npz

Author: FCPM Simulation Team
"""

__version__ = '2.0.0'
__author__ = 'FCPM Simulation Team'

# Core classes and functions
from .core import (
    # Classes
    DirectorField,
    QTensor,
    FCPMSimulator,
    # Director creation
    create_uniform_director,
    create_cholesteric_director,
    create_radial_director,
    # Q-tensor
    director_to_qtensor,
    qtensor_difference,
    # Simulation
    simulate_fcpm,
    simulate_fcpm_extended,
    add_gaussian_noise,
    add_poisson_noise,
    compute_fcpm_observables,
    # Constants
    DTYPE,
)

# Reconstruction
from .reconstruction import (
    # Direct methods
    extract_magnitudes,
    extract_cross_term,
    reconstruct_director_direct,
    reconstruct_director_all_angles,
    # Q-tensor methods
    qtensor_from_fcpm,
    qtensor_from_fcpm_exact,
    reconstruct_via_qtensor,
    compute_qtensor_error,
    # V1 Sign optimization (backward compatible)
    chain_propagation,
    iterative_local_flip,
    wavefront_propagation,
    multi_axis_propagation,
    combined_optimization,
    gradient_energy,
    # V2 base abstractions
    SignOptimizer,
    OptimizationResult,
    compute_gradient_energy,
    FrankConstants,
    compute_frank_energy_anisotropic,
    # V2 optimizer classes
    CombinedOptimizer,
    LayerPropagationOptimizer,
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer,
    # V2 configs
    LayerPropagationConfig,
    GraphCutsConfig,
    SimulatedAnnealingConfig,
    HierarchicalConfig,
    BeliefPropagationConfig,
    # V2 functional interfaces
    graph_cuts_optimization,
    simulated_annealing_optimization,
    hierarchical_optimization,
    belief_propagation_optimization,
    layer_propagation_optimization,
)

# Visualization
from .visualization import (
    # Director plots
    plot_director_slice,
    plot_director_streamlines,
    plot_director_rgb,
    plot_fcpm_intensities,
    compare_directors,
    plot_error_map,
    # TopoVec
    check_topovec_available,
    visualize_topovec,
    prepare_topovec_data,
    visualize_3d_matplotlib,
    export_for_paraview,
    # Analysis
    plot_error_histogram,
    plot_error_by_depth,
    plot_intensity_reconstruction,
    plot_convergence,
    plot_qtensor_components,
    plot_order_parameter,
    summary_statistics,
)

# I/O
from .io import (
    # Auto-detection loaders (recommended)
    load_director,
    load_fcpm,
    # Loaders
    load_director_npz,
    load_director_npy,
    load_director_mat,
    load_director_hdf5,
    load_fcpm_npz,
    load_fcpm_tiff_stack,
    load_fcpm_mat,
    load_qtensor_npz,
    load_simulation_results,
    load_lcsim_npz,
    # Exporters
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

# Preprocessing
from .preprocessing import (
    # Cropping
    crop_director,
    crop_director_center,
    crop_fcpm,
    crop_fcpm_center,
    crop_qtensor,
    pad_director,
    subsample_director,
    subsample_fcpm,
    # Filtering
    gaussian_filter_fcpm,
    median_filter_fcpm,
    remove_background_fcpm,
    normalize_fcpm,
    clip_fcpm,
    smooth_director,
)

# Pipeline
from .pipeline import (
    FCPMPipeline,
    PipelineConfig,
    quick_reconstruct,
)

# Workflows
from .workflows import (
    WorkflowConfig,
    WorkflowResults,
    run_simulation_reconstruction,
    run_reconstruction,
)

# Utilities
from .utils import (
    # Noise
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_pepper_noise,
    add_fcpm_realistic_noise,
    estimate_noise_level,
    signal_to_noise_ratio,
    # Metrics
    angular_error_nematic,
    angular_error_vector,
    euclidean_error,
    euclidean_error_nematic,
    intensity_reconstruction_error,
    qtensor_frobenius_error,
    summary_metrics,
    perfect_reconstruction_test,
    sign_accuracy,
    spatial_error_distribution,
)


# Convenience function for full reconstruction pipeline
def reconstruct(I_fcpm, fix_signs=True, verbose=False):
    """
    Complete reconstruction pipeline from FCPM to director field.

    This is the recommended one-stop function for reconstruction.

    Args:
        I_fcpm: Dictionary of FCPM intensities at angles [0, π/4, π/2, 3π/4].
        fix_signs: Whether to apply sign optimization for consistency.
        verbose: Print progress information.

    Returns:
        Tuple of (DirectorField, info_dict)

    Example:
        >>> director, info = fcpm.reconstruct(I_fcpm)
    """
    # Q-tensor reconstruction
    director, Q, q_info = reconstruct_via_qtensor(I_fcpm)

    if verbose:
        print(f"Q-tensor reconstruction complete")

    # Sign optimization
    if fix_signs:
        director, opt_info = combined_optimization(director, verbose=verbose)
        q_info.update(opt_info)

    return director, q_info


# Define what's exported with "from fcpm import *"
__all__ = [
    # Version
    '__version__',
    # Core classes
    'DirectorField',
    'QTensor',
    'FCPMSimulator',
    # Director creation
    'create_uniform_director',
    'create_cholesteric_director',
    'create_radial_director',
    # Q-tensor
    'director_to_qtensor',
    'qtensor_difference',
    # Simulation
    'simulate_fcpm',
    'simulate_fcpm_extended',
    'add_gaussian_noise',
    'add_poisson_noise',
    'compute_fcpm_observables',
    # Reconstruction
    'extract_magnitudes',
    'extract_cross_term',
    'reconstruct_director_direct',
    'reconstruct_director_all_angles',
    'qtensor_from_fcpm',
    'qtensor_from_fcpm_exact',
    'reconstruct_via_qtensor',
    'compute_qtensor_error',
    'chain_propagation',
    'iterative_local_flip',
    'wavefront_propagation',
    'multi_axis_propagation',
    'combined_optimization',
    'gradient_energy',
    'reconstruct',  # Convenience function
    # V2 base abstractions
    'SignOptimizer',
    'OptimizationResult',
    'compute_gradient_energy',
    'FrankConstants',
    'compute_frank_energy_anisotropic',
    # V2 optimizer classes
    'CombinedOptimizer',
    'LayerPropagationOptimizer',
    'GraphCutsOptimizer',
    'SimulatedAnnealingOptimizer',
    'HierarchicalOptimizer',
    'BeliefPropagationOptimizer',
    # V2 configs
    'LayerPropagationConfig',
    'GraphCutsConfig',
    'SimulatedAnnealingConfig',
    'HierarchicalConfig',
    'BeliefPropagationConfig',
    # V2 functional interfaces
    'graph_cuts_optimization',
    'simulated_annealing_optimization',
    'hierarchical_optimization',
    'belief_propagation_optimization',
    'layer_propagation_optimization',
    # Visualization
    'plot_director_slice',
    'plot_director_streamlines',
    'plot_director_rgb',
    'plot_fcpm_intensities',
    'compare_directors',
    'plot_error_map',
    'check_topovec_available',
    'visualize_topovec',
    'prepare_topovec_data',
    'visualize_3d_matplotlib',
    'export_for_paraview',
    'plot_error_histogram',
    'plot_error_by_depth',
    'plot_intensity_reconstruction',
    'plot_convergence',
    'plot_qtensor_components',
    'plot_order_parameter',
    'summary_statistics',
    # I/O (auto-detection)
    'load_director',
    'load_fcpm',
    # I/O (specific formats)
    'load_director_npz',
    'load_director_npy',
    'load_director_mat',
    'load_director_hdf5',
    'load_fcpm_npz',
    'load_fcpm_tiff_stack',
    'load_fcpm_mat',
    'load_qtensor_npz',
    'load_simulation_results',
    'load_lcsim_npz',
    'save_director_npz',
    'save_director_npy',
    'save_director_hdf5',
    'save_fcpm_npz',
    'save_fcpm_tiff',
    'save_qtensor_npz',
    'save_simulation_results',
    'export_for_matlab',
    'export_for_vtk',
    # Preprocessing
    'crop_director',
    'crop_director_center',
    'crop_fcpm',
    'crop_fcpm_center',
    'crop_qtensor',
    'pad_director',
    'subsample_director',
    'subsample_fcpm',
    'gaussian_filter_fcpm',
    'median_filter_fcpm',
    'remove_background_fcpm',
    'normalize_fcpm',
    'clip_fcpm',
    'smooth_director',
    # Pipeline
    'FCPMPipeline',
    'PipelineConfig',
    'quick_reconstruct',
    # Workflows
    'WorkflowConfig',
    'WorkflowResults',
    'run_simulation_reconstruction',
    'run_reconstruction',
    # Utilities
    'add_fcpm_realistic_noise',
    'estimate_noise_level',
    'signal_to_noise_ratio',
    'angular_error_nematic',
    'angular_error_vector',
    'euclidean_error',
    'euclidean_error_nematic',
    'intensity_reconstruction_error',
    'qtensor_frobenius_error',
    'summary_metrics',
    'perfect_reconstruction_test',
    'sign_accuracy',
    'spatial_error_distribution',
    # Constants
    'DTYPE',
]
