"""
FCPM Reconstruction Module

Methods for reconstructing the director field from FCPM intensity measurements.

Approaches:
1. Direct inversion - algebraic inversion of FCPM formula
2. Q-tensor method - sign-invariant reconstruction via Q-tensor
3. Sign optimization - enforce spatial consistency of director signs

Recommended pipeline:
    1. Use reconstruct_via_qtensor() for initial reconstruction
    2. Apply combined_optimization() for sign consistency
    3. Verify with compute_fcpm_intensity_error()
"""

from .direct import (
    extract_magnitudes,
    extract_cross_term,
    reconstruct_director_direct,
    reconstruct_director_all_angles,
)

from .qtensor_method import (
    qtensor_from_fcpm,
    qtensor_from_fcpm_exact,
    reconstruct_via_qtensor,
    compute_qtensor_error,
)

from .sign_optimization import (
    chain_propagation,
    iterative_local_flip,
    wavefront_propagation,
    multi_axis_propagation,
    combined_optimization,
    gradient_energy,
)

__all__ = [
    # Direct methods
    'extract_magnitudes',
    'extract_cross_term',
    'reconstruct_director_direct',
    'reconstruct_director_all_angles',
    # Q-tensor methods
    'qtensor_from_fcpm',
    'qtensor_from_fcpm_exact',
    'reconstruct_via_qtensor',
    'compute_qtensor_error',
    # Sign optimization
    'chain_propagation',
    'iterative_local_flip',
    'wavefront_propagation',
    'multi_axis_propagation',
    'combined_optimization',
    'gradient_energy',
]
