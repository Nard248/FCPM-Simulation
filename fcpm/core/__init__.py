"""
FCPM Core Module

Core data structures and simulation for FCPM (Fluorescence Confocal Polarizing Microscopy).

Classes:
    DirectorField: 3D director field representation.
    QTensor: Q-tensor field representation (sign-invariant).
    FCPMSimulator: FCPM intensity simulation.

Functions:
    create_uniform_director: Create uniform director field.
    create_cholesteric_director: Create cholesteric (twisted) structure.
    create_radial_director: Create radial (hedgehog) structure.
    director_to_qtensor: Convert director to Q-tensor.
    simulate_fcpm: Generate FCPM intensities.
"""

from .director import (
    DirectorField,
    create_uniform_director,
    create_cholesteric_director,
    create_radial_director,
    DTYPE,
)

from .qtensor import (
    QTensor,
    director_to_qtensor,
    qtensor_difference,
)

from .simulation import (
    FCPMSimulator,
    simulate_fcpm,
    simulate_fcpm_extended,
    add_gaussian_noise,
    add_poisson_noise,
    compute_fcpm_observables,
)

__all__ = [
    # Classes
    'DirectorField',
    'QTensor',
    'FCPMSimulator',
    # Director creation
    'create_uniform_director',
    'create_cholesteric_director',
    'create_radial_director',
    # Q-tensor functions
    'director_to_qtensor',
    'qtensor_difference',
    # Simulation functions
    'simulate_fcpm',
    'simulate_fcpm_extended',
    'add_gaussian_noise',
    'add_poisson_noise',
    'compute_fcpm_observables',
    # Constants
    'DTYPE',
]
