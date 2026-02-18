"""
FCPM Utilities Module

Utility functions for noise simulation and error metrics.

Submodules:
    noise: Noise generation and characterization
    metrics: Reconstruction quality metrics
"""

from .noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_pepper_noise,
    add_fcpm_realistic_noise,
    estimate_noise_level,
    signal_to_noise_ratio,
)

from .metrics import (
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

__all__ = [
    # Noise
    'add_gaussian_noise',
    'add_poisson_noise',
    'add_salt_pepper_noise',
    'add_fcpm_realistic_noise',
    'estimate_noise_level',
    'signal_to_noise_ratio',
    # Metrics
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
]
