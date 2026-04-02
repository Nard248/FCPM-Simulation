"""
Noise modeling for FCPM measurements.

Provides a structured NoiseModel with realistic detector physics
and log-likelihood computation for statistically grounded reconstruction.
"""
from .model import NoiseModel

__all__ = ['NoiseModel']
