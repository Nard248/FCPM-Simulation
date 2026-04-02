"""
Benchmark framework for FCPM reconstruction.

Provides reproducible benchmark suites and sensitivity analysis tools.
"""
from .sensitivity import parameter_sensitivity_study

__all__ = ['parameter_sensitivity_study']
