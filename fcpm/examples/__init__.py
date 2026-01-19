"""
FCPM Example Scripts

Example scripts demonstrating the FCPM library capabilities.

Examples:
    basic_simulation: Simple FCPM simulation workflow
    full_reconstruction: Complete reconstruction pipeline with analysis

Run examples:
    python -m fcpm.examples.basic_simulation
    python -m fcpm.examples.full_reconstruction
"""

from .basic_simulation import main as run_basic_simulation
from .full_reconstruction import main as run_full_reconstruction

__all__ = ['run_basic_simulation', 'run_full_reconstruction']
