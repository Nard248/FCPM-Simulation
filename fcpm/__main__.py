"""
Entry point for running fcpm as a module.

Usage:
    python -m fcpm reconstruct input.npz -o output/
    python -m fcpm simulate director.npz -o fcpm.npz
    python -m fcpm info data.npz
"""

from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
