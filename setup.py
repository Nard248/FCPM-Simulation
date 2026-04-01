#!/usr/bin/env python3
"""
DEPRECATED: This setup.py is kept for backward compatibility only.
The canonical build configuration is in pyproject.toml.

Use:
    uv sync              # recommended
    pip install -e .     # alternative (reads pyproject.toml)

This file will be removed in a future release.
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="fcpm-simulation",
    version="2.0.0",
    author="FCPM Simulation Team",
    author_email="",
    description="FCPM Simulation and Reconstruction for Liquid Crystal Director Fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "ruff",
        ],
        "full": [
            "pymaxflow>=1.2.0",
            "tifffile>=2021.0.0",
            "h5py>=3.8.0",
        ],
        "perf": [
            "numba>=0.57.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="FCPM, liquid crystal, director field, microscopy, simulation, reconstruction",
    project_urls={
        "Bug Reports": "",
        "Source": "",
    },
)
