#!/usr/bin/env python3
"""
FCPM - Fluorescence Confocal Polarizing Microscopy Simulation and Reconstruction

Installation:
    pip install -e .

For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="fcpm",
    version="1.0.0",
    author="FCPM Simulation Team",
    author_email="",
    description="FCPM Simulation and Reconstruction for Liquid Crystal Director Fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "full": [
            "scipy>=1.7.0",
            "tifffile>=2021.0.0",
            "vtk>=9.0.0",
        ],
        "topovec": [
            "topovec",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="FCPM, liquid crystal, director field, microscopy, simulation, reconstruction",
    project_urls={
        "Bug Reports": "",
        "Source": "",
    },
)
