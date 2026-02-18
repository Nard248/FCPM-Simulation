# Installation

## Requirements

- Python >= 3.10
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0

## Using UV (Recommended)

```bash
git clone https://github.com/your-org/FCPM-Simulation.git
cd FCPM-Simulation

uv sync                        # core dependencies
uv sync --extra full           # + pymaxflow, tifffile, h5py
uv sync --extra dev            # + pytest, ruff
uv sync --extra perf           # + numba acceleration
```

## Using pip

```bash
git clone https://github.com/your-org/FCPM-Simulation.git
cd FCPM-Simulation

pip install -e .               # core dependencies
pip install -e ".[full]"       # + optional extras
pip install -e ".[dev]"        # + development tools
pip install -e ".[perf]"       # + numba acceleration
```

## Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `full` | pymaxflow, tifffile, h5py | Graph cuts optimizer, TIFF I/O, HDF5 I/O |
| `dev` | pytest, pytest-cov, ruff | Testing and linting |
| `perf` | numba | JIT-compiled simulated annealing kernels |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | Documentation site |

## Verify Installation

```python
import fcpm
print(fcpm.__version__)  # Should print 2.0.0
```
