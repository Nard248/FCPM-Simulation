"""
Unit tests for the V2 optimizer subpackage.

Each optimizer is tested on a small cholesteric director field (16x16x8)
to verify:
- Returns OptimizationResult
- final_energy <= initial_energy
- Director shape is preserved
- Inherits from SignOptimizer
"""

import numpy as np
import pytest

import fcpm
from fcpm.reconstruction.base import SignOptimizer, OptimizationResult
from fcpm.reconstruction.optimizers import (
    CombinedOptimizer,
    LayerPropagationOptimizer,
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer,
    SimulatedAnnealingConfig,
    BeliefPropagationConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_director():
    """A reconstructed director with scrambled signs (16x16x8)."""
    gt = fcpm.create_cholesteric_director(shape=(16, 16, 8), pitch=6.0)
    I_fcpm = fcpm.simulate_fcpm(gt)
    recon, _, _ = fcpm.reconstruct_via_qtensor(I_fcpm)
    return recon


# ---------------------------------------------------------------------------
# Parameterized test for all optimizers
# ---------------------------------------------------------------------------

def _make_optimizers():
    """Return (name, optimizer) pairs for parameterized tests."""
    return [
        ("CombinedOptimizer", CombinedOptimizer()),
        ("LayerPropagationOptimizer", LayerPropagationOptimizer()),
        ("HierarchicalOptimizer", HierarchicalOptimizer()),
        ("BeliefPropagationOptimizer",
         BeliefPropagationOptimizer(BeliefPropagationConfig(max_iterations=10))),
        ("SimulatedAnnealingOptimizer",
         SimulatedAnnealingOptimizer(SimulatedAnnealingConfig(
             max_iterations=500, seed=42))),
    ]


@pytest.mark.parametrize("name,optimizer", _make_optimizers(), ids=lambda x: x if isinstance(x, str) else "")
class TestOptimizerInterface:
    """Verify that every optimizer fulfils the SignOptimizer contract."""

    def test_inherits_from_base(self, name, optimizer):
        assert isinstance(optimizer, SignOptimizer), \
            f"{name} does not inherit from SignOptimizer"

    def test_returns_optimization_result(self, name, optimizer, raw_director):
        result = optimizer.optimize(raw_director, verbose=False)
        assert isinstance(result, OptimizationResult), \
            f"{name}.optimize() did not return OptimizationResult"

    def test_energy_does_not_increase(self, name, optimizer, raw_director):
        result = optimizer.optimize(raw_director, verbose=False)
        assert result.final_energy <= result.initial_energy + 1e-6, \
            f"{name}: final_energy ({result.final_energy}) > initial_energy ({result.initial_energy})"

    def test_shape_preserved(self, name, optimizer, raw_director):
        result = optimizer.optimize(raw_director, verbose=False)
        assert result.director.shape == raw_director.shape, \
            f"{name}: shape mismatch {result.director.shape} vs {raw_director.shape}"

    def test_method_field_set(self, name, optimizer, raw_director):
        result = optimizer.optimize(raw_director, verbose=False)
        assert result.method, f"{name}: method field is empty"


# ---------------------------------------------------------------------------
# Graph Cuts: skip if maxflow not installed but test networkx fallback
# ---------------------------------------------------------------------------

class TestGraphCuts:
    def test_graph_cuts_runs(self, raw_director):
        optimizer = GraphCutsOptimizer()
        result = optimizer.optimize(raw_director, verbose=False)
        assert isinstance(result, OptimizationResult)
        assert result.final_energy <= result.initial_energy + 1e-6


# ---------------------------------------------------------------------------
# Convenience: top-level imports work
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_import_optimizer_classes(self):
        from fcpm import GraphCutsOptimizer  # noqa: F811
        from fcpm import SimulatedAnnealingOptimizer  # noqa: F811
        from fcpm import HierarchicalOptimizer  # noqa: F811
        from fcpm import BeliefPropagationOptimizer  # noqa: F811
        from fcpm import LayerPropagationOptimizer  # noqa: F811
        from fcpm import CombinedOptimizer  # noqa: F811

    def test_import_base(self):
        from fcpm import SignOptimizer, OptimizationResult  # noqa: F811
        assert SignOptimizer is not None
        assert OptimizationResult is not None

    def test_backward_compat_v1(self):
        """V1 functional exports still work."""
        from fcpm import combined_optimization, gradient_energy  # noqa: F811
        assert callable(combined_optimization)
        assert callable(gradient_energy)

    def test_version_bumped(self):
        assert fcpm.__version__ == "2.0.0"


# ---------------------------------------------------------------------------
# Energy module
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_gradient_energy_uniform(self):
        """Uniform director should have near-zero gradient energy."""
        d = fcpm.create_uniform_director((8, 8, 4))
        from fcpm.reconstruction.energy import compute_gradient_energy
        e = compute_gradient_energy(d)
        assert e < 1e-10

    def test_gradient_energy_accepts_array(self):
        from fcpm.reconstruction.energy import compute_gradient_energy
        arr = np.random.randn(8, 8, 4, 3)
        e = compute_gradient_energy(arr)
        assert e > 0

    def test_frank_energy_uniform_zero(self):
        """Uniform director should have zero Frank energy everywhere."""
        d = fcpm.create_uniform_director((16, 16, 8))
        from fcpm.reconstruction.energy import compute_frank_energy_anisotropic, FrankConstants
        result = compute_frank_energy_anisotropic(d, FrankConstants())
        # Interior voxels should be near zero; boundaries have artefacts from np.roll
        # Check that the bulk is small
        assert result['total_integrated'] < 1.0

    def test_frank_energy_cholesteric_twist(self):
        """Cholesteric with matching pitch should have low twist energy."""
        from fcpm.reconstruction.energy import (
            compute_frank_energy_anisotropic,
            FrankConstants,
        )
        d = fcpm.create_cholesteric_director((16, 16, 16), pitch=8.0)
        result = compute_frank_energy_anisotropic(
            d, FrankConstants(pitch=8.0))
        # The twist term should be smaller than without matching pitch
        result_no_match = compute_frank_energy_anisotropic(
            d, FrankConstants(pitch=None))
        assert result['twist_integrated'] < result_no_match['twist_integrated']


# ---------------------------------------------------------------------------
# OptimizationResult properties
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_energy_reduction(self):
        r = OptimizationResult(
            director=fcpm.create_uniform_director((4, 4, 2)),
            initial_energy=100.0,
            final_energy=30.0,
            method='test',
        )
        assert r.energy_reduction == 70.0
        assert abs(r.energy_reduction_pct - 70.0) < 1e-10
