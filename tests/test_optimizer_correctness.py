"""
Correctness tests for sign optimizers.

These tests go beyond the interface tests in test_optimizers.py and verify
actual correctness against known solutions:

1. Brute-force oracle on small fields (exhaustive search over all 2^N sign configs)
2. Energy monotonicity on random fields (property-based)
3. Idempotency: running twice gives the same result
4. Known-solution test: single wrong-sign voxel in an otherwise consistent field

These tests address audit items 0.13 and 8.2 (test coverage for difficult regimes).
"""

import itertools
import numpy as np
import pytest

import fcpm
from fcpm.reconstruction.energy import compute_gradient_energy
from fcpm.reconstruction.base import OptimizationResult
from fcpm.reconstruction.optimizers import (
    CombinedOptimizer,
    LayerPropagationOptimizer,
    GraphCutsOptimizer,
    SimulatedAnnealingOptimizer,
    SimulatedAnnealingConfig,
    HierarchicalOptimizer,
    BeliefPropagationOptimizer,
    BeliefPropagationConfig,
)


# ---------------------------------------------------------------------------
# Brute-force oracle
# ---------------------------------------------------------------------------

def brute_force_optimal_signs(n: np.ndarray, max_voxels: int = 18) -> float:
    """Find the global minimum gradient energy over all 2^N sign configurations.

    Only feasible for very small fields (N <= ~18 voxels).

    Args:
        n: Director array of shape (ny, nx, nz, 3), assumed unit-normalized.
        max_voxels: Safety limit.

    Returns:
        The global minimum gradient energy.
    """
    ny, nx, nz = n.shape[:3]
    N = ny * nx * nz
    assert N <= max_voxels, f"Too many voxels ({N}) for brute-force search"

    flat = n.reshape(N, 3)
    best_energy = float('inf')

    # Iterate over all 2^N sign configurations
    # The first voxel's sign can be fixed (nematic symmetry n -> -n)
    for bits in range(2 ** (N - 1)):
        signs = np.ones(N)
        for k in range(N - 1):
            if bits & (1 << k):
                signs[k + 1] = -1.0

        n_signed = flat * signs[:, np.newaxis]
        n_3d = n_signed.reshape(ny, nx, nz, 3)
        energy = compute_gradient_energy(n_3d)

        if energy < best_energy:
            best_energy = energy

    return best_energy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cholesteric():
    """A small 4×4×2 cholesteric field with scrambled signs."""
    gt = fcpm.create_cholesteric_director(shape=(4, 4, 2), pitch=4.0)
    rng = np.random.default_rng(42)
    n = gt.to_array().copy()
    mask = rng.random(n.shape[:3]) < 0.5
    n[mask] = -n[mask]
    return fcpm.DirectorField.from_array(n), gt


@pytest.fixture
def single_flip_field():
    """A consistent field with exactly one wrong-sign voxel."""
    gt = fcpm.create_cholesteric_director(shape=(6, 6, 4), pitch=4.0)
    n = gt.to_array().copy()
    # Flip a single interior voxel
    n[3, 3, 2] = -n[3, 3, 2]
    return fcpm.DirectorField.from_array(n), gt


def _all_optimizers():
    """Return (name, optimizer) pairs for parametrised testing."""
    return [
        ("Combined", CombinedOptimizer()),
        ("LayerProp", LayerPropagationOptimizer()),
        ("GraphCuts", GraphCutsOptimizer()),
        ("Hierarchical", HierarchicalOptimizer()),
        ("SA", SimulatedAnnealingOptimizer(
            SimulatedAnnealingConfig(max_iterations=2000, seed=42))),
        ("BP", BeliefPropagationOptimizer(
            BeliefPropagationConfig(max_iterations=30))),
    ]


# ---------------------------------------------------------------------------
# Test: brute-force correctness on small field
# ---------------------------------------------------------------------------

class TestBruteForceCorrectness:
    """Verify optimizers find the true global minimum on small fields."""

    def test_brute_force_oracle_works(self):
        """Sanity check: brute-force finds zero energy for uniform field."""
        n = np.zeros((2, 2, 2, 3))
        n[..., 0] = 1.0  # uniform along x
        assert brute_force_optimal_signs(n) < 1e-10

    def test_brute_force_cholesteric_3x3x2(self):
        """Brute-force on a tiny cholesteric: all deterministic optimizers
        should find the global minimum."""
        gt = fcpm.create_cholesteric_director(shape=(3, 3, 2), pitch=4.0)
        rng = np.random.default_rng(123)
        n = gt.to_array().copy()
        mask = rng.random(n.shape[:3]) < 0.5
        n[mask] = -n[mask]
        scrambled = fcpm.DirectorField.from_array(n)

        global_min = brute_force_optimal_signs(gt.to_array())

        # Combined should find the global minimum
        result = CombinedOptimizer().optimize(scrambled, verbose=False)
        assert result.final_energy <= global_min + 1e-6, \
            f"Combined: {result.final_energy:.4f} > global min {global_min:.4f}"

        # GraphCuts should find the global minimum
        result_gc = GraphCutsOptimizer().optimize(scrambled, verbose=False)
        assert result_gc.final_energy <= global_min + 1e-6, \
            f"GraphCuts: {result_gc.final_energy:.4f} > global min {global_min:.4f}"


# ---------------------------------------------------------------------------
# Test: energy monotonicity on random fields
# ---------------------------------------------------------------------------

class TestEnergyMonotonicity:
    """Verify that no optimizer increases energy."""

    @pytest.mark.parametrize("name,optimizer", _all_optimizers(),
                             ids=lambda x: x if isinstance(x, str) else "")
    def test_energy_does_not_increase_random_field(self, name, optimizer):
        """Run on 50 random 6×6×4 fields and verify energy never increases."""
        rng = np.random.default_rng(seed=hash(name) % 2**31)
        failures = []

        for trial in range(50):
            # Random unit directors
            n = rng.standard_normal((6, 6, 4, 3))
            n /= np.linalg.norm(n, axis=-1, keepdims=True)

            # Random sign scramble
            mask = rng.random((6, 6, 4)) < 0.5
            n[mask] = -n[mask]

            director = fcpm.DirectorField.from_array(n)
            result = optimizer.optimize(director, verbose=False)

            if result.final_energy > result.initial_energy + 1e-6:
                failures.append(
                    f"trial {trial}: {result.final_energy:.4f} > {result.initial_energy:.4f}"
                )

        assert not failures, \
            f"{name}: energy increased in {len(failures)}/50 trials:\n" + \
            "\n".join(failures[:5])


# ---------------------------------------------------------------------------
# Test: single wrong-sign voxel
# ---------------------------------------------------------------------------

class TestSingleFlipRecovery:
    """Test that optimizers can fix a single wrong-sign voxel."""

    @pytest.mark.parametrize("name,optimizer", [
        ("Combined", CombinedOptimizer()),
        ("LayerProp", LayerPropagationOptimizer()),
        ("GraphCuts", GraphCutsOptimizer()),
        ("Hierarchical", HierarchicalOptimizer()),
    ], ids=lambda x: x if isinstance(x, str) else "")
    def test_single_flip_recovery(self, name, optimizer, single_flip_field):
        """A single wrong-sign voxel should be corrected."""
        scrambled, gt = single_flip_field
        result = optimizer.optimize(scrambled, verbose=False)

        gt_energy = compute_gradient_energy(gt)
        # The result should be at least as good as the ground truth
        assert result.final_energy <= gt_energy + 1e-6, \
            f"{name}: final {result.final_energy:.4f} > GT {gt_energy:.4f}"


# ---------------------------------------------------------------------------
# Test: idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Running an optimizer on its own output should be a no-op."""

    @pytest.mark.parametrize("name,optimizer", [
        ("Combined", CombinedOptimizer()),
        ("LayerProp", LayerPropagationOptimizer()),
        ("GraphCuts", GraphCutsOptimizer()),
        ("Hierarchical", HierarchicalOptimizer()),
    ], ids=lambda x: x if isinstance(x, str) else "")
    def test_idempotent(self, name, optimizer, small_cholesteric):
        """Optimizing twice should give the same energy."""
        scrambled, _ = small_cholesteric
        result1 = optimizer.optimize(scrambled, verbose=False)
        result2 = optimizer.optimize(result1.director, verbose=False)

        assert abs(result2.final_energy - result1.final_energy) < 1e-6, \
            f"{name}: not idempotent: {result1.final_energy:.4f} → {result2.final_energy:.4f}"


# ---------------------------------------------------------------------------
# Test: GraphCuts-specific correctness
# ---------------------------------------------------------------------------

class TestGraphCutsCorrectness:
    """Targeted tests for GraphCuts edge cases."""

    def test_uniform_field_zero_energy(self):
        """GraphCuts on a uniform field should return zero energy."""
        d = fcpm.create_uniform_director((8, 8, 4))
        result = GraphCutsOptimizer().optimize(d, verbose=False)
        assert result.final_energy < 1e-6

    def test_fully_flipped_field(self):
        """GraphCuts on a fully-flipped uniform field should fix all signs."""
        d = fcpm.create_uniform_director((8, 8, 4))
        n = -d.to_array()
        flipped = fcpm.DirectorField.from_array(n)
        result = GraphCutsOptimizer().optimize(flipped, verbose=False)
        assert result.final_energy < 1e-6

    def test_checkerboard_scramble(self):
        """GraphCuts should fix a checkerboard sign pattern."""
        gt = fcpm.create_cholesteric_director((8, 8, 4), pitch=4.0)
        n = gt.to_array().copy()
        # Checkerboard: flip every other voxel
        for y in range(8):
            for x in range(8):
                for z in range(4):
                    if (y + x + z) % 2 == 1:
                        n[y, x, z] = -n[y, x, z]
        scrambled = fcpm.DirectorField.from_array(n)

        gt_energy = compute_gradient_energy(gt)
        result = GraphCutsOptimizer().optimize(scrambled, verbose=False)
        assert result.final_energy <= gt_energy + 1e-6, \
            f"GraphCuts failed on checkerboard: {result.final_energy:.4f} > GT {gt_energy:.4f}"
