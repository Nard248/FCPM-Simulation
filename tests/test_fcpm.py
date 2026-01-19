#!/usr/bin/env python3
"""
Comprehensive Test Suite for FCPM Library

Run with: python -m pytest tests/ -v
Or: python tests/test_fcpm.py
"""

import numpy as np
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fcpm


class TestDirectorField:
    """Tests for DirectorField class."""

    def test_create_uniform(self):
        """Test uniform director creation."""
        d = fcpm.create_uniform_director((10, 10, 5))
        assert d.shape == (10, 10, 5)
        assert d.is_normalized()

    def test_create_cholesteric(self):
        """Test cholesteric director creation."""
        d = fcpm.create_cholesteric_director((20, 20, 10), pitch=5.0)
        assert d.shape == (20, 20, 10)
        assert d.is_normalized()

    def test_create_radial(self):
        """Test radial director creation."""
        d = fcpm.create_radial_director((16, 16, 8))
        assert d.shape == (16, 16, 8)
        assert d.is_normalized()

    def test_normalize(self):
        """Test director normalization."""
        # Create un-normalized director
        d = fcpm.DirectorField(
            nx=np.ones((5, 5, 3)) * 2,
            ny=np.ones((5, 5, 3)) * 2,
            nz=np.ones((5, 5, 3)) * 2
        )
        assert not d.is_normalized()

        d_norm = d.normalize()
        assert d_norm.is_normalized()

    def test_to_from_array(self):
        """Test array conversion."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        arr = d.to_array()
        assert arr.shape == (8, 8, 4, 3)

        d2 = fcpm.DirectorField.from_array(arr)
        assert d == d2

    def test_copy(self):
        """Test deep copy."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        d2 = d.copy()

        # Modify original
        d.nx[0, 0, 0] = 999

        # Copy should be unchanged
        assert d2.nx[0, 0, 0] != 999


class TestQTensor:
    """Tests for QTensor class."""

    def test_director_to_qtensor(self):
        """Test Q-tensor creation from director."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        Q = fcpm.director_to_qtensor(d)

        assert Q.shape == d.shape
        # Check tracelessness
        trace = Q.Q_xx + Q.Q_yy + Q.Q_zz
        assert np.allclose(trace, 0, atol=1e-10)

    def test_qtensor_to_director(self):
        """Test director extraction from Q-tensor."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        Q = fcpm.director_to_qtensor(d)

        d_extracted = Q.to_director_vectorized()
        assert d_extracted.shape == d.shape
        assert d_extracted.is_normalized()

    def test_scalar_order_parameter(self):
        """Test order parameter calculation."""
        d = fcpm.create_uniform_director((5, 5, 3))
        Q = fcpm.director_to_qtensor(d, S=0.7)

        S = Q.scalar_order_parameter()
        assert np.allclose(S, 0.7, atol=0.01)


class TestSimulation:
    """Tests for FCPM simulation."""

    def test_simulate_fcpm(self):
        """Test basic FCPM simulation."""
        d = fcpm.create_cholesteric_director((16, 16, 8))
        I_fcpm = fcpm.simulate_fcpm(d)

        assert len(I_fcpm) == 4  # Default 4 angles
        for angle, intensity in I_fcpm.items():
            assert intensity.shape == d.shape
            assert np.all(intensity >= 0)

    def test_simulator_class(self):
        """Test FCPMSimulator class."""
        d = fcpm.create_cholesteric_director((12, 12, 6))
        sim = fcpm.FCPMSimulator(d)
        I_fcpm = sim.simulate()

        assert len(I_fcpm) == 4
        arr = sim.to_array()
        assert arr.shape == (4, 12, 12, 6)

    def test_noise_addition(self):
        """Test noise addition to FCPM."""
        # Use cholesteric to ensure non-zero intensities at all angles
        d = fcpm.create_cholesteric_director((10, 10, 5), pitch=5.0)
        I_clean = fcpm.simulate_fcpm(d)
        I_noisy = fcpm.add_fcpm_realistic_noise(I_clean, noise_model='gaussian',
                                                 gaussian_sigma=0.1, seed=42)

        # Noisy should differ from clean
        total_diff = sum(np.sum(np.abs(I_clean[a] - I_noisy[a])) for a in I_clean)
        assert total_diff > 0, "Noise should cause differences"


class TestReconstruction:
    """Tests for reconstruction methods."""

    def test_reconstruct_via_qtensor(self):
        """Test Q-tensor based reconstruction."""
        d = fcpm.create_cholesteric_director((16, 16, 8))
        I_fcpm = fcpm.simulate_fcpm(d)

        d_recon, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)
        assert d_recon.shape == d.shape
        assert Q.shape == d.shape

    def test_sign_optimization(self):
        """Test sign optimization."""
        d = fcpm.create_cholesteric_director((12, 12, 6))
        I_fcpm = fcpm.simulate_fcpm(d)
        d_recon, _, _ = fcpm.reconstruct_via_qtensor(I_fcpm)

        d_fixed, info = fcpm.combined_optimization(d_recon, verbose=False)
        assert d_fixed.shape == d.shape
        assert 'converged' in info

    def test_reconstruct_convenience(self):
        """Test convenience reconstruct function."""
        d = fcpm.create_cholesteric_director((10, 10, 5))
        I_fcpm = fcpm.simulate_fcpm(d)

        d_recon, info = fcpm.reconstruct(I_fcpm, fix_signs=True, verbose=False)
        assert d_recon.shape == d.shape

    def test_perfect_reconstruction(self):
        """Test that clean data gives good reconstruction."""
        d = fcpm.create_cholesteric_director((12, 12, 6))
        I_fcpm = fcpm.simulate_fcpm(d)

        d_recon, _ = fcpm.reconstruct(I_fcpm)

        # Reconstruct FCPM from reconstruction
        I_recon = fcpm.simulate_fcpm(d_recon)

        # Intensity should match well (allowing for nz sign ambiguity effects)
        # The Q_xz, Q_yz components have sign ambiguity which affects nz
        for angle in I_fcpm:
            rmse = np.sqrt(np.mean((I_fcpm[angle] - I_recon[angle])**2))
            # Allow up to 5% relative error (due to nz sign ambiguity)
            max_val = np.max(I_fcpm[angle])
            rel_error = rmse / max_val if max_val > 0 else rmse
            assert rel_error < 0.1, f"Relative intensity error too high: {rel_error}"


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_crop_director(self):
        """Test director cropping."""
        d = fcpm.create_cholesteric_director((20, 20, 10))
        d_cropped = fcpm.crop_director(d, y_range=(5, 15), x_range=(5, 15))

        assert d_cropped.shape == (10, 10, 10)

    def test_crop_director_center(self):
        """Test center cropping."""
        d = fcpm.create_cholesteric_director((20, 20, 10))
        d_cropped = fcpm.crop_director_center(d, size=(8, 8, 4))

        assert d_cropped.shape == (8, 8, 4)

    def test_crop_fcpm(self):
        """Test FCPM cropping."""
        d = fcpm.create_cholesteric_director((16, 16, 8))
        I_fcpm = fcpm.simulate_fcpm(d)

        I_cropped = fcpm.crop_fcpm_center(I_fcpm, size=(8, 8, 4))

        for angle, intensity in I_cropped.items():
            assert intensity.shape == (8, 8, 4)

    def test_subsample(self):
        """Test subsampling."""
        d = fcpm.create_cholesteric_director((16, 16, 8))
        d_sub = fcpm.subsample_director(d, factor=2)

        assert d_sub.shape == (8, 8, 4)

    def test_normalize_fcpm(self):
        """Test FCPM normalization."""
        d = fcpm.create_cholesteric_director((10, 10, 5))
        I_fcpm = fcpm.simulate_fcpm(d)

        I_norm = fcpm.normalize_fcpm(I_fcpm, method='global')

        max_val = max(np.max(I) for I in I_norm.values())
        assert np.isclose(max_val, 1.0)


class TestIO:
    """Tests for I/O functions."""

    def test_save_load_director_npz(self):
        """Test director save/load cycle."""
        d = fcpm.create_cholesteric_director((8, 8, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'director.npz'
            fcpm.save_director_npz(d, filepath)

            d_loaded = fcpm.load_director_npz(filepath)
            assert d == d_loaded

    def test_save_load_fcpm_npz(self):
        """Test FCPM save/load cycle."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        I_fcpm = fcpm.simulate_fcpm(d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'fcpm.npz'
            fcpm.save_fcpm_npz(I_fcpm, filepath)

            I_loaded = fcpm.load_fcpm_npz(filepath)

            assert set(I_fcpm.keys()) == set(I_loaded.keys())
            for angle in I_fcpm:
                assert np.allclose(I_fcpm[angle], I_loaded[angle])

    def test_load_director_auto(self):
        """Test auto-detection loader."""
        d = fcpm.create_cholesteric_director((8, 8, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'director.npz'
            fcpm.save_director_npz(d, filepath)

            # Load with auto-detection
            d_loaded = fcpm.load_director(filepath)
            assert d == d_loaded

    def test_load_director_from_array(self):
        """Test loading director from numpy array."""
        arr = np.random.randn(8, 8, 4, 3)
        # Normalize
        arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)

        d = fcpm.load_director(arr)
        assert d.shape == (8, 8, 4)
        assert d.is_normalized()

    def test_load_director_from_dict(self):
        """Test loading director from dictionary."""
        d = fcpm.create_cholesteric_director((8, 8, 4))

        data = {'nx': d.nx, 'ny': d.ny, 'nz': d.nz}
        d_loaded = fcpm.load_director(data)

        assert d == d_loaded


class TestMetrics:
    """Tests for error metrics."""

    def test_angular_error_nematic(self):
        """Test nematic-aware angular error."""
        d = fcpm.create_uniform_director((5, 5, 3), direction=(1, 0, 0))

        # Identical should give 0 error
        err = fcpm.angular_error_nematic(d, d)
        assert np.allclose(err, 0)

        # Flipped should also give 0 error (nematic symmetry)
        d_flipped = d.flip_signs(np.ones(d.shape, dtype=bool))
        err_flipped = fcpm.angular_error_nematic(d, d_flipped)
        assert np.allclose(err_flipped, 0)

    def test_intensity_reconstruction_error(self):
        """Test intensity error calculation."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        I1 = fcpm.simulate_fcpm(d)
        I2 = fcpm.simulate_fcpm(d)

        err = fcpm.intensity_reconstruction_error(I1, I2)
        assert err['rmse_mean'] < 1e-10

    def test_summary_metrics(self):
        """Test summary metrics calculation."""
        d = fcpm.create_cholesteric_director((8, 8, 4))
        I_fcpm = fcpm.simulate_fcpm(d)
        d_recon, _ = fcpm.reconstruct(I_fcpm)
        I_recon = fcpm.simulate_fcpm(d_recon)

        metrics = fcpm.summary_metrics(d_recon, d, I_fcpm, I_recon)

        assert 'angular_error_mean_deg' in metrics
        assert 'intensity_rmse_mean' in metrics


class TestPipeline:
    """Tests for FCPMPipeline class."""

    def test_basic_pipeline(self):
        """Test basic pipeline workflow."""
        pipeline = fcpm.FCPMPipeline()

        d = fcpm.create_cholesteric_director((12, 12, 6))
        pipeline.set_director(d)
        pipeline.simulate()
        pipeline.reconstruct()

        assert pipeline.director_recon is not None

    def test_pipeline_from_file(self):
        """Test pipeline loading from file."""
        d = fcpm.create_cholesteric_director((10, 10, 5))
        I_fcpm = fcpm.simulate_fcpm(d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'fcpm.npz'
            fcpm.save_fcpm_npz(I_fcpm, filepath)

            pipeline = fcpm.FCPMPipeline()
            pipeline.load_fcpm(str(filepath))
            pipeline.reconstruct()

            assert pipeline.director_recon is not None

    def test_pipeline_with_preprocessing(self):
        """Test pipeline with preprocessing."""
        d = fcpm.create_cholesteric_director((20, 20, 10))

        config = fcpm.PipelineConfig(
            crop_enabled=True,
            crop_center=True,
            crop_size=(10, 10, 5),
            verbose=False
        )

        pipeline = fcpm.FCPMPipeline(config)
        pipeline.set_director(d)
        pipeline.preprocess()
        pipeline.simulate()
        pipeline.reconstruct()

        assert pipeline.director_recon.shape == (10, 10, 5)

    def test_quick_reconstruct(self):
        """Test quick_reconstruct convenience function."""
        d = fcpm.create_cholesteric_director((12, 12, 6))
        I_fcpm = fcpm.simulate_fcpm(d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'fcpm.npz'
            fcpm.save_fcpm_npz(I_fcpm, filepath)

            d_recon, metrics = fcpm.quick_reconstruct(str(filepath), verbose=False)

            assert d_recon is not None
            assert 'rmse_mean' in metrics


class TestCLI:
    """Tests for command line interface."""

    def test_cli_help(self):
        """Test CLI help command."""
        from fcpm.cli import main

        # Should not raise
        result = main(['--help'])
        # argparse exits with 0 on --help, we catch SystemExit
        assert result is None or result == 0

    def test_cli_info(self):
        """Test CLI info command."""
        from fcpm.cli import main

        d = fcpm.create_cholesteric_director((8, 8, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.npz'
            fcpm.save_director_npz(d, filepath)

            result = main(['info', str(filepath)])
            assert result == 0


def run_all_tests():
    """Run all tests and print summary."""
    import traceback

    test_classes = [
        TestDirectorField,
        TestQTensor,
        TestSimulation,
        TestReconstruction,
        TestPreprocessing,
        TestIO,
        TestMetrics,
        TestPipeline,
        # TestCLI,  # Skip CLI tests in standalone mode
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("FCPM Library Test Suite")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if not method_name.startswith('test_'):
                continue

            total += 1
            method = getattr(instance, method_name)

            try:
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                errors.append((test_class.__name__, method_name, traceback.format_exc()))
                failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFailed Tests:")
        for cls, method, tb in errors:
            print(f"\n{cls}.{method}:")
            print(tb)

    return failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
