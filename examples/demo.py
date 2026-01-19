"""
FCPM Library Demo - Two Workflow Examples

This script demonstrates both main workflows:
1. Simulation + Reconstruction: Load director → Simulate FCPM → Reconstruct → Compare
2. FCPM Reconstruction: Load FCPM data → Reconstruct (no ground truth)

Run from project root: python examples/demo.py
Or from examples/: python demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import fcpm

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
DATA_FILE = PROJECT_ROOT / 'Archive/Clean Simulation/Data/S1HighResol_6-10s.npz'


def demo_workflow_1():
    """
    Workflow 1: Simulation + Reconstruction

    Use case: You have a director field (simulation or known structure)
    and want to simulate FCPM, then test reconstruction quality.
    """
    print("\n" + "="*70)
    print("WORKFLOW 1: Simulation + Reconstruction (with ground truth)")
    print("="*70)

    # Your file contains a DIRECTOR FIELD
    filepath = DATA_FILE

    # Configure the workflow
    config = fcpm.WorkflowConfig(
        crop_size=(64, 64, 32),  # Crop to smaller region
        crop_center=True,
        noise_level=0.03,       # Add some noise
        noise_model='mixed',
        noise_seed=42,
        fix_signs=True,
        save_plots=True,
        save_data=True,
        verbose=True,
        dpi=150,
    )

    # Run the full workflow
    results = fcpm.run_simulation_reconstruction(
        director_source=filepath,
        output_dir=OUTPUT_DIR / 'workflow1_simulation',
        config=config
    )

    # Access results programmatically
    print("\n--- Results Object ---")
    print(f"Reconstructed director shape: {results.director_recon.shape}")
    print(f"Ground truth shape: {results.director_gt.shape}")
    print(f"Angular error (mean): {results.metrics['angular_error_mean_deg']:.2f}°")
    print(f"Angular error (median): {results.metrics['angular_error_median_deg']:.2f}°")

    return results


def demo_workflow_2():
    """
    Workflow 2: FCPM Reconstruction (no ground truth)

    Use case: You have experimental FCPM data and want to reconstruct
    the director field. No ground truth is available for comparison.
    """
    print("\n" + "="*70)
    print("WORKFLOW 2: FCPM Reconstruction (no ground truth)")
    print("="*70)

    # First, let's create some "experimental" FCPM data
    # In practice, you would load this from your experimental files
    print("\n[Preparing simulated FCPM data for demo...]")

    # Load director and simulate FCPM (pretend this is experimental data)
    director = fcpm.load_director(DATA_FILE)
    director_cropped = fcpm.crop_director_center(director, size=(48, 48, 24))
    I_fcpm = fcpm.simulate_fcpm(director_cropped)
    I_fcpm_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm, noise_model='mixed',
                                                  gaussian_sigma=0.05, seed=123)

    # Save this as "experimental data"
    fcpm_data_path = OUTPUT_DIR / 'simulated_fcpm_data.npz'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fcpm.save_fcpm_npz(I_fcpm_noisy, fcpm_data_path)
    print(f"Saved simulated FCPM data to: {fcpm_data_path}")

    # Now run reconstruction workflow on this data
    # (as if we don't know the ground truth)
    config = fcpm.WorkflowConfig(
        crop_size=None,     # No cropping needed, already small
        filter_sigma=0.5,   # Light smoothing
        fix_signs=True,
        save_plots=True,
        save_data=True,
        verbose=True,
    )

    results = fcpm.run_reconstruction(
        fcpm_source=fcpm_data_path,
        output_dir=OUTPUT_DIR / 'workflow2_reconstruction',
        config=config
    )

    # Access results
    print("\n--- Results Object ---")
    print(f"Reconstructed director shape: {results.director_recon.shape}")
    print(f"Gradient energy: {results.metrics['gradient_energy']:.2f}")
    print(f"Intensity RMSE: {results.metrics['intensity_rmse_mean']:.2e}")
    print(f"Ground truth available: {results.director_gt is not None}")

    return results


def demo_simple_api():
    """
    Simple API Demo

    For quick use without the full workflow, you can still use the
    simple function-based API.
    """
    print("\n" + "="*70)
    print("SIMPLE API: Quick reconstruction without workflow")
    print("="*70)

    # Load and crop
    director = fcpm.load_director(DATA_FILE)
    director_cropped = fcpm.crop_director_center(director, size=(32, 32, 16))
    print(f"Loaded director: {director_cropped.shape}")

    # Simulate
    I_fcpm = fcpm.simulate_fcpm(director_cropped)
    I_fcpm_noisy = fcpm.add_fcpm_realistic_noise(I_fcpm, gaussian_sigma=0.02)
    print(f"Simulated FCPM with {len(I_fcpm)} angles")

    # Reconstruct (one-liner)
    director_recon, info = fcpm.reconstruct(I_fcpm_noisy, fix_signs=True, verbose=True)

    # Evaluate
    metrics = fcpm.summary_metrics(director_recon, director_cropped)
    print(f"\nAngular error: {metrics['angular_error_mean_deg']:.2f}° (mean)")

    return director_recon


if __name__ == '__main__':
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("FCPM Library Demonstration")
    print("=" * 70)
    print(f"Library version: {fcpm.__version__}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # Run demos
    results1 = demo_workflow_1()
    results2 = demo_workflow_2()
    director = demo_simple_api()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nCheck the following output directories:")
    print(f"  - {OUTPUT_DIR / 'workflow1_simulation'}")
    print(f"  - {OUTPUT_DIR / 'workflow2_reconstruction'}")
