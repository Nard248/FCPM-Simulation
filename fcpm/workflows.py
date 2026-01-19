"""
FCPM Workflows - High-level interfaces for common use cases.

Two main workflows:
1. Simulation + Reconstruction: Load director → Simulate FCPM → Reconstruct → Evaluate
2. FCPM Reconstruction: Load FCPM data → Reconstruct (no ground truth)

Both workflows support output directory for saving all results.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any, List
from dataclasses import dataclass, field
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt

from .core.director import DirectorField, DTYPE
from .core.qtensor import QTensor, director_to_qtensor
from .core.simulation import simulate_fcpm
from .reconstruction import (
    reconstruct_via_qtensor,
    combined_optimization,
    gradient_energy,
)
from .preprocessing import (
    crop_director,
    crop_director_center,
    crop_fcpm,
    crop_fcpm_center,
    gaussian_filter_fcpm,
    normalize_fcpm,
)
from .io import (
    load_director,
    load_fcpm,
    save_director_npz,
    save_fcpm_npz,
    save_qtensor_npz,
)
from .utils.metrics import (
    angular_error_nematic,
    intensity_reconstruction_error,
    summary_metrics,
)
from .utils.noise import add_fcpm_realistic_noise
from .visualization import (
    plot_director_slice,
    plot_fcpm_intensities,
    compare_directors,
    plot_error_map,
    plot_error_histogram,
    plot_qtensor_components,
)


@dataclass
class WorkflowConfig:
    """Configuration for FCPM workflows."""

    # Cropping
    crop_size: Optional[Tuple[int, int, int]] = None
    crop_center: bool = True

    # Filtering
    filter_sigma: Optional[float] = None

    # Noise (for simulation workflow)
    noise_level: float = 0.0
    noise_model: str = 'mixed'
    noise_seed: Optional[int] = None

    # Reconstruction
    fix_signs: bool = True

    # Visualization
    z_slices: Optional[List[int]] = None  # Which z-slices to visualize
    quiver_step: int = 2  # Subsampling for quiver plots
    save_plots: bool = True
    show_plots: bool = False
    dpi: int = 150

    # Output
    save_data: bool = True
    verbose: bool = True


@dataclass
class WorkflowResults:
    """Results from a workflow run."""

    director_recon: DirectorField
    Q_tensor: QTensor
    metrics: Dict[str, Any]
    info: Dict[str, Any]

    # Optional (only for simulation workflow)
    director_gt: Optional[DirectorField] = None
    I_fcpm: Optional[Dict[float, np.ndarray]] = None
    I_fcpm_clean: Optional[Dict[float, np.ndarray]] = None

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = ["=" * 60, "FCPM Workflow Results", "=" * 60]

        if self.director_gt is not None:
            lines.append(f"\nGround Truth: shape={self.director_gt.shape}")

        lines.append(f"Reconstructed: shape={self.director_recon.shape}")

        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, int):
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def run_simulation_reconstruction(
    director_source: Union[str, Path, DirectorField],
    output_dir: Union[str, Path],
    config: Optional[WorkflowConfig] = None,
) -> WorkflowResults:
    """
    Run full simulation + reconstruction workflow.

    Workflow:
    1. Load director field (ground truth)
    2. Optionally crop and preprocess
    3. Simulate FCPM measurements
    4. Optionally add noise
    5. Reconstruct director from FCPM
    6. Evaluate against ground truth
    7. Save all results to output directory

    Args:
        director_source: Path to director file, or DirectorField object.
        output_dir: Directory to save all outputs.
        config: Workflow configuration.

    Returns:
        WorkflowResults object with all data and metrics.

    Example:
        >>> results = fcpm.run_simulation_reconstruction(
        ...     'director.npz',
        ...     'output/',
        ...     config=WorkflowConfig(crop_size=(64, 64, 32), noise_level=0.05)
        ... )
    """
    config = config or WorkflowConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print("=" * 60)
        print("FCPM Simulation + Reconstruction Workflow")
        print("=" * 60)

    # Step 1: Load director
    if config.verbose:
        print("\n[1/6] Loading director field...")

    if isinstance(director_source, DirectorField):
        director_gt = director_source
    else:
        director_gt = load_director(director_source)

    if config.verbose:
        print(f"      Shape: {director_gt.shape}")

    # Step 2: Preprocess (crop)
    if config.crop_size is not None:
        if config.verbose:
            print(f"\n[2/6] Cropping to {config.crop_size}...")

        if config.crop_center:
            director_gt = crop_director_center(director_gt, config.crop_size)
        else:
            # Use crop_size as ranges
            director_gt = crop_director(
                director_gt,
                y_range=(0, config.crop_size[0]),
                x_range=(0, config.crop_size[1]),
                z_range=(0, config.crop_size[2])
            )

        if config.verbose:
            print(f"      New shape: {director_gt.shape}")
    else:
        if config.verbose:
            print("\n[2/6] No cropping (using full volume)")

    # Step 3: Simulate FCPM
    if config.verbose:
        print("\n[3/6] Simulating FCPM measurements...")

    I_fcpm_clean = simulate_fcpm(director_gt)

    if config.verbose:
        print(f"      Generated {len(I_fcpm_clean)} polarization angles")

    # Step 4: Add noise
    if config.noise_level > 0:
        if config.verbose:
            print(f"\n[4/6] Adding {config.noise_model} noise (level={config.noise_level})...")

        I_fcpm = add_fcpm_realistic_noise(
            I_fcpm_clean,
            noise_model=config.noise_model,
            gaussian_sigma=config.noise_level,
            seed=config.noise_seed
        )
    else:
        if config.verbose:
            print("\n[4/6] No noise added")
        I_fcpm = I_fcpm_clean

    # Normalize
    I_fcpm = normalize_fcpm(I_fcpm, method='global')

    # Filter if requested
    if config.filter_sigma is not None:
        I_fcpm = gaussian_filter_fcpm(I_fcpm, sigma=config.filter_sigma)
        if config.verbose:
            print(f"      Applied Gaussian filter (sigma={config.filter_sigma})")

    # Step 5: Reconstruct
    if config.verbose:
        print("\n[5/6] Reconstructing from FCPM...")

    director_recon, Q_tensor, recon_info = reconstruct_via_qtensor(I_fcpm)

    if config.fix_signs:
        director_recon, opt_info = combined_optimization(director_recon, verbose=config.verbose)
        recon_info.update(opt_info)

    # Step 6: Evaluate
    if config.verbose:
        print("\n[6/6] Evaluating reconstruction...")

    # Compute metrics
    I_fcpm_recon = simulate_fcpm(director_recon)
    metrics = summary_metrics(director_recon, director_gt, I_fcpm_clean, I_fcpm_recon)

    # Add additional metrics
    metrics['gradient_energy'] = gradient_energy(director_recon)
    metrics['noise_level'] = config.noise_level

    # Print summary
    if config.verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Angular error (mean):   {metrics['angular_error_mean_deg']:.2f}°")
        print(f"  Angular error (median): {metrics['angular_error_median_deg']:.2f}°")
        print(f"  Angular error (max):    {metrics['angular_error_max_deg']:.2f}°")
        print(f"  Intensity RMSE:         {metrics.get('intensity_rmse_mean', 0):.2e}")

    # Save results
    _save_workflow_results(
        output_dir=output_dir,
        config=config,
        director_gt=director_gt,
        director_recon=director_recon,
        Q_tensor=Q_tensor,
        I_fcpm=I_fcpm,
        I_fcpm_clean=I_fcpm_clean,
        metrics=metrics,
        info=recon_info,
        has_ground_truth=True
    )

    return WorkflowResults(
        director_recon=director_recon,
        Q_tensor=Q_tensor,
        metrics=metrics,
        info=recon_info,
        director_gt=director_gt,
        I_fcpm=I_fcpm,
        I_fcpm_clean=I_fcpm_clean
    )


def run_reconstruction(
    fcpm_source: Union[str, Path, Dict[float, np.ndarray]],
    output_dir: Union[str, Path],
    config: Optional[WorkflowConfig] = None,
    angles: Optional[List[float]] = None,
) -> WorkflowResults:
    """
    Run FCPM reconstruction workflow (no ground truth).

    Workflow:
    1. Load FCPM intensity data
    2. Optionally crop and preprocess
    3. Reconstruct director from FCPM
    4. Save results to output directory

    Args:
        fcpm_source: Path to FCPM file, or dictionary of intensities.
        output_dir: Directory to save all outputs.
        config: Workflow configuration.
        angles: Polarization angles (for formats that need them).

    Returns:
        WorkflowResults object with reconstruction data.

    Example:
        >>> results = fcpm.run_reconstruction(
        ...     'fcpm_data.npz',
        ...     'output/',
        ...     config=WorkflowConfig(crop_size=(64, 64, 32))
        ... )
    """
    config = config or WorkflowConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print("=" * 60)
        print("FCPM Reconstruction Workflow")
        print("=" * 60)

    # Step 1: Load FCPM data
    if config.verbose:
        print("\n[1/4] Loading FCPM data...")

    if isinstance(fcpm_source, dict):
        I_fcpm = fcpm_source
    else:
        I_fcpm = load_fcpm(fcpm_source, angles=angles)

    first_shape = next(iter(I_fcpm.values())).shape
    if config.verbose:
        print(f"      Shape: {first_shape}")
        print(f"      Angles: {len(I_fcpm)}")

    # Step 2: Preprocess
    if config.crop_size is not None:
        if config.verbose:
            print(f"\n[2/4] Cropping to {config.crop_size}...")

        if config.crop_center:
            I_fcpm = crop_fcpm_center(I_fcpm, config.crop_size)
        else:
            I_fcpm = crop_fcpm(
                I_fcpm,
                y_range=(0, config.crop_size[0]),
                x_range=(0, config.crop_size[1]),
                z_range=(0, config.crop_size[2])
            )

        new_shape = next(iter(I_fcpm.values())).shape
        if config.verbose:
            print(f"      New shape: {new_shape}")
    else:
        if config.verbose:
            print("\n[2/4] No cropping")

    # Normalize
    I_fcpm = normalize_fcpm(I_fcpm, method='global')

    # Filter if requested
    if config.filter_sigma is not None:
        I_fcpm = gaussian_filter_fcpm(I_fcpm, sigma=config.filter_sigma)
        if config.verbose:
            print(f"      Applied Gaussian filter (sigma={config.filter_sigma})")

    # Step 3: Reconstruct
    if config.verbose:
        print("\n[3/4] Reconstructing from FCPM...")

    director_recon, Q_tensor, recon_info = reconstruct_via_qtensor(I_fcpm)

    if config.fix_signs:
        director_recon, opt_info = combined_optimization(director_recon, verbose=config.verbose)
        recon_info.update(opt_info)

    # Step 4: Compute basic metrics (no ground truth comparison)
    if config.verbose:
        print("\n[4/4] Computing metrics...")

    I_fcpm_recon = simulate_fcpm(director_recon)
    intensity_err = intensity_reconstruction_error(I_fcpm, I_fcpm_recon)

    metrics = {
        'gradient_energy': gradient_energy(director_recon),
        'intensity_rmse_mean': intensity_err['rmse_mean'],
        'intensity_relative_error': intensity_err['relative_error_mean'],
        'shape': director_recon.shape,
    }

    if config.verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Reconstructed shape: {director_recon.shape}")
        print(f"  Gradient energy:     {metrics['gradient_energy']:.2f}")
        print(f"  Intensity RMSE:      {metrics['intensity_rmse_mean']:.2e}")

    # Save results
    _save_workflow_results(
        output_dir=output_dir,
        config=config,
        director_gt=None,
        director_recon=director_recon,
        Q_tensor=Q_tensor,
        I_fcpm=I_fcpm,
        I_fcpm_clean=None,
        metrics=metrics,
        info=recon_info,
        has_ground_truth=False
    )

    return WorkflowResults(
        director_recon=director_recon,
        Q_tensor=Q_tensor,
        metrics=metrics,
        info=recon_info,
        director_gt=None,
        I_fcpm=I_fcpm,
        I_fcpm_clean=None
    )


def _save_workflow_results(
    output_dir: Path,
    config: WorkflowConfig,
    director_gt: Optional[DirectorField],
    director_recon: DirectorField,
    Q_tensor: QTensor,
    I_fcpm: Dict[float, np.ndarray],
    I_fcpm_clean: Optional[Dict[float, np.ndarray]],
    metrics: Dict[str, Any],
    info: Dict[str, Any],
    has_ground_truth: bool
) -> None:
    """Save all workflow results to output directory."""

    if config.verbose:
        print(f"\nSaving results to: {output_dir}")

    # Create subdirectories
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    if config.save_data:
        data_dir.mkdir(exist_ok=True)

    if config.save_plots:
        plots_dir.mkdir(exist_ok=True)

    # Determine z-slices for visualization
    shape = director_recon.shape
    if config.z_slices is None:
        # Default: 5 slices evenly spaced
        n_slices = min(5, shape[2])
        z_slices = [int(i * shape[2] / (n_slices + 1)) for i in range(1, n_slices + 1)]
    else:
        z_slices = config.z_slices

    # Save data files
    if config.save_data:
        save_director_npz(director_recon, data_dir / "director_reconstructed.npz")
        save_qtensor_npz(Q_tensor, data_dir / "qtensor.npz")
        save_fcpm_npz(I_fcpm, data_dir / "fcpm_input.npz")

        if director_gt is not None:
            save_director_npz(director_gt, data_dir / "director_ground_truth.npz")

        if I_fcpm_clean is not None:
            save_fcpm_npz(I_fcpm_clean, data_dir / "fcpm_clean.npz")

        if config.verbose:
            print(f"  Saved data files to: {data_dir}")

    # Save plots
    if config.save_plots:
        # Plot FCPM intensities
        for z_idx in z_slices:
            fig = plot_fcpm_intensities(I_fcpm, z_idx=z_idx)
            fig.savefig(plots_dir / f"fcpm_z{z_idx:03d}.png", dpi=config.dpi, bbox_inches='tight')
            plt.close(fig)

        # Plot reconstructed director
        for z_idx in z_slices:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            plot_director_slice(director_recon, z_idx, step=config.quiver_step,
                               ax=ax, title=f'Reconstructed Director (z={z_idx})')
            fig.savefig(plots_dir / f"director_recon_z{z_idx:03d}.png",
                       dpi=config.dpi, bbox_inches='tight')
            plt.close(fig)

        # If ground truth available, plot comparison and error
        if has_ground_truth and director_gt is not None:
            for z_idx in z_slices:
                # Comparison
                fig = compare_directors(director_gt, director_recon, z_idx,
                                       step=config.quiver_step)
                fig.savefig(plots_dir / f"comparison_z{z_idx:03d}.png",
                           dpi=config.dpi, bbox_inches='tight')
                plt.close(fig)

                # Error map
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                plot_error_map(director_recon, director_gt, z_idx, ax=ax)
                fig.savefig(plots_dir / f"error_map_z{z_idx:03d}.png",
                           dpi=config.dpi, bbox_inches='tight')
                plt.close(fig)

            # Error histogram
            fig = plot_error_histogram(director_recon, director_gt)
            fig.savefig(plots_dir / "error_histogram.png",
                       dpi=config.dpi, bbox_inches='tight')
            plt.close(fig)

        # Q-tensor components (middle slice)
        z_mid = shape[2] // 2
        fig = plot_qtensor_components(Q_tensor, z_idx=z_mid)
        fig.savefig(plots_dir / f"qtensor_components_z{z_mid:03d}.png",
                   dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)

        if config.verbose:
            print(f"  Saved plots to: {plots_dir}")

    # Save metrics and summary
    summary = {
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items() if not isinstance(v, tuple)},
        'config': {
            'crop_size': config.crop_size,
            'noise_level': config.noise_level,
            'noise_model': config.noise_model,
            'filter_sigma': config.filter_sigma,
            'fix_signs': config.fix_signs,
        },
        'info': {k: v for k, v in info.items()
                if isinstance(v, (str, int, float, bool, type(None)))},
        'has_ground_truth': has_ground_truth,
        'shape': list(director_recon.shape),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Also save human-readable summary
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("FCPM Workflow Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Reconstructed shape: {director_recon.shape}\n")
        f.write(f"Has ground truth: {has_ground_truth}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Crop size: {config.crop_size}\n")
        f.write(f"  Noise level: {config.noise_level}\n")
        f.write(f"  Filter sigma: {config.filter_sigma}\n")
        f.write(f"  Fix signs: {config.fix_signs}\n\n")

        f.write("Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            elif isinstance(value, (int, str)):
                f.write(f"  {key}: {value}\n")

    if config.verbose:
        print(f"  Saved summary to: {output_dir / 'summary.json'}")
        print("\nWorkflow complete!")
