"""
FCPM Processing Pipeline

High-level interface for complete FCPM simulation and reconstruction workflows.

The Pipeline class provides a streamlined, configurable interface for:
1. Loading data from files or creating synthetic data
2. Preprocessing (cropping, filtering, normalization)
3. Simulation (if starting from director field)
4. Reconstruction via Q-tensor method
5. Sign optimization
6. Evaluation and visualization
7. Saving results

Example:
    >>> pipeline = FCPMPipeline()
    >>> pipeline.load_director('director.npz')
    >>> pipeline.crop(center=True, size=(64, 64, 32))
    >>> pipeline.simulate(noise_level=0.05)
    >>> pipeline.reconstruct()
    >>> pipeline.save_results('output/')
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any, List
from dataclasses import dataclass, field
import json

from .core.director import DirectorField, DTYPE
from .core.qtensor import QTensor, director_to_qtensor
from .core.simulation import simulate_fcpm, FCPMSimulator
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
    remove_background_fcpm,
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


@dataclass
class PipelineConfig:
    """Configuration for FCPM pipeline."""

    # Preprocessing
    crop_enabled: bool = False
    crop_center: bool = True
    crop_size: Optional[Tuple[int, int, int]] = None
    crop_ranges: Optional[Dict[str, Tuple[int, int]]] = None

    filter_enabled: bool = False
    filter_type: str = 'gaussian'
    filter_sigma: float = 1.0

    normalize_enabled: bool = True
    normalize_method: str = 'global'

    background_subtract: bool = False
    background_method: str = 'percentile'

    # Simulation
    polarization_angles: List[float] = field(
        default_factory=lambda: [0, np.pi/4, np.pi/2, 3*np.pi/4]
    )
    noise_enabled: bool = False
    noise_model: str = 'mixed'
    noise_level: float = 0.05
    photon_count: float = 5000.0

    # Reconstruction
    reconstruction_method: str = 'qtensor'
    sign_optimization: bool = True
    sign_method: str = 'combined'

    # Output
    verbose: bool = True


class FCPMPipeline:
    """
    High-level pipeline for FCPM simulation and reconstruction.

    This class provides a fluent interface for building and executing
    FCPM processing workflows.

    Attributes:
        config: Pipeline configuration
        director_gt: Ground truth director (if available)
        director_recon: Reconstructed director
        I_fcpm: FCPM intensity data
        Q_tensor: Reconstructed Q-tensor
        metrics: Reconstruction metrics
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()

        # Data containers
        self.director_gt: Optional[DirectorField] = None
        self.director_recon: Optional[DirectorField] = None
        self.I_fcpm: Optional[Dict[float, np.ndarray]] = None
        self.I_fcpm_clean: Optional[Dict[float, np.ndarray]] = None
        self.Q_tensor: Optional[QTensor] = None

        # Results
        self.metrics: Dict[str, Any] = {}
        self.info: Dict[str, Any] = {}

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_director(self, source: Union[str, Path, np.ndarray, Dict],
                      **kwargs) -> 'FCPMPipeline':
        """
        Load director field from file or array.

        Args:
            source: File path, numpy array, or dictionary.
            **kwargs: Additional arguments for loader.

        Returns:
            self (for method chaining)
        """
        self.director_gt = load_director(source, **kwargs)

        if self.config.verbose:
            print(f"Loaded director field: shape={self.director_gt.shape}")

        return self

    def load_fcpm(self, source: Union[str, Path, np.ndarray, Dict],
                  angles: Optional[List[float]] = None,
                  **kwargs) -> 'FCPMPipeline':
        """
        Load FCPM intensity data from file or array.

        Args:
            source: File path, numpy array, or dictionary.
            angles: Polarization angles in radians.
            **kwargs: Additional arguments for loader.

        Returns:
            self (for method chaining)
        """
        if angles is None:
            angles = self.config.polarization_angles

        self.I_fcpm = load_fcpm(source, angles=angles, **kwargs)

        if self.config.verbose:
            first_shape = next(iter(self.I_fcpm.values())).shape
            print(f"Loaded FCPM data: {len(self.I_fcpm)} angles, shape={first_shape}")

        return self

    def set_director(self, director: DirectorField) -> 'FCPMPipeline':
        """Set director field directly."""
        self.director_gt = director
        return self

    def set_fcpm(self, I_fcpm: Dict[float, np.ndarray]) -> 'FCPMPipeline':
        """Set FCPM data directly."""
        self.I_fcpm = I_fcpm
        return self

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def crop(self,
             y_range: Optional[Tuple[int, int]] = None,
             x_range: Optional[Tuple[int, int]] = None,
             z_range: Optional[Tuple[int, int]] = None,
             center: bool = False,
             size: Optional[Union[int, Tuple[int, int, int]]] = None) -> 'FCPMPipeline':
        """
        Crop data to a region of interest.

        Args:
            y_range, x_range, z_range: Explicit crop ranges.
            center: If True, use center cropping with specified size.
            size: Size for center cropping.

        Returns:
            self (for method chaining)
        """
        if center and size is not None:
            if self.director_gt is not None:
                self.director_gt = crop_director_center(self.director_gt, size)
            if self.I_fcpm is not None:
                self.I_fcpm = crop_fcpm_center(self.I_fcpm, size)
        else:
            if self.director_gt is not None:
                self.director_gt = crop_director(
                    self.director_gt, y_range, x_range, z_range
                )
            if self.I_fcpm is not None:
                self.I_fcpm = crop_fcpm(self.I_fcpm, y_range, x_range, z_range)

        if self.config.verbose:
            if self.director_gt is not None:
                print(f"Cropped to shape: {self.director_gt.shape}")
            elif self.I_fcpm is not None:
                shape = next(iter(self.I_fcpm.values())).shape
                print(f"Cropped FCPM to shape: {shape}")

        return self

    def filter(self,
               method: str = 'gaussian',
               sigma: float = 1.0) -> 'FCPMPipeline':
        """
        Apply filtering to FCPM data.

        Args:
            method: Filter type ('gaussian', 'median').
            sigma: Filter parameter.

        Returns:
            self (for method chaining)
        """
        if self.I_fcpm is None:
            raise ValueError("No FCPM data to filter")

        if method == 'gaussian':
            self.I_fcpm = gaussian_filter_fcpm(self.I_fcpm, sigma=sigma)
        elif method == 'median':
            from .preprocessing import median_filter_fcpm
            self.I_fcpm = median_filter_fcpm(self.I_fcpm, size=int(sigma))

        if self.config.verbose:
            print(f"Applied {method} filter (sigma={sigma})")

        return self

    def normalize(self, method: str = 'global') -> 'FCPMPipeline':
        """
        Normalize FCPM intensities.

        Args:
            method: Normalization method ('global', 'per_angle', 'max').

        Returns:
            self (for method chaining)
        """
        if self.I_fcpm is None:
            raise ValueError("No FCPM data to normalize")

        self.I_fcpm = normalize_fcpm(self.I_fcpm, method=method)

        if self.config.verbose:
            print(f"Normalized FCPM ({method})")

        return self

    def remove_background(self, method: str = 'percentile',
                          **kwargs) -> 'FCPMPipeline':
        """
        Remove background from FCPM data.

        Args:
            method: Background removal method.
            **kwargs: Additional parameters.

        Returns:
            self (for method chaining)
        """
        if self.I_fcpm is None:
            raise ValueError("No FCPM data")

        self.I_fcpm = remove_background_fcpm(self.I_fcpm, method=method, **kwargs)

        if self.config.verbose:
            print(f"Removed background ({method})")

        return self

    def preprocess(self) -> 'FCPMPipeline':
        """
        Run all preprocessing steps according to config.

        Returns:
            self (for method chaining)
        """
        if self.config.crop_enabled:
            if self.config.crop_center and self.config.crop_size:
                self.crop(center=True, size=self.config.crop_size)
            elif self.config.crop_ranges:
                self.crop(**self.config.crop_ranges)

        if self.config.background_subtract and self.I_fcpm is not None:
            self.remove_background(method=self.config.background_method)

        if self.config.filter_enabled and self.I_fcpm is not None:
            self.filter(method=self.config.filter_type,
                       sigma=self.config.filter_sigma)

        if self.config.normalize_enabled and self.I_fcpm is not None:
            self.normalize(method=self.config.normalize_method)

        return self

    # =========================================================================
    # Simulation
    # =========================================================================

    def simulate(self,
                 noise_level: Optional[float] = None,
                 noise_model: Optional[str] = None,
                 seed: Optional[int] = None) -> 'FCPMPipeline':
        """
        Simulate FCPM measurements from director field.

        Args:
            noise_level: Override config noise level.
            noise_model: Override config noise model.
            seed: Random seed for noise.

        Returns:
            self (for method chaining)
        """
        if self.director_gt is None:
            raise ValueError("No director field. Load one first.")

        # Generate clean FCPM
        self.I_fcpm_clean = simulate_fcpm(
            self.director_gt,
            angles=self.config.polarization_angles
        )
        self.I_fcpm = self.I_fcpm_clean.copy()

        if self.config.verbose:
            print(f"Simulated FCPM for {len(self.I_fcpm)} angles")

        # Add noise if requested
        noise = noise_level if noise_level is not None else self.config.noise_level
        model = noise_model if noise_model is not None else self.config.noise_model

        if self.config.noise_enabled or noise_level is not None:
            self.I_fcpm = add_fcpm_realistic_noise(
                self.I_fcpm,
                noise_model=model,
                gaussian_sigma=noise,
                photon_count=self.config.photon_count,
                seed=seed
            )
            if self.config.verbose:
                print(f"Added {model} noise (level={noise})")

        return self

    # =========================================================================
    # Reconstruction
    # =========================================================================

    def reconstruct(self,
                    fix_signs: bool = True,
                    verbose: Optional[bool] = None) -> 'FCPMPipeline':
        """
        Reconstruct director field from FCPM data.

        Args:
            fix_signs: Whether to apply sign optimization.
            verbose: Override config verbosity.

        Returns:
            self (for method chaining)
        """
        if self.I_fcpm is None:
            raise ValueError("No FCPM data. Load or simulate first.")

        v = verbose if verbose is not None else self.config.verbose

        # Q-tensor reconstruction
        self.director_recon, self.Q_tensor, recon_info = reconstruct_via_qtensor(
            self.I_fcpm
        )
        self.info['reconstruction'] = recon_info

        if v:
            print("Q-tensor reconstruction complete")

        # Sign optimization
        if fix_signs and self.config.sign_optimization:
            self.director_recon, opt_info = combined_optimization(
                self.director_recon, verbose=v
            )
            self.info['sign_optimization'] = opt_info

            if v:
                print(f"Sign optimization: converged={opt_info['converged']}")

        return self

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.

        Returns:
            Dictionary of metrics.
        """
        if self.director_recon is None:
            raise ValueError("No reconstruction. Run reconstruct() first.")

        metrics = {}

        # If ground truth available, compute angular error
        if self.director_gt is not None:
            ang_err = angular_error_nematic(self.director_recon, self.director_gt)
            metrics['angular_error_mean'] = float(np.mean(ang_err))
            metrics['angular_error_median'] = float(np.median(ang_err))
            metrics['angular_error_max'] = float(np.max(ang_err))
            metrics['angular_error_90th'] = float(np.percentile(ang_err, 90))

        # Intensity reconstruction error
        I_recon = simulate_fcpm(self.director_recon,
                                angles=self.config.polarization_angles)

        I_ref = self.I_fcpm_clean if self.I_fcpm_clean is not None else self.I_fcpm
        intensity_err = intensity_reconstruction_error(I_ref, I_recon)
        metrics.update(intensity_err)

        # Gradient energy (measure of sign consistency)
        metrics['gradient_energy'] = gradient_energy(self.director_recon)

        self.metrics = metrics

        if self.config.verbose:
            print("\nReconstruction Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

        return metrics

    # =========================================================================
    # Output
    # =========================================================================

    def save_results(self, output_dir: Union[str, Path],
                     prefix: str = '') -> Path:
        """
        Save all results to a directory.

        Args:
            output_dir: Output directory path.
            prefix: Optional prefix for filenames.

        Returns:
            Path to output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save director fields
        if self.director_gt is not None:
            save_director_npz(
                self.director_gt,
                output_dir / f'{prefix}director_gt.npz'
            )

        if self.director_recon is not None:
            save_director_npz(
                self.director_recon,
                output_dir / f'{prefix}director_recon.npz'
            )

        # Save FCPM data
        if self.I_fcpm is not None:
            save_fcpm_npz(self.I_fcpm, output_dir / f'{prefix}fcpm.npz')

        # Save Q-tensor
        if self.Q_tensor is not None:
            save_qtensor_npz(self.Q_tensor, output_dir / f'{prefix}qtensor.npz')

        # Save metrics and info
        results = {
            'metrics': self.metrics,
            'info': self.info,
            'config': {
                'noise_level': self.config.noise_level,
                'noise_model': self.config.noise_model,
                'angles': [float(a) for a in self.config.polarization_angles],
            }
        }

        with open(output_dir / f'{prefix}results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self.config.verbose:
            print(f"Results saved to: {output_dir}")

        return output_dir

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def run(self) -> 'FCPMPipeline':
        """
        Run complete pipeline according to configuration.

        Returns:
            self (for method chaining)
        """
        # Preprocess
        self.preprocess()

        # Simulate if we have director but no FCPM
        if self.director_gt is not None and self.I_fcpm is None:
            self.simulate()

        # Reconstruct
        self.reconstruct()

        # Evaluate
        self.evaluate()

        return self

    def summary(self) -> str:
        """Generate a summary of the pipeline state."""
        lines = ["FCPM Pipeline Summary", "=" * 40]

        if self.director_gt is not None:
            lines.append(f"Ground truth: shape={self.director_gt.shape}")

        if self.I_fcpm is not None:
            shape = next(iter(self.I_fcpm.values())).shape
            lines.append(f"FCPM data: {len(self.I_fcpm)} angles, shape={shape}")

        if self.director_recon is not None:
            lines.append(f"Reconstruction: shape={self.director_recon.shape}")

        if self.metrics:
            lines.append("\nMetrics:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = []
        if self.director_gt is not None:
            parts.append(f"gt={self.director_gt.shape}")
        if self.I_fcpm is not None:
            parts.append(f"fcpm={len(self.I_fcpm)}angles")
        if self.director_recon is not None:
            parts.append("reconstructed")
        return f"FCPMPipeline({', '.join(parts)})"


# =============================================================================
# Convenience function for quick reconstruction
# =============================================================================

def quick_reconstruct(source: Union[str, Path, Dict[float, np.ndarray]],
                      crop_size: Optional[Tuple[int, int, int]] = None,
                      filter_sigma: Optional[float] = None,
                      verbose: bool = True) -> Tuple[DirectorField, Dict]:
    """
    Quick reconstruction from FCPM data with minimal configuration.

    Args:
        source: FCPM data source (file path or dictionary).
        crop_size: Optional center crop size.
        filter_sigma: Optional Gaussian filter sigma.
        verbose: Print progress.

    Returns:
        Tuple of (reconstructed DirectorField, metrics dict).

    Example:
        >>> director, metrics = fcpm.quick_reconstruct('fcpm_data.npz')
    """
    config = PipelineConfig(
        crop_enabled=crop_size is not None,
        crop_center=True,
        crop_size=crop_size,
        filter_enabled=filter_sigma is not None,
        filter_sigma=filter_sigma or 1.0,
        verbose=verbose
    )

    pipeline = FCPMPipeline(config)
    pipeline.load_fcpm(source)
    pipeline.preprocess()
    pipeline.reconstruct()
    pipeline.evaluate()

    return pipeline.director_recon, pipeline.metrics
