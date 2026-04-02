"""
Parameter sensitivity analysis for FCPM reconstruction.

Systematically sweeps physical or experimental parameters to measure
how reconstruction quality degrades under model mismatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.director import DirectorField
from ..utils.metrics import angular_error_nematic, sign_accuracy, summary_metrics


@dataclass
class SensitivityResult:
    """Result of a single sensitivity trial."""
    param_name: str
    param_value: float
    metrics: Dict[str, float]
    time_s: float


def parameter_sensitivity_study(
    director_gt: DirectorField,
    param_name: str,
    param_values: Sequence[float],
    reconstruction_fn: Callable[..., DirectorField],
    param_kwarg: Optional[str] = None,
    verbose: bool = False,
) -> List[SensitivityResult]:
    """Sweep a parameter and measure reconstruction quality at each value.

    The reconstruction function is called once per parameter value.
    The parameter is passed as a keyword argument.

    Example::

        from fcpm import simulate_fcpm, reconstruct

        def recon_with_noise(director_gt, noise_level=0.0):
            I = simulate_fcpm(director_gt)
            # Add noise at the specified level
            from fcpm.utils.noise import add_gaussian_noise
            I_noisy = {a: add_gaussian_noise(v, sigma=noise_level)
                       for a, v in I.items()}
            d_recon, _ = reconstruct(I_noisy)
            return d_recon

        results = parameter_sensitivity_study(
            director_gt=gt,
            param_name='noise_level',
            param_values=[0.0, 0.01, 0.02, 0.05, 0.1],
            reconstruction_fn=recon_with_noise,
            param_kwarg='noise_level',
        )

    Args:
        director_gt: Ground truth director field.
        param_name: Human-readable name of the parameter being swept.
        param_values: Values to test.
        reconstruction_fn: Function that takes director_gt and the swept
            parameter, returns a reconstructed DirectorField.
        param_kwarg: Keyword argument name to pass the parameter value.
            If None, uses param_name.
        verbose: Print progress.

    Returns:
        List of SensitivityResult, one per parameter value.
    """
    kwarg = param_kwarg or param_name
    results = []

    for i, value in enumerate(param_values):
        if verbose:
            print(f"  [{i+1}/{len(param_values)}] {param_name}={value}")

        t0 = time.perf_counter()
        d_recon = reconstruction_fn(director_gt, **{kwarg: value})
        elapsed = time.perf_counter() - t0

        metrics = summary_metrics(d_recon, director_gt)

        results.append(SensitivityResult(
            param_name=param_name,
            param_value=float(value),
            metrics=metrics,
            time_s=elapsed,
        ))

    return results
