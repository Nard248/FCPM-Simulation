"""
Benchmark Script for Advanced Sign Optimization Approaches.

Compares all implemented approaches across:
- Multiple noise levels
- Multiple volume sizes
- Various metrics (energy, angular error, time)

Usage:
    python benchmark_approaches.py

Results saved to v2/benchmark_results/
"""

import sys
from pathlib import Path
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import fcpm
from fcpm.reconstruction import reconstruct_via_qtensor

# Import all approaches
from sign_optimization_v2 import (
    layer_then_refine,
    compute_gradient_energy,
    OptimizationResult
)
from approaches.graph_cuts import GraphCutsOptimizer, GraphCutsConfig
from approaches.simulated_annealing import SimulatedAnnealingOptimizer, SimulatedAnnealingConfig
from approaches.hierarchical import HierarchicalOptimizer, HierarchicalConfig
from approaches.belief_propagation import BeliefPropagationOptimizer, BeliefPropagationConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    approach: str
    noise_level: float
    shape: tuple
    initial_energy: float
    final_energy: float
    energy_reduction_pct: float
    angular_error_mean: float
    angular_error_median: float
    time_seconds: float
    converged: Optional[bool] = None
    iterations: Optional[int] = None


def run_single_benchmark(
    director_raw,
    director_gt,
    approach_name: str,
    optimizer,
    noise_level: float
) -> BenchmarkResult:
    """Run a single approach and collect metrics."""

    initial_energy = compute_gradient_energy(director_raw.to_array())

    # Time the optimization
    t0 = time.time()
    result = optimizer.optimize(director_raw, verbose=False)
    elapsed = time.time() - t0

    # Compute metrics
    metrics = fcpm.summary_metrics(result.director, director_gt)

    energy_reduction = (initial_energy - result.final_energy) / initial_energy * 100

    benchmark_result = BenchmarkResult(
        approach=approach_name,
        noise_level=noise_level,
        shape=director_raw.shape,
        initial_energy=initial_energy,
        final_energy=result.final_energy,
        energy_reduction_pct=energy_reduction,
        angular_error_mean=metrics['angular_error_mean_deg'],
        angular_error_median=metrics['angular_error_median_deg'],
        time_seconds=elapsed,
        converged=getattr(result, 'converged', None),
        iterations=getattr(result, 'iterations', None),
    )

    return benchmark_result


def run_benchmark_suite(
    shape: tuple = (48, 48, 24),
    pitch: float = 6.0,
    noise_levels: List[float] = [0.01, 0.03, 0.05, 0.08, 0.10],
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        shape: Volume shape
        pitch: Cholesteric pitch
        noise_levels: List of noise levels to test
        verbose: Print progress

    Returns:
        List of BenchmarkResult
    """
    results = []

    # Create ground truth director
    if verbose:
        print("=" * 70)
        print("BENCHMARK: Advanced Sign Optimization Approaches")
        print("=" * 70)
        print(f"Shape: {shape}")
        print(f"Pitch: {pitch}")
        print(f"Noise levels: {noise_levels}")
        print()

    director_gt = fcpm.create_cholesteric_director(shape=shape, pitch=pitch)

    # Define approaches
    approaches = {
        'V2 Layer+Refine': lambda: LayerRefineWrapper(),
        'Graph Cuts': lambda: GraphCutsOptimizer(),
        'Simulated Annealing': lambda: SimulatedAnnealingOptimizer(
            SimulatedAnnealingConfig(max_iterations=30000, use_cluster_moves=True)
        ),
        'Hierarchical': lambda: HierarchicalOptimizer(),
        'Belief Propagation': lambda: BeliefPropagationOptimizer(
            BeliefPropagationConfig(max_iterations=100, damping=0.5)
        ),
    }

    for noise in noise_levels:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Noise Level: {noise*100:.0f}%")
            print('='*50)

        # Simulate and reconstruct
        I_fcpm = fcpm.simulate_fcpm(director_gt)
        I_fcpm_noisy = fcpm.add_fcpm_realistic_noise(
            I_fcpm, noise_model='mixed', gaussian_sigma=noise
        )
        I_fcpm_noisy = fcpm.normalize_fcpm(I_fcpm_noisy)

        director_raw, Q, info = reconstruct_via_qtensor(I_fcpm_noisy)

        raw_energy = compute_gradient_energy(director_raw.to_array())
        raw_metrics = fcpm.summary_metrics(director_raw, director_gt)

        if verbose:
            print(f"Raw (no sign fix): energy={raw_energy:.1f}, "
                  f"error={raw_metrics['angular_error_mean_deg']:.1f}°")

        for approach_name, optimizer_factory in approaches.items():
            if verbose:
                print(f"\n  {approach_name}...", end=" ", flush=True)

            try:
                optimizer = optimizer_factory()
                result = run_single_benchmark(
                    director_raw, director_gt, approach_name, optimizer, noise
                )
                results.append(result)

                if verbose:
                    print(f"energy={result.final_energy:.1f}, "
                          f"error={result.angular_error_mean:.1f}°, "
                          f"time={result.time_seconds:.2f}s")

            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

    return results


class LayerRefineWrapper:
    """Wrapper to make layer_then_refine follow the optimizer interface."""

    def optimize(self, director, verbose=False):
        return layer_then_refine(director, verbose=verbose)


def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)

    # Group by noise level
    noise_levels = sorted(set(r.noise_level for r in results))
    approaches = sorted(set(r.approach for r in results))

    # Header
    header = f"{'Approach':<25}"
    for noise in noise_levels:
        header += f" | {noise*100:.0f}% Err"
    header += " | Avg Time"
    print(header)
    print("-" * 90)

    for approach in approaches:
        row = f"{approach:<25}"
        times = []

        for noise in noise_levels:
            matching = [r for r in results
                        if r.approach == approach and r.noise_level == noise]
            if matching:
                r = matching[0]
                row += f" | {r.angular_error_mean:6.2f}°"
                times.append(r.time_seconds)
            else:
                row += f" |   N/A  "

        if times:
            row += f" | {np.mean(times):7.2f}s"
        else:
            row += " |    N/A "

        print(row)

    print("=" * 90)


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"benchmark_{timestamp}.json"

    data = {
        'timestamp': timestamp,
        'results': [asdict(r) for r in results]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {filename}")
    return filename


def plot_results(results: List[BenchmarkResult], output_dir: Path):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize data
    noise_levels = sorted(set(r.noise_level for r in results))
    approaches = sorted(set(r.approach for r in results))

    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(approaches)))
    approach_colors = {a: c for a, c in zip(approaches, colors)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Angular Error vs Noise
    ax = axes[0]
    for approach in approaches:
        noise_vals = []
        error_vals = []
        for noise in noise_levels:
            matching = [r for r in results
                        if r.approach == approach and r.noise_level == noise]
            if matching:
                noise_vals.append(noise * 100)
                error_vals.append(matching[0].angular_error_mean)
        if noise_vals:
            ax.plot(noise_vals, error_vals, '-o', label=approach,
                    color=approach_colors[approach], linewidth=2, markersize=6)

    ax.set_xlabel('Noise Level (%)', fontsize=11)
    ax.set_ylabel('Mean Angular Error (°)', fontsize=11)
    ax.set_title('Angular Error vs Noise', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy vs Noise
    ax = axes[1]
    for approach in approaches:
        noise_vals = []
        energy_vals = []
        for noise in noise_levels:
            matching = [r for r in results
                        if r.approach == approach and r.noise_level == noise]
            if matching:
                noise_vals.append(noise * 100)
                energy_vals.append(matching[0].final_energy)
        if noise_vals:
            ax.plot(noise_vals, energy_vals, '-o', label=approach,
                    color=approach_colors[approach], linewidth=2, markersize=6)

    ax.set_xlabel('Noise Level (%)', fontsize=11)
    ax.set_ylabel('Final Energy', fontsize=11)
    ax.set_title('Energy vs Noise', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Computation Time
    ax = axes[2]
    approach_times = {}
    for approach in approaches:
        times = [r.time_seconds for r in results if r.approach == approach]
        if times:
            approach_times[approach] = np.mean(times)

    if approach_times:
        bars = ax.bar(range(len(approach_times)), list(approach_times.values()),
                      color=[approach_colors[a] for a in approach_times.keys()])
        ax.set_xticks(range(len(approach_times)))
        ax.set_xticklabels(list(approach_times.keys()), rotation=45, ha='right')
        ax.set_ylabel('Average Time (s)', fontsize=11)
        ax.set_title('Computation Time', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_file = output_dir / 'benchmark_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")

    plt.close()


def main():
    """Run full benchmark suite."""
    output_dir = Path(__file__).parent / 'benchmark_results'

    # Run benchmark
    results = run_benchmark_suite(
        shape=(48, 48, 24),
        pitch=6.0,
        noise_levels=[0.01, 0.03, 0.05, 0.08, 0.10],
        verbose=True
    )

    # Print summary
    print_summary_table(results)

    # Save results
    save_results(results, output_dir)

    # Generate plots
    try:
        plot_results(results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
