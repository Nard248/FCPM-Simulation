#!/usr/bin/env python3
"""
Soliton Validation Script

Validates all sign-optimization approaches on real liquid-crystal structures
(cholesteric fingers, Z-solitons, torons) from article.lcpen / LCSim data.

Usage:
    python examples/03_soliton_validation.py --input data/*.npz
    python examples/03_soliton_validation.py --input data/CF1.npz --output results/

If no real data is available, uses synthetic cholesteric fields as fallback.

Output:
    - JSON results file with energy recovery, sign accuracy, timing
    - PNG figures (director slices, error maps, method comparison)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import fcpm
from fcpm.reconstruction.base import OptimizationResult
from fcpm.reconstruction.optimizers import (
    BeliefPropagationConfig,
    BeliefPropagationOptimizer,
    CombinedOptimizer,
    GraphCutsOptimizer,
    HierarchicalOptimizer,
    LayerPropagationOptimizer,
    SimulatedAnnealingConfig,
    SimulatedAnnealingOptimizer,
)


def scramble_signs(director: fcpm.DirectorField, seed: int = 42) -> fcpm.DirectorField:
    """Randomly flip signs of 50% of voxels (controlled test)."""
    rng = np.random.default_rng(seed)
    n = director.to_array().copy()
    mask = rng.random(n.shape[:3]) < 0.5
    n[mask] = -n[mask]
    return fcpm.DirectorField.from_array(n, metadata=director.metadata)


def load_test_field(filepath: Optional[str] = None) -> fcpm.DirectorField:
    """Load from file or create synthetic cholesteric fallback."""
    if filepath and Path(filepath).exists():
        try:
            director, settings = fcpm.load_lcsim_npz(filepath)
            print(f"  Loaded LCSim data from {filepath}")
            print(f"  Shape: {director.shape}")
            if settings:
                print(f"  Settings: {settings}")
            return director
        except (ValueError, KeyError):
            director = fcpm.load_director(filepath)
            print(f"  Loaded director from {filepath}")
            print(f"  Shape: {director.shape}")
            return director

    print("  No input file. Using synthetic cholesteric (64x64x32, pitch=8).")
    return fcpm.create_cholesteric_director(shape=(64, 64, 32), pitch=8.0)


def run_optimizer(
    name: str,
    optimizer,
    director_scrambled: fcpm.DirectorField,
    verbose: bool = False,
) -> Dict:
    """Run a single optimizer and collect metrics."""
    t0 = time.perf_counter()
    result: OptimizationResult = optimizer.optimize(director_scrambled, verbose=verbose)
    elapsed = time.perf_counter() - t0

    return {
        "method": name,
        "initial_energy": result.initial_energy,
        "final_energy": result.final_energy,
        "energy_reduction_pct": result.energy_reduction_pct,
        "total_flips": result.total_flips,
        "time_s": round(elapsed, 3),
    }


def run_validation(
    filepath: Optional[str] = None,
    output_dir: str = "results/validation",
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict]:
    """Run all optimizers on a test field and return results."""
    print("=" * 70)
    print("FCPM Sign Optimization — Soliton Validation")
    print("=" * 70)

    # Load or create test field
    print("\n1. Loading director field...")
    director_gt = load_test_field(filepath)

    # Scramble signs
    print("\n2. Scrambling signs (50% random flips)...")
    director_scrambled = scramble_signs(director_gt, seed=seed)
    scrambled_energy = fcpm.compute_gradient_energy(director_scrambled)
    gt_energy = fcpm.compute_gradient_energy(director_gt)
    print(f"  Ground truth energy:  {gt_energy:.2f}")
    print(f"  Scrambled energy:     {scrambled_energy:.2f}")

    # Define optimizers
    optimizers = [
        ("Combined (V1)", CombinedOptimizer()),
        ("LayerPropagation", LayerPropagationOptimizer()),
        ("GraphCuts", GraphCutsOptimizer()),
        ("SimulatedAnnealing", SimulatedAnnealingOptimizer(
            SimulatedAnnealingConfig(max_iterations=10000, seed=seed))),
        ("Hierarchical", HierarchicalOptimizer()),
        ("BeliefPropagation", BeliefPropagationOptimizer(
            BeliefPropagationConfig(max_iterations=30))),
    ]

    # Run all optimizers
    print(f"\n3. Running {len(optimizers)} optimizers...")
    results = []
    for name, optimizer in optimizers:
        print(f"\n--- {name} ---")
        metrics = run_optimizer(name, optimizer, director_scrambled, verbose=verbose)

        # Add sign accuracy if we have ground truth
        result_obj = optimizer.optimize(director_scrambled, verbose=False)
        metrics["sign_accuracy"] = fcpm.sign_accuracy(result_obj.director, director_gt)

        # Energy recovery: how close to ground-truth energy?
        if scrambled_energy > gt_energy:
            energy_gap = scrambled_energy - gt_energy
            recovered = scrambled_energy - metrics["final_energy"]
            metrics["energy_recovery_pct"] = round(
                100.0 * recovered / energy_gap, 2) if energy_gap > 0 else 100.0
        else:
            metrics["energy_recovery_pct"] = 100.0

        results.append(metrics)
        print(f"  Energy: {metrics['initial_energy']:.1f} -> {metrics['final_energy']:.1f} "
              f"({metrics['energy_reduction_pct']:.1f}% reduction)")
        print(f"  Sign accuracy: {metrics['sign_accuracy']:.3f}")
        print(f"  Energy recovery: {metrics['energy_recovery_pct']:.1f}%")
        print(f"  Time: {metrics['time_s']:.3f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Energy Red%':>10} {'Sign Acc':>10} {'E Recovery%':>12} {'Time(s)':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<25} {r['energy_reduction_pct']:>9.1f}% "
              f"{r['sign_accuracy']:>9.3f} "
              f"{r['energy_recovery_pct']:>11.1f}% "
              f"{r['time_s']:>7.3f}")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_file = out_path / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "input": str(filepath) if filepath else "synthetic",
            "shape": list(director_gt.shape),
            "seed": seed,
            "gt_energy": gt_energy,
            "scrambled_energy": scrambled_energy,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate comparison bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = [r['method'] for r in results]
        x = range(len(methods))

        # Energy reduction
        axes[0].bar(x, [r['energy_reduction_pct'] for r in results], color='steelblue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Energy Reduction (%)')
        axes[0].set_title('Energy Reduction')

        # Sign accuracy
        axes[1].bar(x, [r['sign_accuracy'] for r in results], color='forestgreen')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Sign Accuracy')
        axes[1].set_title('Sign Accuracy')
        axes[1].set_ylim(0, 1.05)

        # Timing
        axes[2].bar(x, [r['time_s'] for r in results], color='coral')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[2].set_ylabel('Time (s)')
        axes[2].set_title('Execution Time')

        plt.tight_layout()
        fig_path = out_path / "method_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison figure saved to {fig_path}")
    except Exception as e:
        print(f"Could not generate figures: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate sign optimization on soliton structures")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to NPZ director file (LCSim or standard)")
    parser.add_argument("--output", "-o", type=str, default="results/validation",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed optimizer output")

    args = parser.parse_args()
    run_validation(
        filepath=args.input,
        output_dir=args.output,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
