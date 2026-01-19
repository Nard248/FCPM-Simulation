"""
Unified Experiment Runner for FCPM Toron Simulations

This script runs a complete simulation pipeline based on a configuration,
organizing all outputs into a timestamped experiment folder.

Usage:
    python run_experiment.py                    # Interactive mode
    python run_experiment.py --config my_exp.json  # Config file mode
    python run_experiment.py --preset t3-1-standard  # Preset mode
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from skimage import measure
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import argparse

from toron_simulator_3d import ToronSimulator3D, SimulationParams3D
from optical_simulator import CrossPolarizerSimulator, OpticalParams
from advanced_visualization import TopologicalAnalyzer, IsosurfaceRenderer
from visualize_torons import visualize_director_field_3d
from enhanced_visualization import (create_onion_visualization,
                                   plot_cross_sections_enhanced,
                                   create_comprehensive_figure_fixed,
                                   export_for_topovec)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""

    # Experiment metadata
    experiment_name: str = "toron_experiment"
    description: str = ""

    # Structure parameters
    structure_type: str = 't3-1'  # 't3-1', 't3-2', 't3-3', 'uniform_vertical', 'uniform_helical'
    n_x: int = 48
    n_y: int = 48
    n_z: int = 48
    x_size: float = 6.0  # microns
    y_size: float = 6.0
    z_size: float = 6.0
    pitch: float = 2.0
    toron_radius: float = 2.0
    toron_center: tuple = (3.0, 3.0, 3.0)

    # Optical parameters
    wavelength: float = 0.55  # microns (green light)
    no: float = 1.5  # ordinary refractive index
    ne: float = 1.7  # extraordinary refractive index
    polarizer_angle: float = 0.0  # radians
    analyzer_angle: float = np.pi/2  # radians (crossed polarizers)
    add_noise: bool = False
    noise_level: float = 0.02

    # Visualization options
    generate_director_field_viz: bool = True
    generate_cross_sections: bool = True
    generate_isosurfaces: bool = True
    generate_optical_simulation: bool = True
    generate_comprehensive_figure: bool = True

    # Advanced options
    save_npz: bool = True
    save_raw_intensity: bool = True
    dpi: int = 150

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        # Handle tuple conversion
        if 'toron_center' in data and isinstance(data['toron_center'], list):
            data['toron_center'] = tuple(data['toron_center'])
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str):
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """Runs complete FCPM experiment pipeline"""

    def __init__(self, config: ExperimentConfig, output_dir: Optional[str] = None):
        self.config = config

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = config.experiment_name.replace(" ", "_").replace("/", "-")
            output_dir = f"experiments/{safe_name}_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"FCPM EXPERIMENT: {config.experiment_name}")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Structure type: {config.structure_type}")
        print(f"Grid: {config.n_x} × {config.n_y} × {config.n_z}")
        print(f"Size: {config.x_size} × {config.y_size} × {config.z_size} μm³")
        print("="*70 + "\n")

        # Storage for results
        self.simulator = None
        self.optical_sim = None
        self.intensity = None
        self.defects = None

    def run(self):
        """Execute complete experiment pipeline"""

        # Step 1: Save configuration
        self._save_config()

        # Step 2: Generate structure
        self._generate_structure()

        # Step 3: Run optical simulation
        if self.config.generate_optical_simulation:
            self._run_optical_simulation()

        # Step 4: Generate visualizations
        self._generate_visualizations()

        # Step 5: Create summary report
        self._create_summary_report()

        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.name:50s} ({size:8.1f} KB)")
        print("="*70 + "\n")

        return self.output_dir

    def _save_config(self):
        """Save experiment configuration"""
        print("Step 1: Saving configuration...")
        config_path = self.output_dir / "config.json"
        self.config.save_json(str(config_path))
        print(f"  ✓ Configuration saved to {config_path.name}\n")

    def _generate_structure(self):
        """Generate toron structure"""
        print("Step 2: Generating structure...")

        # Create simulation parameters
        params = SimulationParams3D(
            n_x=self.config.n_x,
            n_y=self.config.n_y,
            n_z=self.config.n_z,
            x_size=self.config.x_size,
            y_size=self.config.y_size,
            z_size=self.config.z_size,
            pitch=self.config.pitch,
            toron_radius=self.config.toron_radius,
            toron_center=self.config.toron_center,
            structure_type=self.config.structure_type,
            no=self.config.no,
            ne=self.config.ne
        )

        # Generate
        self.simulator = ToronSimulator3D(params)
        self.simulator.generate_structure()
        self.simulator.print_summary()

        # Save NPZ
        if self.config.save_npz:
            npz_path = self.output_dir / "director_field.npz"
            self.simulator.save_to_npz(str(npz_path))
            print(f"  ✓ Director field saved to {npz_path.name}")

        print()

    def _run_optical_simulation(self):
        """Run optical simulation"""
        print("Step 3: Running optical simulation...")

        optical_params = OpticalParams(
            wavelength=self.config.wavelength,
            no=self.config.no,
            ne=self.config.ne,
            polarizer_angle=self.config.polarizer_angle,
            analyzer_angle=self.config.analyzer_angle,
            add_noise=self.config.add_noise,
            noise_level=self.config.noise_level
        )

        self.optical_sim = CrossPolarizerSimulator(optical_params)

        info = self.simulator.get_grid_info()
        self.intensity = self.optical_sim.simulate_fcpm(
            self.simulator.director_field,
            info['physical_size']
        )

        # Save raw intensity
        if self.config.save_raw_intensity:
            intensity_path = self.output_dir / "fcpm_intensity.npy"
            np.save(intensity_path, self.intensity)
            print(f"  ✓ Intensity data saved to {intensity_path.name}")

        print()

    def _generate_visualizations(self):
        """Generate all requested visualizations"""
        print("Step 4: Generating visualizations...")

        if self.config.generate_director_field_viz:
            self._generate_director_field_viz()

        if self.config.generate_cross_sections:
            self._generate_cross_sections()

        if self.config.generate_isosurfaces:
            self._generate_isosurfaces()

        if self.config.generate_optical_simulation and self.optical_sim:
            self._generate_optical_viz()

        if self.config.generate_comprehensive_figure:
            self._generate_comprehensive_figure()

        print()

    def _generate_director_field_viz(self):
        """Generate director field visualization"""
        print("  - Creating director field visualization...")
        fig = visualize_director_field_3d(self.simulator, figsize=(18, 12))

        path = self.output_dir / "01_director_field.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved {path.name}")

    def _generate_cross_sections(self):
        """Generate cross-sections (ENHANCED - fixed bottom row!)"""
        print("  - Creating enhanced cross-sections...")

        fig = plot_cross_sections_enhanced(self.simulator, n_slices=5, figsize=(18, 10))

        path = self.output_dir / "02_cross_sections.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved {path.name}")

    def _generate_isosurfaces(self):
        """Generate onion-like isosurface visualization (Nature Materials style)"""
        print("  - Creating onion-like nested surface visualization...")

        # Create the beautiful onion structure!
        fig = create_onion_visualization(
            self.simulator,
            levels=[-0.9, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8],
            figsize=(14, 12),
            alpha=0.15,
            view_angles=(30, 45)
        )

        path = self.output_dir / "03_isosurfaces.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved {path.name}")

        # Also export for topovec
        topovec_path = self.output_dir / "topovec_export.npz"
        export_for_topovec(self.simulator, str(topovec_path))
        print(f"    ✓ Topovec export: {topovec_path.name}")

    def _generate_optical_viz(self):
        """Generate optical simulation visualization"""
        print("  - Creating optical simulation visualization...")

        fig = self.optical_sim.plot_results(figsize=(15, 10))

        path = self.output_dir / "04_optical_simulation.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved {path.name}")

    def _generate_comprehensive_figure(self):
        """Generate comprehensive figure (FIXED - no overlap!)"""
        print("  - Creating comprehensive figure (enhanced)...")

        fig = create_comprehensive_figure_fixed(
            self.simulator,
            optical_sim=self.optical_sim,
            config=self.config,
            figsize=(22, 14)
        )

        path = self.output_dir / "00_comprehensive.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved {path.name}")

    def _create_summary_report(self):
        """Create markdown summary report"""
        print("Step 5: Creating summary report...")

        report = f"""# Experiment Report: {self.config.experiment_name}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

| Parameter | Value |
|-----------|-------|
| Structure Type | {self.config.structure_type} |
| Grid Size | {self.config.n_x} × {self.config.n_y} × {self.config.n_z} |
| Physical Size | {self.config.x_size} × {self.config.y_size} × {self.config.z_size} μm³ |
| Pitch | {self.config.pitch} μm |
"""

        if self.config.structure_type.startswith('t3'):
            report += f"""| Toron Radius | {self.config.toron_radius} μm |
| Toron Center | ({self.config.toron_center[0]}, {self.config.toron_center[1]}, {self.config.toron_center[2]}) μm |
"""

        report += f"""
## Optical Parameters

| Parameter | Value |
|-----------|-------|
| Wavelength (λ) | {self.config.wavelength} μm |
| Ordinary Index (no) | {self.config.no} |
| Extraordinary Index (ne) | {self.config.ne} |
| Birefringence (Δn) | {self.config.ne - self.config.no:.3f} |
| Polarizer Angle | {self.config.polarizer_angle:.3f} rad ({np.degrees(self.config.polarizer_angle):.1f}°) |
| Analyzer Angle | {self.config.analyzer_angle:.3f} rad ({np.degrees(self.config.analyzer_angle):.1f}°) |

## Description

{self.config.description if self.config.description else "No description provided."}

## Topological Features

"""

        if self.simulator.defect_locations:
            report += f"**Total Defects:** {len(self.simulator.defect_locations)}\n\n"

            for i, defect in enumerate(self.simulator.defect_locations, 1):
                report += f"### Defect {i}: {defect['type']}\n"
                report += f"- **Charge:** {defect.get('charge', 'N/A')}\n"

                if 'physical_position' in defect:
                    pos = defect['physical_position']
                    report += f"- **Position:** ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) μm\n"
                elif 'center' in defect:
                    center = defect['center']
                    report += f"- **Center:** ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) μm\n"
                    if 'radius' in defect:
                        report += f"- **Radius:** {defect['radius']:.2f} μm\n"

                if 'description' in defect:
                    report += f"- **Description:** {defect['description']}\n"

                report += "\n"

            total_charge = sum(d.get('charge', 0) for d in self.simulator.defect_locations)
            report += f"**Total Topological Charge:** {total_charge:+d} (conserved)\n\n"
        else:
            report += "No defects detected.\n\n"

        # Statistics
        if hasattr(self.simulator, 'metadata') and self.simulator.metadata:
            stats = self.simulator.metadata.get('statistics', {})
            report += f"""## Director Field Statistics

| Component | Mean | Std Dev |
|-----------|------|---------|
| nx | {stats.get('nx_mean', 0):.4f} | {stats.get('nx_std', 0):.4f} |
| ny | {stats.get('ny_mean', 0):.4f} | {stats.get('ny_std', 0):.4f} |
| nz | {stats.get('nz_mean', 0):.4f} | {stats.get('nz_std', 0):.4f} |

"""

        # Optical results
        if self.intensity is not None:
            report += f"""## Optical Simulation Results

| Metric | Value |
|--------|-------|
| Min Intensity | {np.min(self.intensity):.4f} |
| Max Intensity | {np.max(self.intensity):.4f} |
| Mean Intensity | {np.mean(self.intensity):.4f} |
| Std Dev | {np.std(self.intensity):.4f} |

"""

        # Files generated
        report += f"""## Generated Files

"""
        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                report += f"- `{file.name}` ({size:.1f} KB)\n"

        # Save report
        report_path = self.output_dir / "REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"  ✓ Report saved to {report_path.name}\n")


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    't3-1-standard': ExperimentConfig(
        experiment_name="T3-1 Standard Resolution",
        description="Standard T3-1 toron with two hyperbolic point defects",
        structure_type='t3-1',
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0)
    ),

    't3-1-highres': ExperimentConfig(
        experiment_name="T3-1 High Resolution",
        description="High-resolution T3-1 toron for detailed analysis",
        structure_type='t3-1',
        n_x=64, n_y=64, n_z=64,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0)
    ),

    't3-2-standard': ExperimentConfig(
        experiment_name="T3-2 Standard Resolution",
        description="T3-2 toron with one point defect and one disclination ring",
        structure_type='t3-2',
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0)
    ),

    't3-3-standard': ExperimentConfig(
        experiment_name="T3-3 Standard Resolution",
        description="T3-3 toron with two disclination rings",
        structure_type='t3-3',
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0)
    ),

    'uniform-helical': ExperimentConfig(
        experiment_name="Uniform Helical Structure",
        description="Uniform cholesteric helix for baseline comparison",
        structure_type='uniform_helical',
        n_x=48, n_y=48, n_z=64,
        x_size=5.0, y_size=5.0, z_size=8.0,
        pitch=2.0
    ),

    'comparison-suite': ExperimentConfig(
        experiment_name="Toron Comparison Suite",
        description="Standard parameters for comparing all toron types",
        structure_type='t3-1',  # Will be run for all types
        n_x=48, n_y=48, n_z=48,
        x_size=6.0, y_size=6.0, z_size=6.0,
        pitch=2.0,
        toron_radius=2.0,
        toron_center=(3.0, 3.0, 3.0)
    )
}


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def interactive_mode():
    """Interactive configuration builder"""
    print("\n" + "="*70)
    print("FCPM EXPERIMENT - INTERACTIVE CONFIGURATION")
    print("="*70 + "\n")

    config = ExperimentConfig()

    # Basic info
    config.experiment_name = input("Experiment name [toron_experiment]: ").strip() or "toron_experiment"
    config.description = input("Description (optional): ").strip()

    # Structure type
    print("\nStructure types:")
    print("  1. t3-1 (2 point defects)")
    print("  2. t3-2 (1 point + 1 ring)")
    print("  3. t3-3 (2 rings)")
    print("  4. uniform_helical")
    print("  5. uniform_vertical")

    choice = input("Select structure type [1]: ").strip() or "1"
    structure_map = {'1': 't3-1', '2': 't3-2', '3': 't3-3',
                     '4': 'uniform_helical', '5': 'uniform_vertical'}
    config.structure_type = structure_map.get(choice, 't3-1')

    # Resolution
    print("\nResolution presets:")
    print("  1. Low (32³) - Fast")
    print("  2. Standard (48³) - Recommended")
    print("  3. High (64³) - Slow but detailed")
    print("  4. Custom")

    res_choice = input("Select resolution [2]: ").strip() or "2"

    if res_choice == '4':
        config.n_x = int(input("  n_x [48]: ").strip() or "48")
        config.n_y = int(input("  n_y [48]: ").strip() or "48")
        config.n_z = int(input("  n_z [48]: ").strip() or "48")
    else:
        res_map = {'1': 32, '2': 48, '3': 64}
        n = res_map.get(res_choice, 48)
        config.n_x = config.n_y = config.n_z = n

    # Physical size
    use_defaults = input("\nUse default physical parameters? [y]/n: ").strip().lower()
    if use_defaults == 'n':
        config.x_size = float(input("  x_size [6.0] μm: ").strip() or "6.0")
        config.y_size = float(input("  y_size [6.0] μm: ").strip() or "6.0")
        config.z_size = float(input("  z_size [6.0] μm: ").strip() or "6.0")
        config.pitch = float(input("  pitch [2.0] μm: ").strip() or "2.0")

        if config.structure_type.startswith('t3'):
            config.toron_radius = float(input("  toron_radius [2.0] μm: ").strip() or "2.0")

    print("\nConfiguration complete!")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run FCPM toron simulation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py                           # Interactive mode
  python run_experiment.py --preset t3-1-standard    # Use preset
  python run_experiment.py --config my_config.json   # Load config
  python run_experiment.py --list-presets            # Show available presets
        """
    )

    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--preset', type=str, help='Use preset configuration')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable presets:")
        print("-" * 70)
        for name, preset in PRESETS.items():
            print(f"\n{name}:")
            print(f"  Name: {preset.experiment_name}")
            print(f"  Type: {preset.structure_type}")
            print(f"  Grid: {preset.n_x}×{preset.n_y}×{preset.n_z}")
            print(f"  Description: {preset.description}")
        print("\n" + "-" * 70 + "\n")
        return

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = ExperimentConfig.from_json(args.config)
    elif args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'")
            print(f"Use --list-presets to see available options")
            return
        print(f"Using preset: {args.preset}")
        config = PRESETS[args.preset]
    else:
        # Interactive mode
        config = interactive_mode()

    # Run experiment
    runner = ExperimentRunner(config, output_dir=args.output_dir)
    runner.run()


if __name__ == "__main__":
    main()
