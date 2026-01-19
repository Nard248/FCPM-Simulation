#!/usr/bin/env python3
"""
FCPM Command Line Interface

Usage:
    python -m fcpm reconstruct input.npz -o output/
    python -m fcpm simulate director.npz -o fcpm_data.npz --noise 0.05
    python -m fcpm info data.npz

Commands:
    reconstruct: Reconstruct director from FCPM data
    simulate: Simulate FCPM from director field
    info: Display information about a data file
    convert: Convert between file formats
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional


def main(args: Optional[List[str]] = None):
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog='fcpm',
        description='FCPM Simulation and Reconstruction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fcpm reconstruct fcpm_data.npz -o output/
  fcpm simulate director.npz -o fcpm.npz --noise 0.05
  fcpm info data.npz
  fcpm convert director.npz -o director.mat --format mat
        """
    )

    parser.add_argument('--version', action='version', version='fcpm 1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Reconstruct command
    recon_parser = subparsers.add_parser(
        'reconstruct',
        help='Reconstruct director field from FCPM data'
    )
    recon_parser.add_argument('input', type=str, help='Input FCPM data file')
    recon_parser.add_argument('-o', '--output', type=str, required=True,
                              help='Output directory or file')
    recon_parser.add_argument('--crop', type=int, nargs=3, metavar=('Y', 'X', 'Z'),
                              help='Center crop to size (y, x, z)')
    recon_parser.add_argument('--filter', type=float, default=None,
                              help='Gaussian filter sigma')
    recon_parser.add_argument('--no-sign-fix', action='store_true',
                              help='Skip sign optimization')
    recon_parser.add_argument('-v', '--verbose', action='store_true',
                              help='Verbose output')

    # Simulate command
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Simulate FCPM from director field'
    )
    sim_parser.add_argument('input', type=str, help='Input director field file')
    sim_parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output FCPM file')
    sim_parser.add_argument('--noise', type=float, default=0.0,
                            help='Noise level (0-1)')
    sim_parser.add_argument('--noise-model', choices=['gaussian', 'poisson', 'mixed'],
                            default='mixed', help='Noise model')
    sim_parser.add_argument('--angles', type=float, nargs='+',
                            help='Polarization angles in degrees')
    sim_parser.add_argument('--crop', type=int, nargs=3, metavar=('Y', 'X', 'Z'),
                            help='Center crop to size before simulation')
    sim_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose output')

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Display information about a data file'
    )
    info_parser.add_argument('input', type=str, help='Input file')

    # Convert command
    conv_parser = subparsers.add_parser(
        'convert',
        help='Convert between file formats'
    )
    conv_parser.add_argument('input', type=str, help='Input file')
    conv_parser.add_argument('-o', '--output', type=str, required=True,
                             help='Output file')
    conv_parser.add_argument('--format', choices=['npz', 'npy', 'mat', 'vtk'],
                             help='Output format (inferred from extension if not specified)')
    conv_parser.add_argument('--type', choices=['director', 'fcpm', 'qtensor'],
                             default='director', help='Data type')

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == 'reconstruct':
        return cmd_reconstruct(parsed)
    elif parsed.command == 'simulate':
        return cmd_simulate(parsed)
    elif parsed.command == 'info':
        return cmd_info(parsed)
    elif parsed.command == 'convert':
        return cmd_convert(parsed)
    else:
        parser.print_help()
        return 1


def cmd_reconstruct(args) -> int:
    """Run reconstruction command."""
    import fcpm

    print(f"Loading FCPM data from: {args.input}")

    try:
        I_fcpm = fcpm.load_fcpm(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    shape = next(iter(I_fcpm.values())).shape
    print(f"  Shape: {shape}")
    print(f"  Angles: {len(I_fcpm)}")

    # Preprocessing
    if args.crop:
        print(f"Cropping to center {args.crop}...")
        I_fcpm = fcpm.crop_fcpm_center(I_fcpm, tuple(args.crop))

    if args.filter:
        print(f"Applying Gaussian filter (sigma={args.filter})...")
        I_fcpm = fcpm.gaussian_filter_fcpm(I_fcpm, sigma=args.filter)

    # Normalize
    I_fcpm = fcpm.normalize_fcpm(I_fcpm, method='global')

    # Reconstruct
    print("Reconstructing via Q-tensor method...")
    director, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)

    if not args.no_sign_fix:
        print("Fixing sign ambiguity...")
        director, opt_info = fcpm.combined_optimization(director, verbose=args.verbose)
        print(f"  Converged: {opt_info['converged']}")

    # Save results
    output_path = Path(args.output)
    if output_path.suffix:
        # Single file output
        fcpm.save_director_npz(director, output_path)
        print(f"Saved to: {output_path}")
    else:
        # Directory output
        output_path.mkdir(parents=True, exist_ok=True)
        fcpm.save_director_npz(director, output_path / 'director_recon.npz')
        fcpm.save_qtensor_npz(Q, output_path / 'qtensor.npz')
        print(f"Saved results to: {output_path}")

    print("Done!")
    return 0


def cmd_simulate(args) -> int:
    """Run simulation command."""
    import fcpm

    print(f"Loading director field from: {args.input}")

    try:
        director = fcpm.load_director(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    print(f"  Shape: {director.shape}")

    # Preprocessing
    if args.crop:
        print(f"Cropping to center {args.crop}...")
        director = fcpm.crop_director_center(director, tuple(args.crop))
        print(f"  New shape: {director.shape}")

    # Set up angles
    if args.angles:
        angles = [np.radians(a) for a in args.angles]
    else:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    print(f"Simulating FCPM for {len(angles)} angles...")
    I_fcpm = fcpm.simulate_fcpm(director, angles=angles)

    # Add noise
    if args.noise > 0:
        print(f"Adding {args.noise_model} noise (level={args.noise})...")
        I_fcpm = fcpm.add_fcpm_realistic_noise(
            I_fcpm,
            noise_model=args.noise_model,
            gaussian_sigma=args.noise
        )

    # Save
    output_path = Path(args.output)
    fcpm.save_fcpm_npz(I_fcpm, output_path)
    print(f"Saved to: {output_path}")

    print("Done!")
    return 0


def cmd_info(args) -> int:
    """Display file information."""
    filepath = Path(args.input)

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return 1

    print(f"File: {filepath}")
    print(f"Size: {filepath.stat().st_size / 1024:.1f} KB")

    suffix = filepath.suffix.lower()

    if suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        print(f"Format: NumPy NPZ")
        print(f"Keys: {list(data.keys())}")

        for key in data.keys():
            if key.startswith('_'):
                continue
            arr = data[key]
            if hasattr(arr, 'shape'):
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  {key}: {type(arr)}")

        # Try to detect content type
        if 'nx' in data and 'ny' in data and 'nz' in data:
            print("\nDetected: Director field")
            print(f"  Shape: {data['nx'].shape}")
        elif 'Q_xx' in data:
            print("\nDetected: Q-tensor field")
            print(f"  Shape: {data['Q_xx'].shape}")
        else:
            # Check for angle keys
            angle_keys = [k for k in data.keys() if k.replace('.', '').replace('-', '').isdigit()]
            if angle_keys:
                print(f"\nDetected: FCPM intensities ({len(angle_keys)} angles)")

    elif suffix == '.npy':
        data = np.load(filepath)
        print(f"Format: NumPy NPY")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")

    elif suffix in ['.tif', '.tiff']:
        try:
            import tifffile
            data = tifffile.imread(str(filepath))
            print(f"Format: TIFF")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
        except ImportError:
            print("Format: TIFF (install tifffile for details)")

    elif suffix == '.mat':
        try:
            from scipy.io import loadmat
            data = loadmat(str(filepath))
            print(f"Format: MATLAB MAT")
            print(f"Variables: {[k for k in data.keys() if not k.startswith('_')]}")
        except ImportError:
            print("Format: MATLAB MAT (install scipy for details)")

    else:
        print(f"Unknown format: {suffix}")

    return 0


def cmd_convert(args) -> int:
    """Convert between file formats."""
    import fcpm

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Determine output format
    if args.format:
        out_format = args.format
    else:
        out_format = output_path.suffix.lstrip('.').lower()

    print(f"Converting {input_path} -> {output_path} ({out_format})")

    if args.type == 'director':
        try:
            director = fcpm.load_director(input_path)
        except Exception as e:
            print(f"Error loading: {e}")
            return 1

        if out_format == 'npz':
            fcpm.save_director_npz(director, output_path)
        elif out_format == 'npy':
            fcpm.save_director_npy(director, output_path)
        elif out_format == 'mat':
            fcpm.export_for_matlab(director, output_path)
        elif out_format == 'vtk':
            fcpm.export_for_vtk(director, output_path)
        else:
            print(f"Unsupported output format: {out_format}")
            return 1

    elif args.type == 'fcpm':
        try:
            I_fcpm = fcpm.load_fcpm(input_path)
        except Exception as e:
            print(f"Error loading: {e}")
            return 1

        if out_format == 'npz':
            fcpm.save_fcpm_npz(I_fcpm, output_path)
        elif out_format in ['tif', 'tiff']:
            fcpm.save_fcpm_tiff(I_fcpm, output_path)
        else:
            print(f"Unsupported output format for FCPM: {out_format}")
            return 1

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
