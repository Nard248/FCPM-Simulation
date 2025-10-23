"""
PHASE 3: Cross-Polarizer Optical Simulator
Implements Jones calculus for FCPM image generation from 3D director fields

Based on optics.py from lcsim, but adapted for our toron structures
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Optional, Tuple
import json


@dataclass
class OpticalParams:
    """Parameters for optical simulation"""
    wavelength: float = 0.55  # Wavelength in microns (green light)
    no: float = 1.5          # Ordinary refractive index
    ne: float = 1.7          # Extraordinary refractive index

    # Polarizer configuration
    polarizer_angle: float = 0.0      # Input polarizer angle (radians)
    analyzer_angle: float = np.pi/2   # Output analyzer angle (radians, π/2 = crossed)

    # Simulation settings
    dz_integration: Optional[float] = None  # Integration step (auto if None)
    beam_type: str = 'uniform'              # 'uniform' or 'gaussian'
    beam_width: Optional[float] = None      # For Gaussian beam

    # Image processing
    add_noise: bool = False
    noise_level: float = 0.02


class CrossPolarizerSimulator:
    """
    Simulates FCPM with crossed polarizers using Jones calculus

    Key innovation: Handles ARBITRARY 3D director orientation
    This was the main limitation of the old code!
    """

    def __init__(self, optical_params: OpticalParams = None):
        self.params = optical_params or OpticalParams()
        self.intensity_image = None
        self.electric_field = None

    def simulate_fcpm(self, director_field: np.ndarray,
                     physical_size: Tuple[float, float, float]) -> np.ndarray:
        """
        Main FCPM simulation function

        Args:
            director_field: Shape (n_x, n_y, n_z, 3) - director components
            physical_size: (x_size, y_size, z_size) in microns

        Returns:
            intensity_image: Shape (n_x, n_y) - FCPM intensity pattern
        """
        n_x, n_y, n_z, _ = director_field.shape
        x_size, y_size, z_size = physical_size

        # Auto-determine integration step
        if self.params.dz_integration is None:
            dz = z_size / n_z
        else:
            dz = self.params.dz_integration

        print(f"Optical Simulation:")
        print(f"  Grid: {n_x} × {n_y} × {n_z}")
        print(f"  Sample thickness: {z_size:.2f} μm")
        print(f"  Integration step: {dz:.4f} μm")
        print(f"  Wavelength: {self.params.wavelength:.3f} μm")
        print(f"  Δn = ne - no: {self.params.ne - self.params.no:.3f}")

        # Initialize electric field (uniform illumination)
        E = self._initialize_beam(n_x, n_y)

        # Apply input polarizer
        E = self._apply_polarizer(E, self.params.polarizer_angle)

        print(f"  Propagating through {n_z} layers...")

        # Layer-by-layer propagation (THIS IS THE KEY!)
        for k in range(n_z):
            if k % max(1, n_z // 10) == 0:
                print(f"    Layer {k}/{n_z} ({100*k/n_z:.1f}%)")

            # Get director at this z-layer
            director_layer = director_field[:, :, k, :]  # Shape: (n_x, n_y, 3)

            # Apply Jones matrix transformation
            E = self._propagate_through_layer(E, director_layer, dz)

        print(f"  ✓ Propagation complete")

        # Apply output analyzer (crossed polarizer)
        E = self._apply_polarizer(E, self.params.analyzer_angle)

        # Calculate intensity
        intensity = self._calculate_intensity(E)

        # Add noise if requested
        if self.params.add_noise:
            intensity = self._add_noise(intensity)

        self.intensity_image = intensity
        self.electric_field = E

        print(f"  Intensity range: [{np.min(intensity):.3f}, {np.max(intensity):.3f}]")

        return intensity

    def _initialize_beam(self, n_x: int, n_y: int) -> np.ndarray:
        """
        Initialize electric field

        Returns:
            E: Shape (n_x, n_y, 2) - Complex electric field [Ex, Ey]
        """
        E = np.zeros((n_x, n_y, 2), dtype=complex)

        if self.params.beam_type == 'uniform':
            # Uniform illumination
            E[:, :, 0] = 1.0 + 0j
            E[:, :, 1] = 0.0 + 0j

        elif self.params.beam_type == 'gaussian':
            # Gaussian beam
            if self.params.beam_width is None:
                self.params.beam_width = min(n_x, n_y) / 4

            x = np.arange(n_x) - n_x / 2
            y = np.arange(n_y) - n_y / 2
            X, Y = np.meshgrid(x, y, indexing='ij')

            r2 = X**2 + Y**2
            gaussian = np.exp(-r2 / (2 * self.params.beam_width**2))

            E[:, :, 0] = gaussian
            E[:, :, 1] = 0

        return E

    def _apply_polarizer(self, E: np.ndarray, angle: float) -> np.ndarray:
        """
        Apply linear polarizer

        Polarizer axis: [cos(angle), sin(angle)]
        """
        axis = np.array([np.cos(angle), np.sin(angle)])

        # Project E onto polarizer axis
        E_proj = np.sum(E * axis, axis=-1)[..., np.newaxis]

        # New E field along polarizer direction
        E_out = E_proj * axis

        return E_out

    def _propagate_through_layer(self, E: np.ndarray,
                                 director: np.ndarray,
                                 dz: float) -> np.ndarray:
        """
        Propagate electric field through one layer using Jones calculus

        This is the KEY function that handles ARBITRARY director orientation!

        Args:
            E: Electric field (n_x, n_y, 2)
            director: Director field (n_x, n_y, 3) with components [nx, ny, nz]
            dz: Layer thickness

        Returns:
            E_new: Updated electric field
        """
        n_x, n_y, _ = director.shape

        # Extract director components
        nx = director[:, :, 0]
        ny = director[:, :, 1]
        nz = director[:, :, 2]

        # Handle NaN values (defect locations)
        # Set director to vertical at defects (minimal optical effect)
        defect_mask = np.isnan(nx)
        if np.any(defect_mask):
            nx = np.where(defect_mask, 0.0, nx)
            ny = np.where(defect_mask, 0.0, ny)
            nz = np.where(defect_mask, 1.0, nz)

        # Normalize in-plane components
        # l2 = nx² + ny² (in-plane magnitude squared)
        l2 = nx**2 + ny**2
        ll = np.sqrt(l2) + 1e-15  # Avoid division by zero

        nx_norm = nx / ll
        ny_norm = ny / ll

        # CRITICAL FORMULA: Effective refractive index for ARBITRARY orientation
        # From optics.py line 243
        # This accounts for director tilt out of the x-y plane!
        neff = self.params.no / np.sqrt(
            (self.params.no / self.params.ne)**2 * l2 + nz**2
        )

        # Phase retardation for this layer
        # γ = π · dz · (neff - no) / λ
        gamma = np.pi * dz / self.params.wavelength * (neff - self.params.no)

        # Jones matrix application (compact form from optics.py lines 245-247)
        # This is equivalent to: E_new = R^T · [[e^(iγ), 0], [0, e^(-iγ)]] · R · E
        # where R is rotation matrix to director coordinates

        # Project E onto director
        p = E[:, :, 0] * nx_norm + E[:, :, 1] * ny_norm

        # Apply phase factor
        factor = p * (1 - np.cos(2*gamma) + 1j*np.sin(2*gamma))

        # Update E field
        E_new = E.copy()
        E_new[:, :, 0] -= factor * nx_norm
        E_new[:, :, 1] -= factor * ny_norm

        return E_new

    def _calculate_intensity(self, E: np.ndarray) -> np.ndarray:
        """Calculate intensity from electric field"""
        return np.sum(np.abs(E)**2, axis=-1)

    def _add_noise(self, intensity: np.ndarray) -> np.ndarray:
        """Add realistic noise to intensity"""
        # Multiplicative noise (proportional to signal)
        noise = np.random.normal(0, self.params.noise_level, intensity.shape)
        noisy = intensity * (1 + noise)

        # Additive noise (detector noise)
        additive = np.random.normal(0, self.params.noise_level * 0.1, intensity.shape)
        noisy += additive

        return np.clip(noisy, 0, None)

    def plot_results(self, figsize=(15, 10)):
        """Plot optical simulation results"""
        if self.intensity_image is None:
            raise ValueError("No simulation results. Run simulate_fcpm() first.")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Panel 1: FCPM Intensity
        im1 = axes[0, 0].imshow(self.intensity_image.T, cmap='gray',
                               origin='lower', aspect='auto')
        axes[0, 0].set_title('FCPM Intensity\\n(Crossed Polarizers)')
        axes[0, 0].set_xlabel('X position')
        axes[0, 0].set_ylabel('Y position')
        plt.colorbar(im1, ax=axes[0, 0])

        # Panel 2: Intensity with enhanced contrast
        im2 = axes[0, 1].imshow(self.intensity_image.T**0.5, cmap='gray',
                               origin='lower', aspect='auto')
        axes[0, 1].set_title('Enhanced Contrast\\n(sqrt transform)')
        axes[0, 1].set_xlabel('X position')
        axes[0, 1].set_ylabel('Y position')
        plt.colorbar(im2, ax=axes[0, 1])

        # Panel 3: Intensity histogram
        axes[0, 2].hist(self.intensity_image.flatten(), bins=50,
                       alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Intensity Distribution')
        axes[0, 2].grid(True, alpha=0.3)

        # Panel 4: Horizontal line profiles
        n_y = self.intensity_image.shape[1]
        y_positions = [n_y//4, n_y//2, 3*n_y//4]

        for i, y_pos in enumerate(y_positions):
            profile = self.intensity_image[:, y_pos]
            axes[1, 0].plot(profile, alpha=0.8, label=f'y = {y_pos}')

        axes[1, 0].set_xlabel('X position')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].set_title('Horizontal Line Profiles')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Panel 5: Vertical line profiles
        n_x = self.intensity_image.shape[0]
        x_positions = [n_x//4, n_x//2, 3*n_x//4]

        for i, x_pos in enumerate(x_positions):
            profile = self.intensity_image[x_pos, :]
            axes[1, 1].plot(profile, alpha=0.8, label=f'x = {x_pos}')

        axes[1, 1].set_xlabel('Y position')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].set_title('Vertical Line Profiles')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Panel 6: Power spectrum
        fft_2d = np.fft.fft2(self.intensity_image)
        power_spectrum = np.abs(np.fft.fftshift(fft_2d))**2

        # Log scale for better visualization
        im6 = axes[1, 2].imshow(np.log10(power_spectrum.T + 1),
                               cmap='hot', origin='lower')
        axes[1, 2].set_title('Power Spectrum (log scale)')
        axes[1, 2].set_xlabel('kx')
        axes[1, 2].set_ylabel('ky')
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        return fig


def demo_optical_simulation():
    """Demonstrate optical simulation on toron structures"""
    print("\\n" + "="*60)
    print("PHASE 3 DEMO: Optical Simulation")
    print("="*60 + "\\n")

    # Load a toron structure
    from toron_simulator_3d import ToronSimulator3D

    print("Loading T3-1 toron structure...")
    sim = ToronSimulator3D.load_from_npz('toron_t3_1.npz')

    # Get physical size
    info = sim.get_grid_info()
    physical_size = info['physical_size']

    # Create optical simulator
    optical_params = OpticalParams(
        wavelength=0.55,      # Green light
        no=1.5,
        ne=1.7,
        polarizer_angle=0,    # 0 degrees
        analyzer_angle=np.pi/2,  # 90 degrees (crossed)
        add_noise=False
    )

    optical_sim = CrossPolarizerSimulator(optical_params)

    # Run simulation
    print("\\nRunning optical simulation...")
    intensity = optical_sim.simulate_fcpm(
        sim.director_field,
        physical_size
    )

    # Plot results
    print("\\nGenerating plots...")
    fig = optical_sim.plot_results()
    plt.savefig('fcpm_optical_simulation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: fcpm_optical_simulation.png")

    # Save intensity data
    np.save('fcpm_intensity.npy', intensity)
    print("✓ Saved: fcpm_intensity.npy")

    print("\\n" + "="*60)
    print("OPTICAL SIMULATION COMPLETE!")
    print("="*60)

    return optical_sim


if __name__ == "__main__":
    demo_optical_simulation()
    plt.show()
