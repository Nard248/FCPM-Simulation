import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter, rotate
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import json

@dataclass
class SimulationParams:
    """Parameters for FCPM simulation"""
    # Grid parameters
    n_z: int = 1100  # Points along Z (vertical)
    n_x: int = 55   # Points along X (horizontal)
    z_size: float = 11.0  # Physical size in Z direction
    x_size: float = 6.5   # Physical size in X direction
    
    # Pattern parameters
    n_bright_bands: int = 6
    n_dark_bands: int = 5
    pitch: float = 2.0  # Cholesteric pitch parameter
    phase_offset: float = 0.0
    
    # Noise parameters
    noise_level: float = 0.1  # 10% noise
    noise_type: str = 'gaussian'  # 'gaussian', 'poisson', 'salt_pepper'
    
    # Intensity parameters
    intensity_min: float = 0.0
    intensity_max: float = 1.0
    contrast: float = 1.0  # Contrast enhancement factor
    
    # Defect parameters
    include_defects: bool = False
    defect_types: List[str] = None
    defect_density: float = 0.1
    
    def __post_init__(self):
        if self.defect_types is None:
            self.defect_types = []

class EnhancedFCPMSimulator:
    """
    Enhanced FCPM simulator with presets, defects, and advanced features
    """
    
    def __init__(self, params: SimulationParams = None):
        self.params = params or SimulationParams()
        self.intensity_data = None
        self.defect_locations = []
        self.metadata = {}
        
    @classmethod
    def load_preset(cls, preset_name: str):
        """Load predefined simulation presets"""
        presets = {
            'basic': SimulationParams(),
            
            'high_resolution': SimulationParams(
                n_z=220, n_x=110, 
                noise_level=0.05
            ),
            
            'low_contrast': SimulationParams(
                contrast=0.5,
                noise_level=0.15
            ),
            
            'fine_pitch': SimulationParams(
                n_bright_bands=12,
                n_dark_bands=11,
                pitch=1.0
            ),
            
            'coarse_pitch': SimulationParams(
                n_bright_bands=3,
                n_dark_bands=2,
                pitch=4.0
            ),
            
            'with_dislocations': SimulationParams(
                include_defects=True,
                defect_types=['dislocation_b_half', 'dislocation_b_full'],
                defect_density=0.2
            ),
            
            'with_disclinations': SimulationParams(
                include_defects=True,
                defect_types=['tau_disclination', 'lambda_disclination'],
                defect_density=0.15
            ),
            
            'heavy_defects': SimulationParams(
                include_defects=True,
                defect_types=['dislocation_b_half', 'dislocation_b_full', 
                             'tau_disclination', 'lambda_disclination', 'kink'],
                defect_density=0.3,
                noise_level=0.12
            ),
            
            'experimental_like': SimulationParams(
                n_z=150, n_x=75,
                noise_level=0.08,
                include_defects=True,
                defect_types=['dislocation_b_half'],
                defect_density=0.1
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
            
        return cls(presets[preset_name])

    def generate_base_pattern(self) -> np.ndarray:
        """Generate base FCPM intensity pattern without defects"""
        # Create coordinate arrays
        z = np.linspace(0, self.params.z_size, self.params.n_z)

        # FIX: Correct periodicity for pseudo-vector nature
        # Director completes 2π rotation, but intensity modulates with π period
        theta = np.pi * z / (self.params.pitch / 2)  # π period instead of 2π

        intensity_1d = np.cos(theta) ** 4

        # Apply contrast enhancement
        intensity_1d = self._enhance_contrast(intensity_1d)

        # Create 2D pattern by repeating along X
        intensity_2d = np.tile(intensity_1d.reshape(-1, 1), (1, self.params.n_x))

        return intensity_2d
    
    def _enhance_contrast(self, intensity: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement"""
        if self.params.contrast != 1.0:
            # Apply power law transformation
            intensity = np.power(intensity, 1.0/self.params.contrast)
        return intensity

    def _smooth_intensity_profile(self, intensity: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to reduce sharp intensity variations
        Addresses issues 2 and 3: sharp variations and overly sharp high intensities
        """
        # Apply Gaussian smoothing along z-direction to smooth the intensity profile
        # Use different sigma for different intensity levels to preserve features
        # while smoothing sharp peaks

        smoothed = gaussian_filter(intensity, sigma=(1.5, 0.3), mode='reflect')

        # Additional smoothing for high intensity regions to reduce sharpness
        # Identify high intensity regions (> 0.7)
        high_intensity_mask = intensity > 0.7

        # Apply stronger smoothing to high intensity regions
        if np.any(high_intensity_mask):
            extra_smoothed = gaussian_filter(intensity, sigma=(2.5, 0.3), mode='reflect')
            # Blend based on intensity level
            blend_factor = np.clip((intensity - 0.7) / 0.3, 0, 1)
            smoothed = smoothed * (1 - blend_factor) + extra_smoothed * blend_factor

        return smoothed

    def add_noise(self, intensity: np.ndarray) -> np.ndarray:
        """Add different types of noise to the intensity pattern"""
        if self.params.noise_level <= 0:
            return intensity

        if self.params.noise_type == 'gaussian':
            # FIX: Add noise to ALL regions including zero-intensity lines
            noise_level = self.params.noise_level * 0.5  # Reduce noise

            # Add horizontal variation (0.95-1.0 as requested)
            horizontal_variation = np.random.uniform(0.95, 1.0, intensity.shape[1])

            # Apply horizontal variation
            for i in range(intensity.shape[0]):
                intensity[i, :] *= horizontal_variation

            # FIX: Add both multiplicative AND additive noise
            # This ensures zero-intensity regions also get noise
            multiplicative_noise = np.random.normal(0, 1, intensity.shape)
            scaled_mult_noise = multiplicative_noise * (intensity * noise_level)

            # Add small additive noise component (especially for low-intensity regions)
            additive_noise = np.random.normal(0, noise_level * 0.05, intensity.shape)

            noisy_intensity = intensity + scaled_mult_noise + additive_noise

        elif self.params.noise_type == 'poisson':
            # Keep existing poisson code but reduce intensity
            scaled_intensity = intensity * 50  # Reduced from 100
            noisy_scaled = np.random.poisson(scaled_intensity)
            noisy_intensity = noisy_scaled / 50.0

        elif self.params.noise_type == 'salt_pepper':
            # Keep existing salt_pepper code
            noisy_intensity = intensity.copy()
            salt_coords = np.random.random(intensity.shape) < self.params.noise_level / 2
            noisy_intensity[salt_coords] = 1.0
            pepper_coords = np.random.random(intensity.shape) < self.params.noise_level / 2
            noisy_intensity[pepper_coords] = 0.0
        else:
            raise ValueError(f"Unknown noise type: {self.params.noise_type}")

        # Ensure values stay in valid range
        return np.clip(noisy_intensity, self.params.intensity_min, self.params.intensity_max)
    
    def simulate(self) -> np.ndarray:
        """Main simulation method"""
        # Generate base pattern
        intensity = self.generate_base_pattern()

        # Add defects if requested
        if self.params.include_defects:
            intensity = self._add_defects(intensity)

        # Add noise
        intensity = self.add_noise(intensity)

        # FIX: Apply smoothing to reduce sharp intensity variations
        # This addresses issues 2 and 3 (sharp variations and high intensity peaks)
        intensity = self._smooth_intensity_profile(intensity)

        # Store results
        self.intensity_data = intensity
        self._generate_metadata()

        return intensity
    
    def _add_defects(self, intensity: np.ndarray) -> np.ndarray:
        """Add various types of defects to the intensity pattern"""
        defected_intensity = intensity.copy()
        self.defect_locations = []
        
        for defect_type in self.params.defect_types:
            if defect_type == 'dislocation_b_half':
                defected_intensity = self._add_dislocation_b_half(defected_intensity)
            elif defect_type == 'dislocation_b_full':
                defected_intensity = self._add_dislocation_b_full(defected_intensity)
            elif defect_type == 'tau_disclination':
                defected_intensity = self._add_tau_disclination(defected_intensity)
            elif defect_type == 'lambda_disclination':
                defected_intensity = self._add_lambda_disclination(defected_intensity)
            elif defect_type == 'kink':
                defected_intensity = self._add_kink_defect(defected_intensity)
            elif defect_type == 'lehmann_cluster':
                defected_intensity = self._add_lehmann_cluster(defected_intensity)
            elif defect_type == 'oily_streak':
                defected_intensity = self._add_oily_streak(defected_intensity)
            else:
                print(f"Warning: Unknown defect type '{defect_type}'")
        
        return defected_intensity
    
    def _add_dislocation_b_half(self, intensity: np.ndarray) -> np.ndarray:
        """Add b=p/2 dislocation defect"""
        # Randomly place dislocation
        if np.random.random() > self.params.defect_density:
            return intensity
            
        x_pos = np.random.randint(self.params.n_x//4, 3*self.params.n_x//4)
        z_start = np.random.randint(self.params.n_z//4, 3*self.params.n_z//4)
        
        # Create phase discontinuity
        for z in range(z_start, min(z_start + 20, self.params.n_z)):
            # Add intensity disruption around dislocation core
            core_width = 3
            for dx in range(-core_width, core_width+1):
                x_idx = x_pos + dx
                if 0 <= x_idx < self.params.n_x:
                    # Reduce intensity at core
                    intensity[z, x_idx] *= 0.3
                    
            # Create phase shift in surrounding area
            for x in range(max(0, x_pos-10), min(self.params.n_x, x_pos+10)):
                if x < x_pos:
                    # Shift stripes by half period on one side
                    shift = np.pi / (2 * self.params.pitch)
                    z_coord = z * self.params.z_size / self.params.n_z
                    new_theta = 2 * np.pi * z_coord / self.params.pitch + shift
                    intensity[z, x] = np.cos(new_theta)**4
        
        self.defect_locations.append({
            'type': 'dislocation_b_half',
            'position': (z_start, x_pos),
            'burgers_vector': 0.5
        })
        
        return intensity
    
    def _add_dislocation_b_full(self, intensity: np.ndarray) -> np.ndarray:
        """Add b=p dislocation defect"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        x_pos = np.random.randint(self.params.n_x//4, 3*self.params.n_x//4)
        z_start = np.random.randint(self.params.n_z//4, 3*self.params.n_z//4)
        
        # Create larger phase discontinuity
        for z in range(z_start, min(z_start + 25, self.params.n_z)):
            # Larger core region
            core_width = 4
            for dx in range(-core_width, core_width+1):
                x_idx = x_pos + dx
                if 0 <= x_idx < self.params.n_x:
                    intensity[z, x_idx] *= 0.2  # Stronger intensity reduction
            
            # Create full period shift
            for x in range(max(0, x_pos-15), min(self.params.n_x, x_pos+15)):
                if x < x_pos:
                    shift = np.pi / self.params.pitch  # Full period shift
                    z_coord = z * self.params.z_size / self.params.n_z
                    new_theta = 2 * np.pi * z_coord / self.params.pitch + shift
                    intensity[z, x] = np.cos(new_theta)**4
        
        self.defect_locations.append({
            'type': 'dislocation_b_full',
            'position': (z_start, x_pos),
            'burgers_vector': 1.0
        })
        
        return intensity
    
    def _add_tau_disclination(self, intensity: np.ndarray) -> np.ndarray:
        """Add τ disclination (singular defect)"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        x_pos = np.random.randint(5, self.params.n_x-5)
        z_pos = np.random.randint(5, self.params.n_z-5)
        
        # Create singular point defect
        radius = 8
        for dz in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                z_idx = z_pos + dz
                x_idx = x_pos + dx
                if 0 <= z_idx < self.params.n_z and 0 <= x_idx < self.params.n_x:
                    dist = np.sqrt(dx**2 + dz**2)
                    if dist <= radius:
                        # Create radial pattern around singularity
                        angle = np.arctan2(dz, dx)
                        intensity[z_idx, x_idx] *= (0.1 + 0.9 * np.cos(angle)**4)
        
        self.defect_locations.append({
            'type': 'tau_disclination',
            'position': (z_pos, x_pos),
            'winding_number': 0.5
        })
        
        return intensity
    
    def _add_lambda_disclination(self, intensity: np.ndarray) -> np.ndarray:
        """Add λ disclination (non-singular defect)"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        x_pos = np.random.randint(5, self.params.n_x-5)
        z_start = np.random.randint(5, self.params.n_z-20)
        
        # Create line defect
        for z in range(z_start, min(z_start + 15, self.params.n_z)):
            # Smooth intensity variation around line
            for dx in range(-6, 7):
                x_idx = x_pos + dx
                if 0 <= x_idx < self.params.n_x:
                    weight = np.exp(-dx**2 / 8)  # Gaussian profile
                    intensity[z, x_idx] *= (0.5 + 0.5 * weight)
        
        self.defect_locations.append({
            'type': 'lambda_disclination',
            'position': (z_start, x_pos),
            'length': 15
        })
        
        return intensity
    
    def _add_kink_defect(self, intensity: np.ndarray) -> np.ndarray:
        """Add kink along existing stripe"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        # Find existing stripe and create kink
        z_center = np.random.randint(10, self.params.n_z-10)
        x_start = np.random.randint(5, self.params.n_x//2)
        
        # Create zigzag pattern
        kink_length = 20
        for i, x in enumerate(range(x_start, min(x_start + kink_length, self.params.n_x))):
            z_offset = int(3 * np.sin(i * np.pi / 5))  # Sinusoidal kink
            z_idx = z_center + z_offset
            if 0 <= z_idx < self.params.n_z:
                # Modify stripe pattern
                intensity[z_idx, x] *= 0.7
                if z_idx + 1 < self.params.n_z:
                    intensity[z_idx + 1, x] *= 0.8
        
        self.defect_locations.append({
            'type': 'kink',
            'position': (z_center, x_start),
            'length': kink_length
        })
        
        return intensity
    
    def _add_lehmann_cluster(self, intensity: np.ndarray) -> np.ndarray:
        """Add Lehmann cluster (four disclinations)"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        x_center = np.random.randint(10, self.params.n_x-10)
        z_center = np.random.randint(10, self.params.n_z-10)
        
        # Four disclinations in square arrangement
        positions = [(z_center-3, x_center-3), (z_center-3, x_center+3),
                     (z_center+3, x_center-3), (z_center+3, x_center+3)]
        
        for z_pos, x_pos in positions:
            if 0 <= z_pos < self.params.n_z and 0 <= x_pos < self.params.n_x:
                # Small intensity modulation
                for dz in range(-2, 3):
                    for dx in range(-2, 3):
                        z_idx, x_idx = z_pos + dz, x_pos + dx
                        if 0 <= z_idx < self.params.n_z and 0 <= x_idx < self.params.n_x:
                            intensity[z_idx, x_idx] *= 0.6
        
        self.defect_locations.append({
            'type': 'lehmann_cluster',
            'position': (z_center, x_center),
            'cluster_size': 6
        })
        
        return intensity
    
    def _add_oily_streak(self, intensity: np.ndarray) -> np.ndarray:
        """Add oily streak defect"""
        if np.random.random() > self.params.defect_density:
            return intensity
            
        # Create wavy line across image
        z_start = np.random.randint(0, self.params.n_z//4)
        z_end = np.random.randint(3*self.params.n_z//4, self.params.n_z)
        
        for z in range(z_start, z_end):
            # Wavy pattern
            x_center = self.params.n_x//2 + int(10 * np.sin(z * np.pi / 20))
            
            # Create streak
            for dx in range(-4, 5):
                x_idx = x_center + dx
                if 0 <= x_idx < self.params.n_x:
                    weight = np.exp(-dx**2 / 4)
                    intensity[z, x_idx] *= (0.3 + 0.4 * weight)
        
        self.defect_locations.append({
            'type': 'oily_streak',
            'position': (z_start, self.params.n_x//2),
            'length': z_end - z_start
        })
        
        return intensity
    
    def _generate_metadata(self):
        """Generate metadata about the simulation"""
        self.metadata = {
            'simulation_params': {
                'grid_size': (self.params.n_z, self.params.n_x),
                'physical_size': (self.params.z_size, self.params.x_size),
                'pitch': self.params.pitch,
                'noise_level': self.params.noise_level,
                'noise_type': self.params.noise_type
            },
            'defects': {
                'include_defects': self.params.include_defects,
                'defect_types': self.params.defect_types,
                'defect_density': self.params.defect_density,
                'defect_locations': self.defect_locations
            },
            'statistics': {
                'mean_intensity': float(np.mean(self.intensity_data)),
                'std_intensity': float(np.std(self.intensity_data)),
                'min_intensity': float(np.min(self.intensity_data)),
                'max_intensity': float(np.max(self.intensity_data)),
                'n_defects': len(self.defect_locations)
            }
        }
    
    def plot_results(self, show_defects=True, figsize=(15, 10)):
        """Plot simulation results with optional defect highlighting"""
        if self.intensity_data is None:
            raise ValueError("No simulation data. Run simulate() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Main intensity pattern
        im1 = axes[0,0].imshow(self.intensity_data, cmap='gray', aspect='auto',
                              extent=[0, self.params.x_size, 0, self.params.z_size], 
                              origin='lower')
        axes[0,0].set_title('FCPM Pattern with Defects')
        axes[0,0].set_xlabel('X position')
        axes[0,0].set_ylabel('Z position')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Highlight defects
        if show_defects and self.defect_locations:
            for defect in self.defect_locations:
                z_pos, x_pos = defect['position']
                z_phys = z_pos * self.params.z_size / self.params.n_z
                x_phys = x_pos * self.params.x_size / self.params.n_x
                axes[0,0].plot(x_phys, z_phys, 'ro', markersize=8, alpha=0.7)
                axes[0,0].text(x_phys, z_phys+0.3, defect['type'][:4], 
                              color='red', fontsize=8, ha='center')
        
        # 1D intensity profile (averaged)
        z_coords = np.linspace(0, self.params.z_size, self.params.n_z)
        intensity_1d = np.mean(self.intensity_data, axis=1)
        axes[0,1].plot(z_coords, intensity_1d, 'b-', linewidth=2)
        axes[0,1].set_xlabel('Z position')
        axes[0,1].set_ylabel('Average Intensity')
        axes[0,1].set_title('1D Intensity Profile')
        axes[0,1].grid(True, alpha=0.3)
        
        # Histogram of intensities
        axes[0,2].hist(self.intensity_data.flatten(), bins=50, alpha=0.7, density=True)
        axes[0,2].set_xlabel('Intensity')
        axes[0,2].set_ylabel('Probability Density')
        axes[0,2].set_title('Intensity Distribution')
        axes[0,2].grid(True, alpha=0.3)
        
        # Horizontal line profiles at different Z positions
        z_positions = [self.params.n_z//4, self.params.n_z//2, 3*self.params.n_z//4]
        x_coords = np.linspace(0, self.params.x_size, self.params.n_x)
        
        for i, z_pos in enumerate(z_positions):
            profile = self.intensity_data[z_pos, :]
            z_phys = z_pos * self.params.z_size / self.params.n_z
            axes[1,0].plot(x_coords, profile, alpha=0.8, 
                          label=f'Z = {z_phys:.1f}')
        
        axes[1,0].set_xlabel('X position')
        axes[1,0].set_ylabel('Intensity')
        axes[1,0].set_title('Horizontal Line Profiles')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Defect statistics
        defect_types = {}
        for defect in self.defect_locations:
            dtype = defect['type']
            defect_types[dtype] = defect_types.get(dtype, 0) + 1
        
        if defect_types:
            types, counts = zip(*defect_types.items())
            axes[1,1].bar(range(len(types)), counts)
            axes[1,1].set_xticks(range(len(types)))
            axes[1,1].set_xticklabels(types, rotation=45, ha='right')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title('Defect Type Distribution')
        else:
            axes[1,1].text(0.5, 0.5, 'No Defects', ha='center', va='center',
                          transform=axes[1,1].transAxes, fontsize=14)
            axes[1,1].set_title('Defect Type Distribution')
        
        # Power spectrum analysis
        fft_1d = np.fft.fft(intensity_1d)
        freqs = np.fft.fftfreq(len(intensity_1d), d=self.params.z_size/self.params.n_z)
        power_spectrum = np.abs(fft_1d)**2
        
        # Plot only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_power = power_spectrum[:len(power_spectrum)//2]
        
        axes[1,2].plot(pos_freqs, pos_power)
        axes[1,2].set_xlabel('Spatial Frequency (1/unit)')
        axes[1,2].set_ylabel('Power')
        axes[1,2].set_title('Power Spectrum')
        axes[1,2].set_xlim(0, 2)  # Focus on low frequencies
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metadata
        self._print_simulation_summary()
    
    def plot_3d_director_field(self, n_z_planes=10, n_x_grid=15, n_y_grid=15, figsize=(16, 12)):
        """
        Visualize 3D director field showing orientation vectors in x-y planes along z-axis

        This addresses requirement 4: Show orientation vectors for x,y planes along z axis.
        For cholesteric liquid crystals, the director field rotates helically along z.

        Args:
            n_z_planes: Number of z-planes to visualize
            n_x_grid: Number of grid points in x direction
            n_y_grid: Number of grid points in y direction
            figsize: Figure size
        """
        if self.intensity_data is None:
            raise ValueError("No simulation data. Run simulate() first.")

        # Create z-planes to visualize
        z_indices = np.linspace(0, self.params.n_z - 1, n_z_planes, dtype=int)
        z_coords = z_indices * self.params.z_size / self.params.n_z

        # Create x-y grid for each plane
        x_grid = np.linspace(0, self.params.x_size, n_x_grid)
        y_grid = np.linspace(0, self.params.x_size, n_y_grid)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Calculate director angles at each z position
        # For cholesteric LC: β(z) = 2π*z/pitch + phase_offset
        angles = 2 * np.pi * z_coords / self.params.pitch + self.params.phase_offset

        # Create figure with multiple subplots
        fig = plt.figure(figsize=figsize)

        # 3D plot showing all planes
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        # Plot director field in each z-plane
        for i, (z_idx, z_pos, angle) in enumerate(zip(z_indices, z_coords, angles)):
            # Director components in x-y plane
            # The director rotates in the x-y plane with angle β
            nx = np.cos(angle) * np.ones_like(X)
            ny = np.sin(angle) * np.ones_like(Y)
            nz = np.zeros_like(X)

            # Color based on z position
            color = plt.cm.viridis(i / len(z_indices))

            # Plot quiver for this plane
            Z_plane = z_pos * np.ones_like(X)
            ax1.quiver(X, Y, Z_plane, nx, ny, nz,
                      length=0.3, color=color, alpha=0.7,
                      arrow_length_ratio=0.3, linewidth=1.5)

        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        ax1.set_title('3D Director Field\n(Cholesteric Helix)')
        ax1.set_xlim(0, self.params.x_size)
        ax1.set_ylim(0, self.params.x_size)
        ax1.set_zlim(0, self.params.z_size)

        # 2D projection showing director rotation along z
        ax2 = fig.add_subplot(2, 2, 2)

        # Show director angle as function of z
        z_fine = np.linspace(0, self.params.z_size, 200)
        angles_fine = 2 * np.pi * z_fine / self.params.pitch + self.params.phase_offset

        # Plot director components
        ax2.plot(z_fine, np.cos(angles_fine), 'r-', linewidth=2, label='nx (x-component)')
        ax2.plot(z_fine, np.sin(angles_fine), 'b-', linewidth=2, label='ny (y-component)')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)

        # Mark the sampled planes
        for z_pos in z_coords:
            ax2.axvline(z_pos, color='gray', alpha=0.3, linestyle=':')

        ax2.set_xlabel('Z position')
        ax2.set_ylabel('Director Component')
        ax2.set_title('Director Components vs Z')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Individual x-y plane views
        ax3 = fig.add_subplot(2, 2, 3)

        # Show several x-y planes side by side
        n_show = min(4, n_z_planes)
        show_indices = np.linspace(0, len(z_indices) - 1, n_show, dtype=int)

        for idx, i in enumerate(show_indices):
            z_idx = z_indices[i]
            z_pos = z_coords[i]
            angle = angles[i]

            # Director components
            nx = np.cos(angle)
            ny = np.sin(angle)

            # Plot as arrow in subplot
            offset = idx * 1.5
            ax3.arrow(offset, 0, nx * 0.5, ny * 0.5,
                     head_width=0.15, head_length=0.1,
                     fc=plt.cm.viridis(i / len(z_indices)),
                     ec=plt.cm.viridis(i / len(z_indices)),
                     linewidth=2, alpha=0.8)
            ax3.text(offset, -0.8, f'z={z_pos:.2f}',
                    ha='center', fontsize=10)

        ax3.set_xlim(-0.5, n_show * 1.5)
        ax3.set_ylim(-1, 1)
        ax3.set_aspect('equal')
        ax3.set_title('Director Orientation at Different Z-Planes')
        ax3.set_xlabel('Plane Index')
        ax3.set_ylabel('Director Components')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(-0.3, color='k', linestyle='-', alpha=0.3)

        # Intensity pattern overlay
        ax4 = fig.add_subplot(2, 2, 4)

        im = ax4.imshow(self.intensity_data, cmap='gray', aspect='auto',
                       extent=[0, self.params.x_size, 0, self.params.z_size],
                       origin='lower', alpha=0.7)

        # Overlay director orientations as arrows
        for z_pos, angle in zip(z_coords, angles):
            # Draw director orientation at center of image
            x_center = self.params.x_size / 2
            dx = np.cos(angle) * 0.5
            dy = 0  # Director rotates in x-y plane, but we're viewing x-z projection

            ax4.arrow(x_center, z_pos, dx, dy,
                     head_width=0.15, head_length=0.1,
                     fc='red', ec='red', alpha=0.8, linewidth=1.5)

        ax4.set_xlabel('X position')
        ax4.set_ylabel('Z position')
        ax4.set_title('Intensity Pattern with Director Field Overlay')
        plt.colorbar(im, ax=ax4, label='Intensity')

        plt.tight_layout()
        plt.suptitle('3D Director Field Visualization\n(Cholesteric Liquid Crystal)',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.show()

    def _print_simulation_summary(self):
        """Print summary of simulation results"""
        print("\n" + "="*50)
        print("ENHANCED FCPM SIMULATION SUMMARY")
        print("="*50)

        print(f"Grid Size: {self.params.n_z} × {self.params.n_x}")
        print(f"Physical Size: {self.params.z_size} × {self.params.x_size}")
        print(f"Pitch Parameter: {self.params.pitch}")
        print(f"Noise: {self.params.noise_type} ({self.params.noise_level*100:.1f}%)")

        stats = self.metadata['statistics']
        print(f"\nIntensity Statistics:")
        print(f"  Mean: {stats['mean_intensity']:.3f}")
        print(f"  Std:  {stats['std_intensity']:.3f}")
        print(f"  Range: [{stats['min_intensity']:.3f}, {stats['max_intensity']:.3f}]")

        if self.defect_locations:
            print(f"\nDefects Found: {len(self.defect_locations)}")
            defect_summary = {}
            for defect in self.defect_locations:
                dtype = defect['type']
                defect_summary[dtype] = defect_summary.get(dtype, 0) + 1

            for dtype, count in defect_summary.items():
                print(f"  {dtype}: {count}")
        else:
            print("\nNo Defects Present")
    
    def save_simulation(self, filename_base: str):
        """Save simulation data and metadata"""
        # Save intensity data
        np.save(f"{filename_base}_intensity.npy", self.intensity_data)
        
        # Save metadata as JSON
        with open(f"{filename_base}_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save parameters
        params_dict = {
            'n_z': self.params.n_z,
            'n_x': self.params.n_x,
            'z_size': self.params.z_size,
            'x_size': self.params.x_size,
            'pitch': self.params.pitch,
            'noise_level': self.params.noise_level,
            'noise_type': self.params.noise_type,
            'include_defects': self.params.include_defects,
            'defect_types': self.params.defect_types,
            'defect_density': self.params.defect_density
        }
        
        with open(f"{filename_base}_params.json", 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"Simulation saved as '{filename_base}_*'")
    
    @classmethod
    def load_simulation(cls, filename_base: str):
        """Load saved simulation"""
        # Load intensity data
        intensity_data = np.load(f"{filename_base}_intensity.npy")
        
        # Load parameters
        with open(f"{filename_base}_params.json", 'r') as f:
            params_dict = json.load(f)
        
        # Create params object
        params = SimulationParams(**params_dict)
        
        # Create simulator instance
        simulator = cls(params)
        simulator.intensity_data = intensity_data
        
        # Load metadata if available
        try:
            with open(f"{filename_base}_metadata.json", 'r') as f:
                simulator.metadata = json.load(f)
                simulator.defect_locations = simulator.metadata['defects']['defect_locations']
        except FileNotFoundError:
            print("Metadata file not found, generating new metadata")
            simulator._generate_metadata()
        
        return simulator

# Example usage and demonstration
if __name__ == "__main__":
    print("Enhanced FCPM Simulator - Demonstration")
    print("Available presets:", EnhancedFCPMSimulator.load_preset.__doc__)
    
    # Test different presets
    presets_to_test = ['heavy_defects']
    
    for preset_name in presets_to_test:
        print(f"\n{'='*20} Testing {preset_name} {'='*20}")
        
        # Load preset and simulate
        simulator = EnhancedFCPMSimulator.load_preset(preset_name)
        intensity_data = simulator.simulate()
        
        # Plot results
        simulator.plot_results()
        
        # Save simulation
        simulator.save_simulation(f"enhanced_fcpm_{preset_name}")
        
        break  # Remove this to test all presets
