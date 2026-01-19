import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

class FCPMReconstruction:
    """
    Class for reconstructing director field from FCPM intensity data
    
    Main challenge: Ambiguity in director field reconstruction
    I ∝ cos⁴(β) where β is angle between polarization and director
    """
    
    def __init__(self, intensity_data):
        """
        Initialize with FCPM intensity data
        
        Parameters:
        intensity_data: 2D numpy array of FCPM intensities
        """
        self.intensity = intensity_data
        self.n_z, self.n_x = intensity_data.shape
        self.reconstructed_angles = None
        
    def extract_1d_profile(self, method='average'):
        """
        Extract 1D intensity profile along Z direction
        
        Methods:
        - 'average': Average across all X positions
        - 'center': Take center line
        - 'median': Median across X positions
        """
        if method == 'average':
            profile = np.mean(self.intensity, axis=1)
        elif method == 'center':
            center_x = self.n_x // 2
            profile = self.intensity[:, center_x]
        elif method == 'median':
            profile = np.median(self.intensity, axis=1)
        else:
            raise ValueError("Method must be 'average', 'center', or 'median'")
            
        return profile
    
    def reconstruct_angle_simple(self, intensity_profile, smooth=True):
        """
        Simple reconstruction: β = arccos((I)^(1/4))
        
        This is the direct inversion of I = cos⁴(β)
        Main ambiguity: cos⁴(β) = cos⁴(π - β) = cos⁴(β + π)
        """
        # Ensure intensity values are in valid range
        intensity_clipped = np.clip(intensity_profile, 1e-6, 1.0)
        
        # Direct inversion: β = arccos(I^(1/4))
        angles = np.arccos(intensity_clipped**(1/4))
        
        if smooth:
            # Apply gentle smoothing to reduce noise effects
            angles = gaussian_filter1d(angles, sigma=1.0)
            
        return angles
    
    def resolve_ambiguity_continuity(self, angles):
        """
        Resolve angle ambiguity using continuity assumption
        
        The director field should vary continuously, so we choose
        the branch that minimizes discontinuities
        """
        resolved_angles = angles.copy()
        
        for i in range(1, len(angles)):
            # Check both possibilities: θ and π - θ
            option1 = angles[i]
            option2 = np.pi - angles[i]
            
            # Choose the one closer to previous angle
            diff1 = abs(option1 - resolved_angles[i-1])
            diff2 = abs(option2 - resolved_angles[i-1])
            
            if diff2 < diff1:
                resolved_angles[i] = option2
                
        return resolved_angles
    
    def reconstruct_angle_fitting(self, intensity_profile):
        """
        Reconstruction by fitting to expected helical pattern
        
        Assume: β(z) = 2πz/p + φ₀
        where p is pitch and φ₀ is phase offset
        """
        z_coords = np.linspace(0, 11, len(intensity_profile))
        
        def model_intensity(params):
            """Model: I = cos⁴(2πz/p + φ₀)"""
            pitch, phase = params
            beta = 2 * np.pi * z_coords / pitch + phase
            return np.cos(beta)**4
        
        def objective(params):
            """Minimize difference between model and data"""
            model = model_intensity(params)
            return np.sum((model - intensity_profile)**2)
        
        # Initial guess: pitch = 2 (from 6 bright bands in 11 units)
        initial_params = [2.0, 0.0]
        
        # Optimize
        result = minimize(objective, initial_params, 
                         bounds=[(1.5, 3.0), (-np.pi, np.pi)])
        
        if result.success:
            optimal_pitch, optimal_phase = result.x
            z_coords = np.linspace(0, 11, len(intensity_profile))
            fitted_angles = 2 * np.pi * z_coords / optimal_pitch + optimal_phase
            return fitted_angles, optimal_pitch, optimal_phase
        else:
            print("Fitting failed, using simple reconstruction")
            return self.reconstruct_angle_simple(intensity_profile), None, None
    
    def reconstruct_director_field(self, method='fitting'):
        """
        Main reconstruction method
        
        Methods:
        - 'simple': Direct inversion with continuity resolution
        - 'fitting': Fit to helical model
        """
        # Extract 1D intensity profile
        intensity_1d = self.extract_1d_profile(method='average')
        
        if method == 'simple':
            # Simple reconstruction
            angles = self.reconstruct_angle_simple(intensity_1d)
            angles = self.resolve_ambiguity_continuity(angles)
            self.reconstructed_angles = angles
            return angles, None, None
            
        elif method == 'fitting':
            # Fitting reconstruction
            angles, pitch, phase = self.reconstruct_angle_fitting(intensity_1d)
            self.reconstructed_angles = angles
            return angles, pitch, phase
        
        else:
            raise ValueError("Method must be 'simple' or 'fitting'")
    
    def plot_reconstruction_results(self, angles, pitch=None, phase=None):
        """
        Plot reconstruction results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original intensity pattern
        im1 = ax1.imshow(self.intensity, cmap='gray', aspect='auto',
                        extent=[0, 6.5, 0, 11], origin='lower')
        ax1.set_title('Original FCPM Pattern')
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Z position')
        plt.colorbar(im1, ax=ax1)
        
        # 1D intensity profile
        z_coords = np.linspace(0, 11, self.n_z)
        intensity_1d = self.extract_1d_profile()
        ax2.plot(z_coords, intensity_1d, 'b-', linewidth=2, label='Experimental')
        
        # Reconstructed intensity from angles
        if angles is not None:
            reconstructed_intensity = np.cos(angles)**4
            ax2.plot(z_coords, reconstructed_intensity, 'r--', linewidth=2, 
                    label='Reconstructed')
        
        ax2.set_xlabel('Z position')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Intensity Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reconstructed angles
        if angles is not None:
            ax3.plot(z_coords, angles, 'g-', linewidth=2)
            ax3.set_xlabel('Z position')
            ax3.set_ylabel('Angle β (radians)')
            ax3.set_title('Reconstructed Director Angles')
            ax3.grid(True, alpha=0.3)
            
            if pitch is not None:
                ax3.text(0.1, 0.9, f'Fitted pitch: {pitch:.2f}', 
                        transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Director field visualization (2D projection)
        if angles is not None:
            # Create 2D director field
            director_x = np.cos(angles)
            director_y = np.sin(angles)
            
            # Plot as vector field
            z_sample = z_coords[::5]  # Sample every 5th point
            x_sample = np.linspace(0, 6.5, 10)
            Z_grid, X_grid = np.meshgrid(z_sample, x_sample, indexing='ij')
            
            # Director components at sampled points
            U = np.tile(director_x[::5].reshape(-1, 1), (1, 10))
            V = np.tile(director_y[::5].reshape(-1, 1), (1, 10))
            
            ax4.quiver(X_grid, Z_grid, U, V, scale=20, alpha=0.7)
            ax4.set_xlabel('X position')
            ax4.set_ylabel('Z position')
            ax4.set_title('Reconstructed Director Field')
            ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def validate_reconstruction(self, true_angles=None):
        """
        Validate reconstruction if true angles are known
        """
        if true_angles is None or self.reconstructed_angles is None:
            print("Cannot validate: missing true angles or reconstruction")
            return
            
        # Calculate error metrics
        mse = np.mean((self.reconstructed_angles - true_angles)**2)
        mae = np.mean(np.abs(self.reconstructed_angles - true_angles))
        
        print(f"Reconstruction Validation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Plot comparison
        z_coords = np.linspace(0, 11, len(true_angles))
        plt.figure(figsize=(10, 6))
        plt.plot(z_coords, true_angles, 'b-', linewidth=2, label='True angles')
        plt.plot(z_coords, self.reconstructed_angles, 'r--', linewidth=2, label='Reconstructed')
        plt.xlabel('Z position')
        plt.ylabel('Angle β (radians)')
        plt.title('Reconstruction Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Load simulated data
    try:
        intensity_data = np.load('fcpm_simulated_intensity.npy')
        print("Loaded simulated FCPM data")
    except FileNotFoundError:
        print("Run the simulation code first to generate data")
        exit()
    
    # Initialize reconstruction
    reconstructor = FCPMReconstruction(intensity_data)
    
    # Test both reconstruction methods
    print("\n=== Testing Simple Reconstruction ===")
    angles_simple, _, _ = reconstructor.reconstruct_director_field(method='simple')
    
    print("\n=== Testing Fitting Reconstruction ===")
    angles_fitted, pitch, phase = reconstructor.reconstruct_director_field(method='fitting')
    
    if pitch is not None:
        print(f"Fitted parameters: pitch = {pitch:.3f}, phase = {phase:.3f}")
    
    # Plot results
    reconstructor.plot_reconstruction_results(angles_fitted, pitch, phase)
    
    print("\nStep 2 Framework Complete: Director field reconstruction")
    print("Main challenges addressed:")
    print("1. Ambiguity resolution using continuity assumption")
    print("2. Alternative fitting approach for helical patterns")
    print("3. Validation and error analysis framework")
