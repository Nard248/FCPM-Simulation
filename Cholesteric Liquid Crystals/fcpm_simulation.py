import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate_fcpm_pattern():
    """
    Simulate FCPM intensity pattern for cholesteric liquid crystals
    Based on experimental data mimicking approach
    """
    
    # Parameters from instruction document
    n_bands = 11  # 6 bright + 5 dark bands
    points_per_band = 10
    n_z = n_bands * points_per_band  # 110 points along Z
    n_x = 55  # 55 points along X
    
    # Create Z coordinate array
    z = np.linspace(0, n_bands, n_z)
    
    # Method 1: Using cos^4 relationship
    # theta varies linearly with Z
    # theta = 0 at bright band centers, pi/2 at dark band centers
    theta = np.pi * z / 2  # This gives the right periodicity
    
    # FCPM intensity proportional to cos^4(theta)
    intensity_cos4 = np.cos(theta)**4
    
    # Method 2: Alternative approach using sine + square wave combination
    # Create sine wave component
    sine_component = np.sin(2 * np.pi * z / 2)  # Period = 2 bands
    
    # Create square wave component
    square_component = signal.square(2 * np.pi * z / 2)
    
    # Combine and normalize
    combined_signal = sine_component + 0.3 * square_component
    # Shift to positive values and normalize to 0-1
    intensity_combined = (combined_signal - np.min(combined_signal)) / (np.max(combined_signal) - np.min(combined_signal))
    
    # Choose the cos^4 method as it's more physically motivated
    intensity_1d = intensity_cos4
    
    # Create 2D intensity matrix by repeating along X
    intensity_2d = np.tile(intensity_1d.reshape(-1, 1), (1, n_x))
    
    # Add random noise (10% of intensity value)
    noise_level = 0.1
    noise = np.random.normal(0, 1, intensity_2d.shape)
    # Scale noise to be 10% of local intensity
    scaled_noise = noise * (intensity_2d * noise_level)
    
    # Add noise to intensity
    intensity_noisy = intensity_2d + scaled_noise
    
    # Ensure all values are in range [0, 1]
    intensity_final = np.clip(intensity_noisy, 0, 1)
    
    return intensity_final, z, intensity_1d

def plot_fcpm_results(intensity_2d, z, intensity_1d):
    """
    Plot the simulated FCPM results
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1D intensity profile
    ax1.plot(z, intensity_1d, 'b-', linewidth=2)
    ax1.set_xlabel('Z position')
    ax1.set_ylabel('Intensity')
    ax1.set_title('1D Intensity Profile (cos⁴ relationship)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2D intensity pattern
    im = ax2.imshow(intensity_2d, cmap='gray', aspect='auto', 
                    extent=[0, 6.5, 0, 11], origin='lower')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Z position')
    ax2.set_title('2D FCPM Pattern (with noise)')
    plt.colorbar(im, ax=ax2, label='Intensity')
    
    # Plot a few horizontal line profiles to show the stripe pattern
    z_positions = [20, 40, 60, 80, 100]  # Different Z positions
    for i, z_pos in enumerate(z_positions):
        ax3.plot(intensity_2d[z_pos, :], alpha=0.7, 
                label=f'Z = {z_pos/10:.1f}')
    
    ax3.set_xlabel('X position (pixels)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Horizontal Line Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_pattern_statistics(intensity_2d):
    """
    Analyze the statistical properties of the simulated pattern
    """
    print("Pattern Statistics:")
    print(f"Shape: {intensity_2d.shape}")
    print(f"Intensity range: {np.min(intensity_2d):.3f} - {np.max(intensity_2d):.3f}")
    print(f"Mean intensity: {np.mean(intensity_2d):.3f}")
    print(f"Standard deviation: {np.std(intensity_2d):.3f}")
    
    # Count bright and dark regions
    threshold = 0.5
    bright_pixels = np.sum(intensity_2d > threshold)
    dark_pixels = np.sum(intensity_2d <= threshold)
    total_pixels = intensity_2d.size
    
    print(f"Bright pixels (>0.5): {bright_pixels} ({100*bright_pixels/total_pixels:.1f}%)")
    print(f"Dark pixels (≤0.5): {dark_pixels} ({100*dark_pixels/total_pixels:.1f}%)")

# Main execution
if __name__ == "__main__":
    # Generate the simulated FCPM pattern
    intensity_2d, z_coords, intensity_1d = simulate_fcpm_pattern()
    
    # Plot results
    plot_fcpm_results(intensity_2d, z_coords, intensity_1d)
    
    # Analyze pattern
    analyze_pattern_statistics(intensity_2d)
    
    print("\nStep 1 Complete: FCPM pattern simulation")
    print("Next step: Implement director field reconstruction algorithm")
    
    # Save data for further processing
    np.save('fcpm_simulated_intensity.npy', intensity_2d)
    np.save('z_coordinates.npy', z_coords)
    
    print("Data saved as 'fcmp_simulated_intensity.npy' and 'z_coordinates.npy'")
