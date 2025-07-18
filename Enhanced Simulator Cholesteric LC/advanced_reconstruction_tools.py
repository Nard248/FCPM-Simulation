import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import gaussian_filter1d, label, center_of_mass
from scipy.signal import find_peaks, peak_widths
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json


def simple_dbscan_clustering(points, eps=0.5, min_samples=2):
    """Simple clustering function to replace sklearn.DBSCAN"""
    if len(points) == 0:
        return np.array([])

    points = np.array(points)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    clusters = np.full(n_points, -1)  # -1 means noise
    cluster_id = 0

    for i in range(n_points):
        if visited[i]:
            continue

        visited[i] = True

        # Find neighbors
        distances = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        neighbors = np.where(distances <= eps)[0]

        if len(neighbors) < min_samples:
            continue  # Mark as noise

        # Start new cluster
        clusters[i] = cluster_id
        neighbor_queue = list(neighbors)

        while neighbor_queue:
            neighbor_idx = neighbor_queue.pop(0)

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True

                # Find new neighbors
                new_distances = np.sqrt(np.sum((points - points[neighbor_idx]) ** 2, axis=1))
                new_neighbors = np.where(new_distances <= eps)[0]

                if len(new_neighbors) >= min_samples:
                    neighbor_queue.extend(new_neighbors)

            if clusters[neighbor_idx] == -1:
                clusters[neighbor_idx] = cluster_id

        cluster_id += 1

    return clusters


@dataclass
class ReconstructionResult:
    """Container for reconstruction results"""
    angles: np.ndarray
    confidence: np.ndarray
    fitted_params: Dict
    defect_detections: List[Dict]
    reconstruction_method: str
    error_metrics: Dict
    metadata: Dict


class AdvancedFCPMReconstructor:
    """
    Advanced FCPM reconstruction with defect detection and analysis
    """

    def __init__(self, intensity_data: np.ndarray, z_coordinates: np.ndarray = None):
        self.intensity = intensity_data
        self.n_z, self.n_x = intensity_data.shape

        if z_coordinates is None:
            self.z_coords = np.linspace(0, 11, self.n_z)
        else:
            self.z_coords = z_coordinates

        self.results = None

    def preprocess_data(self, method='robust'):
        """
        Preprocess intensity data to improve reconstruction

        Methods:
        - 'basic': Simple noise reduction
        - 'robust': Outlier removal + smoothing
        - 'adaptive': Adaptive filtering based on local statistics
        """
        processed = self.intensity.copy()

        if method == 'basic':
            # Simple Gaussian filtering
            processed = gaussian_filter1d(processed, sigma=0.5, axis=0)

        elif method == 'robust':
            # Remove outliers using percentile clipping
            for x in range(self.n_x):
                column = processed[:, x]
                q1, q3 = np.percentile(column, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                processed[:, x] = np.clip(column, lower_bound, upper_bound)

            # Apply gentle smoothing
            processed = gaussian_filter1d(processed, sigma=0.7, axis=0)

        elif method == 'adaptive':
            # Adaptive filtering based on local variance
            for x in range(self.n_x):
                column = processed[:, x]
                # Calculate local variance with sliding window
                window_size = 5
                local_var = np.zeros_like(column)

                for i in range(self.n_z):
                    start = max(0, i - window_size // 2)
                    end = min(self.n_z, i + window_size // 2 + 1)
                    local_var[i] = np.var(column[start:end])

                # Apply stronger smoothing where variance is high (likely noise)
                sigma = 0.3 + 2.0 * (local_var / np.max(local_var))
                processed[:, x] = gaussian_filter1d(column, sigma=np.mean(sigma))

        return processed

    def extract_intensity_profile(self, method='weighted_average', exclude_defects=True):
        """
        Extract 1D intensity profile with various strategies

        Methods:
        - 'simple_average': Simple mean across X
        - 'weighted_average': Weighted by local consistency
        - 'median': Robust median
        - 'mode_analysis': Find most common pattern
        """
        if method == 'simple_average':
            return np.mean(self.intensity, axis=1)

        elif method == 'weighted_average':
            # Weight by consistency across X direction
            weights = np.zeros((self.n_z, self.n_x))

            for z in range(self.n_z):
                row = self.intensity[z, :]
                # Weight by inverse of local deviation
                mean_val = np.mean(row)
                deviations = np.abs(row - mean_val)
                weights[z, :] = 1.0 / (deviations + 0.1)  # Avoid division by zero

            # Weighted average
            weighted_sum = np.sum(self.intensity * weights, axis=1)
            weight_sum = np.sum(weights, axis=1)

            return weighted_sum / weight_sum

        elif method == 'median':
            return np.median(self.intensity, axis=1)

        elif method == 'mode_analysis':
            # Find the most representative profile
            profiles = []
            for x in range(0, self.n_x, 5):  # Sample every 5th column
                profiles.append(self.intensity[:, x])

            profiles = np.array(profiles)

            # Find profile closest to median profile
            median_profile = np.median(profiles, axis=0)
            distances = np.sum((profiles - median_profile) ** 2, axis=1)
            best_idx = np.argmin(distances)

            return profiles[best_idx]

    def detect_defects(self, threshold_factor=2.0):
        """
        Detect defects in the intensity pattern

        Returns list of detected defects with positions and types
        """
        defects = []

        # Method 1: Detect intensity anomalies
        defects.extend(self._detect_intensity_anomalies(threshold_factor))

        # Method 2: Detect stripe discontinuities
        defects.extend(self._detect_stripe_discontinuities())

        # Method 3: Detect periodic pattern breaks
        defects.extend(self._detect_pattern_breaks())

        return defects

    def _detect_intensity_anomalies(self, threshold_factor=2.0):
        """Detect regions with anomalous intensity values"""
        defects = []

        # Calculate local statistics
        mean_intensity = np.mean(self.intensity)
        std_intensity = np.std(self.intensity)

        # Find pixels significantly different from expected
        threshold = threshold_factor * std_intensity
        anomalous = np.abs(self.intensity - mean_intensity) > threshold

        # Group connected anomalous regions
        labeled_regions, num_regions = label(anomalous)

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            if np.sum(region_mask) > 10:  # Minimum size threshold
                # Find center of mass
                z_center, x_center = center_of_mass(region_mask)

                defects.append({
                    'type': 'intensity_anomaly',
                    'position': (int(z_center), int(x_center)),
                    'size': int(np.sum(region_mask)),
                    'confidence': min(1.0, np.sum(region_mask) / 50.0)
                })

        return defects

    def _detect_stripe_discontinuities(self):
        """Detect discontinuities in stripe patterns"""
        defects = []

        # Analyze each row for discontinuities
        for z in range(self.n_z):
            row = self.intensity[z, :]

            # Find sharp changes in intensity
            gradient = np.abs(np.gradient(row))
            mean_grad = np.mean(gradient)
            std_grad = np.std(gradient)

            # Find points with unusually high gradient
            high_grad_points = np.where(gradient > mean_grad + 3 * std_grad)[0]

            # Group nearby points
            if len(high_grad_points) > 0:
                groups = []
                current_group = [high_grad_points[0]]

                for point in high_grad_points[1:]:
                    if point - current_group[-1] <= 3:  # Close points
                        current_group.append(point)
                    else:
                        groups.append(current_group)
                        current_group = [point]
                groups.append(current_group)

                # Convert groups to defects
                for group in groups:
                    if len(group) >= 2:  # Minimum width
                        x_center = int(np.mean(group))
                        defects.append({
                            'type': 'stripe_discontinuity',
                            'position': (z, x_center),
                            'width': len(group),
                            'confidence': min(1.0, len(group) / 10.0)
                        })

        return defects

    def _detect_pattern_breaks(self):
        """Detect breaks in periodic patterns"""
        defects = []

        # Extract 1D profile
        profile = self.extract_intensity_profile(method='median')

        # Find expected periodicity
        fft_profile = np.fft.fft(profile)
        freqs = np.fft.fftfreq(len(profile))

        # Find dominant frequency (excluding DC component)
        magnitude = np.abs(fft_profile[1:len(fft_profile) // 2])
        dominant_freq_idx = np.argmax(magnitude) + 1
        dominant_freq = freqs[dominant_freq_idx]
        expected_period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else len(profile)

        # Look for phase jumps indicating defects
        # Compute local phase using Hilbert transform equivalent
        analytic_signal = np.fft.ifft(fft_profile * (freqs == dominant_freq))
        phase = np.angle(analytic_signal)

        # Find phase discontinuities
        phase_diff = np.diff(np.unwrap(phase))
        phase_jumps = np.where(np.abs(phase_diff) > np.pi / 2)[0]

        for jump_idx in phase_jumps:
            z_pos = jump_idx
            x_pos = self.n_x // 2  # Assume center

            defects.append({
                'type': 'phase_discontinuity',
                'position': (z_pos, x_pos),
                'magnitude': float(np.abs(phase_diff[jump_idx])),
                'confidence': min(1.0, np.abs(phase_diff[jump_idx]) / np.pi)
            })

        return defects

    def reconstruct_with_defects(self, method='adaptive_fitting'):
        """
        Reconstruct director field accounting for detected defects

        Methods:
        - 'adaptive_fitting': Fit piecewise models around defects
        - 'robust_inversion': Robust inversion with outlier handling
        - 'global_optimization': Global optimization with defect penalties
        """
        # Preprocess data
        processed_intensity = self.preprocess_data(method='robust')

        # Extract profile
        intensity_1d = self.extract_intensity_profile(method='weighted_average')

        # Detect defects
        detected_defects = self.detect_defects()

        if method == 'adaptive_fitting':
            angles, params = self._adaptive_piecewise_fitting(intensity_1d, detected_defects)
        elif method == 'robust_inversion':
            angles, params = self._robust_inversion(intensity_1d, detected_defects)
        elif method == 'global_optimization':
            angles, params = self._global_optimization(intensity_1d, detected_defects)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate confidence estimates
        confidence = self._calculate_confidence(intensity_1d, angles, detected_defects)

        # Calculate error metrics
        error_metrics = self._calculate_error_metrics(intensity_1d, angles)

        # Create result object
        self.results = ReconstructionResult(
            angles=angles,
            confidence=confidence,
            fitted_params=params,
            defect_detections=detected_defects,
            reconstruction_method=method,
            error_metrics=error_metrics,
            metadata={
                'preprocessing': 'robust',
                'profile_method': 'weighted_average',
                'n_defects_detected': len(detected_defects),
                'grid_size': (self.n_z, self.n_x)
            }
        )

        return self.results

    def _adaptive_piecewise_fitting(self, intensity_1d, defects):
        """Fit piecewise helical models around defects"""
        # Sort defects by position
        defect_positions = [d['position'][0] for d in defects]
        defect_positions.sort()

        # Create segments between defects
        segment_boundaries = [0] + defect_positions + [len(intensity_1d)]
        segments = []

        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]
            segments.append((start, end))

        # Fit each segment separately
        angles = np.zeros(len(intensity_1d))
        fitted_params = {'segments': []}

        for start, end in segments:
            if end - start < 10:  # Skip very short segments
                continue

            segment_z = self.z_coords[start:end]
            segment_intensity = intensity_1d[start:end]

            # Fit helical model to segment
            def model(params):
                pitch, phase, amplitude, offset = params
                theta = 2 * np.pi * segment_z / pitch + phase
                return amplitude * np.cos(theta) ** 4 + offset

            def objective(params):
                if params[0] <= 0 or params[2] <= 0:  # Invalid parameters
                    return 1e6
                predicted = model(params)
                return np.sum((predicted - segment_intensity) ** 2)

            # Initial guess
            initial_params = [2.0, 0.0, 1.0, 0.0]

            # Optimize
            bounds = [(0.5, 5.0), (-np.pi, np.pi), (0.1, 2.0), (-0.5, 0.5)]
            result = minimize(objective, initial_params, bounds=bounds)

            if result.success:
                segment_angles = 2 * np.pi * segment_z / result.x[0] + result.x[1]
                angles[start:end] = segment_angles

                fitted_params['segments'].append({
                    'range': (start, end),
                    'pitch': result.x[0],
                    'phase': result.x[1],
                    'amplitude': result.x[2],
                    'offset': result.x[3]
                })

        return angles, fitted_params

    def _robust_inversion(self, intensity_1d, defects):
        """Robust inversion with outlier handling"""
        # Create mask for defect regions
        defect_mask = np.zeros(len(intensity_1d), dtype=bool)
        for defect in defects:
            z_pos = defect['position'][0]
            # Mark region around defect
            start = max(0, z_pos - 5)
            end = min(len(intensity_1d), z_pos + 5)
            defect_mask[start:end] = True

        # Direct inversion for non-defect regions
        angles = np.zeros(len(intensity_1d))
        clean_intensity = np.clip(intensity_1d, 1e-6, 1.0)

        # Invert cos^4 relationship
        angles = np.arccos(clean_intensity ** (1 / 4))

        # Interpolate through defect regions
        if np.any(defect_mask):
            clean_indices = np.where(~defect_mask)[0]
            defect_indices = np.where(defect_mask)[0]

            if len(clean_indices) > 0:
                angles[defect_indices] = np.interp(defect_indices, clean_indices, angles[clean_indices])

        # Apply continuity constraint
        angles = self._enforce_continuity(angles)

        fitted_params = {
            'method': 'robust_inversion',
            'n_defect_points': int(np.sum(defect_mask)),
            'interpolated_fraction': float(np.sum(defect_mask) / len(defect_mask))
        }

        return angles, fitted_params

    def _global_optimization(self, intensity_1d, defects):
        """Global optimization with defect penalties"""

        def global_objective(params):
            """Objective function for global optimization"""
            n_params = len(params)
            n_segments = n_params // 4  # Each segment has 4 parameters

            total_error = 0
            angles = np.zeros(len(intensity_1d))

            # Process each segment
            segment_length = len(intensity_1d) // n_segments

            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(intensity_1d))

                param_start = i * 4
                pitch = params[param_start]
                phase = params[param_start + 1]
                amplitude = params[param_start + 2]
                offset = params[param_start + 3]

                # Skip invalid parameters
                if pitch <= 0 or amplitude <= 0:
                    return 1e6

                # Calculate model for this segment
                z_segment = self.z_coords[start_idx:end_idx]
                theta = 2 * np.pi * z_segment / pitch + phase
                model_intensity = amplitude * np.cos(theta) ** 4 + offset

                # Calculate error
                segment_intensity = intensity_1d[start_idx:end_idx]
                error = np.sum((model_intensity - segment_intensity) ** 2)
                total_error += error

                # Store angles
                angles[start_idx:end_idx] = theta

            # Add penalty for discontinuities between segments
            continuity_penalty = 0
            for i in range(n_segments - 1):
                boundary_idx = (i + 1) * segment_length
                if boundary_idx < len(angles):
                    angle_jump = abs(angles[boundary_idx] - angles[boundary_idx - 1])
                    continuity_penalty += angle_jump ** 2

            return total_error + 10.0 * continuity_penalty

        # Set up optimization
        n_segments = max(1, len(defects) + 1)
        n_params = n_segments * 4

        # Parameter bounds: [pitch, phase, amplitude, offset] for each segment
        bounds = []
        for _ in range(n_segments):
            bounds.extend([(0.5, 5.0), (-np.pi, np.pi), (0.1, 2.0), (-0.5, 0.5)])

        # Run global optimization
        result = differential_evolution(global_objective, bounds, maxiter=100, seed=42)

        if result.success:
            # Reconstruct angles from optimal parameters
            angles = np.zeros(len(intensity_1d))
            segment_length = len(intensity_1d) // n_segments

            fitted_params = {'segments': []}

            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(intensity_1d))

                param_start = i * 4
                pitch = result.x[param_start]
                phase = result.x[param_start + 1]

                z_segment = self.z_coords[start_idx:end_idx]
                theta = 2 * np.pi * z_segment / pitch + phase
                angles[start_idx:end_idx] = theta

                fitted_params['segments'].append({
                    'pitch': pitch,
                    'phase': phase,
                    'amplitude': result.x[param_start + 2],
                    'offset': result.x[param_start + 3]
                })

            fitted_params['optimization_success'] = True
            fitted_params['final_error'] = result.fun
        else:
            # Fallback to simple reconstruction
            angles = np.arccos(np.clip(intensity_1d, 1e-6, 1.0) ** (1 / 4))
            fitted_params = {'optimization_success': False, 'fallback_used': True}

        return angles, fitted_params

    def _enforce_continuity(self, angles):
        """Enforce continuity in angle reconstruction"""
        continuous_angles = angles.copy()

        for i in range(1, len(angles)):
            # Check for large jumps
            diff = angles[i] - continuous_angles[i - 1]

            # Try alternative branches: θ, π-θ, θ+π, 2π-θ
            alternatives = [
                angles[i],
                np.pi - angles[i],
                angles[i] + np.pi,
                2 * np.pi - angles[i]
            ]

            # Choose alternative with smallest jump
            diffs = [abs(alt - continuous_angles[i - 1]) for alt in alternatives]
            best_idx = np.argmin(diffs)
            continuous_angles[i] = alternatives[best_idx]

        return continuous_angles

    def _calculate_confidence(self, intensity_1d, angles, defects):
        """Calculate confidence estimates for reconstruction"""
        confidence = np.ones(len(angles))

        # Reduce confidence near defects
        for defect in defects:
            z_pos = defect['position'][0]
            defect_confidence = defect.get('confidence', 0.5)

            # Gaussian reduction around defect
            distances = np.abs(np.arange(len(angles)) - z_pos)
            gaussian_weight = np.exp(-distances ** 2 / (2 * 5 ** 2))  # σ = 5
            confidence *= (1 - 0.8 * gaussian_weight * (1 - defect_confidence))

        # Reduce confidence where model fit is poor
        predicted_intensity = np.cos(angles) ** 4
        residuals = np.abs(predicted_intensity - intensity_1d)
        residual_threshold = 2 * np.std(residuals)

        poor_fit_mask = residuals > residual_threshold
        confidence[poor_fit_mask] *= 0.5

        return np.clip(confidence, 0.1, 1.0)

    def _calculate_error_metrics(self, intensity_1d, angles):
        """Calculate various error metrics"""
        predicted_intensity = np.cos(angles) ** 4

        residuals = predicted_intensity - intensity_1d

        metrics = {
            'mse': float(np.mean(residuals ** 2)),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals))),
            'max_error': float(np.max(np.abs(residuals))),
            'r_squared': float(1 - np.sum(residuals ** 2) / np.sum((intensity_1d - np.mean(intensity_1d)) ** 2))
        }

        return metrics

    def plot_reconstruction_analysis(self, figsize=(16, 12)):
        """Comprehensive plot of reconstruction results"""
        if self.results is None:
            raise ValueError("No reconstruction results. Run reconstruct_with_defects() first.")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Original intensity pattern with defects marked
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.intensity, cmap='gray', aspect='auto',
                         extent=[0, 6.5, 0, 11], origin='lower')

        # Mark detected defects
        for defect in self.results.defect_detections:
            z_pos, x_pos = defect['position']
            z_phys = z_pos * 11 / self.n_z
            x_phys = x_pos * 6.5 / self.n_x

            color = {'intensity_anomaly': 'red', 'stripe_discontinuity': 'yellow',
                     'phase_discontinuity': 'blue'}.get(defect['type'], 'white')

            ax1.plot(x_phys, z_phys, 'o', color=color, markersize=6, alpha=0.8)

        ax1.set_title('Original Pattern + Defects')
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Z position')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # 2. 1D intensity profile and fit
        ax2 = fig.add_subplot(gs[0, 1])
        intensity_1d = self.extract_intensity_profile()
        predicted_intensity = np.cos(self.results.angles) ** 4

        ax2.plot(self.z_coords, intensity_1d, 'b-', linewidth=2, label='Observed', alpha=0.7)
        ax2.plot(self.z_coords, predicted_intensity, 'r--', linewidth=2, label='Reconstructed')

        # Mark defect regions
        for defect in self.results.defect_detections:
            z_pos = defect['position'][0]
            z_phys = z_pos * 11 / self.n_z
            ax2.axvline(z_phys, color='red', alpha=0.3, linestyle=':')

        ax2.set_xlabel('Z position')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Intensity Profile Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Reconstructed angles
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.z_coords, self.results.angles, 'g-', linewidth=2)
        ax3.set_xlabel('Z position')
        ax3.set_ylabel('Angle β (radians)')
        ax3.set_title('Reconstructed Director Angles')
        ax3.grid(True, alpha=0.3)

        # 4. Confidence map
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(self.z_coords, self.results.confidence, 'purple', linewidth=2)
        ax4.fill_between(self.z_coords, 0, self.results.confidence, alpha=0.3, color='purple')
        ax4.set_xlabel('Z position')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Reconstruction Confidence')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)

        # 5. Residuals analysis
        ax5 = fig.add_subplot(gs[1, 0])
        residuals = predicted_intensity - intensity_1d
        ax5.plot(self.z_coords, residuals, 'k-', linewidth=1)
        ax5.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax5.fill_between(self.z_coords, residuals, alpha=0.3)
        ax5.set_xlabel('Z position')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Fit Residuals')
        ax5.grid(True, alpha=0.3)

        # 6. Error metrics text
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.axis('off')

        metrics_text = "Error Metrics:\n"
        for key, value in self.results.error_metrics.items():
            metrics_text += f"{key.upper()}: {value:.4f}\n"

        metrics_text += f"\nDefects Detected: {len(self.results.defect_detections)}\n"
        metrics_text += f"Method: {self.results.reconstruction_method}\n"

        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # 7. Defect statistics
        ax7 = fig.add_subplot(gs[1, 2])

        if self.results.defect_detections:
            defect_types = {}
            for defect in self.results.defect_detections:
                dtype = defect['type']
                defect_types[dtype] = defect_types.get(dtype, 0) + 1

            types, counts = zip(*defect_types.items())
            bars = ax7.bar(range(len(types)), counts, alpha=0.7)
            ax7.set_xticks(range(len(types)))
            ax7.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=8)
            ax7.set_ylabel('Count')
            ax7.set_title('Defect Type Statistics')

            # Color bars by type
            colors = ['red', 'yellow', 'blue', 'green', 'orange']
            for bar, color in zip(bars, colors[:len(bars)]):
                bar.set_color(color)
        else:
            ax7.text(0.5, 0.5, 'No Defects\nDetected', ha='center', va='center',
                     transform=ax7.transAxes, fontsize=12)
            ax7.set_title('Defect Type Statistics')

        # 8. Director field visualization
        ax8 = fig.add_subplot(gs[1, 3])

        # Sample director field for visualization
        z_sample = self.z_coords[::5]
        x_sample = np.linspace(0, 6.5, 8)
        Z_grid, X_grid = np.meshgrid(z_sample, x_sample, indexing='ij')

        # Director components
        angles_sample = self.results.angles[::5]
        U = np.tile(np.cos(angles_sample).reshape(-1, 1), (1, 8))
        V = np.tile(np.sin(angles_sample).reshape(-1, 1), (1, 8))

        # Color by confidence
        confidence_sample = self.results.confidence[::5]
        colors = plt.cm.viridis(np.tile(confidence_sample.reshape(-1, 1), (1, 8)))

        ax8.quiver(X_grid, Z_grid, U, V, scale=15, alpha=0.8, width=0.003)
        ax8.set_xlabel('X position')
        ax8.set_ylabel('Z position')
        ax8.set_title('Director Field')
        ax8.set_aspect('equal')

        # 9. Power spectrum comparison
        ax9 = fig.add_subplot(gs[2, :2])

        # FFT of original and reconstructed
        fft_orig = np.fft.fft(intensity_1d)
        fft_recon = np.fft.fft(predicted_intensity)
        freqs = np.fft.fftfreq(len(intensity_1d), d=self.z_coords[1] - self.z_coords[0])

        # Plot positive frequencies only
        pos_freqs = freqs[:len(freqs) // 2]
        ax9.plot(pos_freqs, np.abs(fft_orig[:len(freqs) // 2]), 'b-', label='Original', alpha=0.7)
        ax9.plot(pos_freqs, np.abs(fft_recon[:len(freqs) // 2]), 'r--', label='Reconstructed', alpha=0.7)

        ax9.set_xlabel('Spatial Frequency')
        ax9.set_ylabel('Magnitude')
        ax9.set_title('Power Spectrum Comparison')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_xlim(0, 2)

        # 10. Reconstruction segments (if available)
        ax10 = fig.add_subplot(gs[2, 2:])

        if 'segments' in self.results.fitted_params:
            segments = self.results.fitted_params['segments']

            for i, segment in enumerate(segments):
                if 'range' in segment:
                    start, end = segment['range']
                    z_start = self.z_coords[start]
                    z_end = self.z_coords[end - 1] if end < len(self.z_coords) else self.z_coords[-1]

                    ax10.axvspan(z_start, z_end, alpha=0.3, label=f'Segment {i + 1}')

                    # Add segment parameters as text
                    z_center = (z_start + z_end) / 2
                    pitch = segment.get('pitch', 'N/A')
                    ax10.text(z_center, 0.8 - i * 0.1, f'p={pitch:.2f}',
                              ha='center', fontsize=8, transform=ax10.get_xaxis_transform())

            ax10.plot(self.z_coords, self.results.angles, 'k-', linewidth=2, alpha=0.7)
            ax10.set_xlabel('Z position')
            ax10.set_ylabel('Angle β (radians)')
            ax10.set_title('Reconstruction Segments')
            ax10.grid(True, alpha=0.3)

            if len(segments) <= 5:  # Only show legend if not too many segments
                ax10.legend(fontsize=8)
        else:
            ax10.plot(self.z_coords, self.results.angles, 'k-', linewidth=2)
            ax10.set_xlabel('Z position')
            ax10.set_ylabel('Angle β (radians)')
            ax10.set_title('Reconstructed Angles')
            ax10.grid(True, alpha=0.3)

        plt.suptitle(f'Advanced FCPM Reconstruction Analysis\n{self.results.reconstruction_method.title()}',
                     fontsize=14, fontweight='bold')
        plt.show()

    def save_results(self, filename_base: str):
        """Save reconstruction results"""
        if self.results is None:
            raise ValueError("No results to save. Run reconstruction first.")

        # Save main results
        np.save(f"{filename_base}_angles.npy", self.results.angles)
        np.save(f"{filename_base}_confidence.npy", self.results.confidence)

        # Save metadata and parameters
        save_data = {
            'fitted_params': self.results.fitted_params,
            'defect_detections': self.results.defect_detections,
            'reconstruction_method': self.results.reconstruction_method,
            'error_metrics': self.results.error_metrics,
            'metadata': self.results.metadata
        }

        with open(f"{filename_base}_results.json", 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Results saved as '{filename_base}_*'")


# Example usage
if __name__ == "__main__":
    print("Advanced FCPM Reconstruction Tools - Demo")

    # Try to load data from enhanced simulator
    try:
        intensity_data = np.load('enhanced_fcpm_with_dislocations_intensity.npy')
        print("Loaded enhanced simulation data with defects")
    except FileNotFoundError:
        print("Enhanced simulation data not found. Run enhanced simulator first.")
        print("Using basic simulation data instead...")
        try:
            intensity_data = np.load('fcpm_simulated_intensity.npy')
        except FileNotFoundError:
            print("No simulation data found. Run simulators first.")
            exit()

    # Create reconstructor
    reconstructor = AdvancedFCPMReconstructor(intensity_data)

    # Test reconstruction with defect handling
    print("\nRunning advanced reconstruction with defect detection...")
    results = reconstructor.reconstruct_with_defects(method='adaptive_fitting')

    # Display results
    reconstructor.plot_reconstruction_analysis()

    # Save results
    reconstructor.save_results('advanced_reconstruction_demo')

    print(f"\nReconstruction complete!")
    print(f"Defects detected: {len(results.defect_detections)}")
    print(f"RMSE: {results.error_metrics['rmse']:.4f}")
    print(f"R²: {results.error_metrics['r_squared']:.4f}")