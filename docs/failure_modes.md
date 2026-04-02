# Known Failure Modes

A catalog of situations where the library produces unreliable results, with explanations and diagnostics. This is part of the library's product — honest failure-mode reporting is as important as successful examples.

## 1. Low Signal (Nearly Vertical Director)

**Symptom**: Large angular errors in regions where $n_z \approx \pm 1$.

**Cause**: The FCPM intensity $I(\alpha) = [n_x \cos\alpha + n_y \sin\alpha]^4$ depends only on the in-plane projection. When the director is nearly vertical, $n_x \approx n_y \approx 0$ and all four intensity images are near zero. The fourth-power dependence makes this worse — even moderate tilt produces very weak signal.

**Diagnostic**: `confidence_map` drops below 0.1 in these regions. `ambiguity_mask` is True.

**Mitigation**: Use `mode='observed_Q'` to avoid contaminating the result with guessed $Q_{xz}, Q_{yz}$ signs. Report only in regions where `confidence_map > threshold`.

## 2. Half-Integer Defects (No Smooth Director)

**Symptom**: A sharp line of high angular error cutting through the volume, even after sign optimization.

**Cause**: For topological charge $\pm 1/2$ disclinations, no globally smooth director field exists. Any vectorial representation requires at least one branch cut (a line of discontinuity). This is a mathematical consequence of the topology, not an algorithm failure.

**Diagnostic**: `compute_branch_cut_map(director)` returns True along the cut. The error is concentrated on a 1D line, not scattered randomly.

**Mitigation**: Compare using Q-tensor metrics (`qtensor_frobenius_error`), which are sign-invariant and do not penalize branch cuts. Or use the `'observed_Q'` reconstruction mode.

## 3. GraphCuts on Multi-Domain Sign Patterns

**Symptom**: GraphCuts returns a solution with large domains of wrong sign (bimodal accuracy distribution, some structures near 0.5 accuracy).

**Cause**: The s-t min-cut partitions voxels into exactly two groups. If the true sign pattern requires three or more disconnected domains, the single cut cannot represent the solution. The BFS pre-alignment partially addresses this, but complex sign topologies may still defeat the algorithm.

**Diagnostic**: Bimodal error distribution. The 3-axis error projections show large contiguous regions of high error, not scattered voxels.

**Mitigation**: Use `CombinedOptimizer` or `HierarchicalOptimizer` which do not have this topological limitation. Or run GraphCuts followed by iterative local refinement.

## 4. Energy Minimization Introduces Bias

**Symptom**: Smooth-looking director field but with systematic angular offset compared to ground truth.

**Cause**: The gradient energy $\sum |n_i - n_j|^2$ acts as a smoothness regularizer. In regions with rapid director rotation (near defects, at soliton walls), the energy minimizer prefers a smoother field than the physical one. The energy minimum over sign configurations is not the same as the maximum-likelihood estimate under noise.

**Diagnostic**: Plot the angular error map — bias appears as a systematic offset in high-curvature regions, not random noise.

**Mitigation**: This is a fundamental limitation of the current energy-based approach. Priority 3 of the audit remediation plan addresses this with likelihood-based reconstruction.

## 5. Wrong Elastic Constants

**Symptom**: Frank energy decomposition gives physically implausible results (e.g., negative energy density, or twist energy lower with wrong pitch).

**Cause**: The Frank constants ($K_1, K_2, K_3$) and cholesteric pitch are material properties that must be known *a priori*. If they are wrong, the Frank energy is still computed correctly as a mathematical quantity, but its physical interpretation is invalid.

**Diagnostic**: Compare with measured values from the literature. Run `parameter_sensitivity_study()` to quantify how sensitive your result is to the assumed constants.

**Mitigation**: Use the gradient energy (single-constant approximation) for sign optimization, which does not require knowledge of individual Frank constants. Reserve the anisotropic Frank energy for analysis, not optimization.

## 6. Periodic Boundary Artifacts

**Symptom**: Elevated error at volume boundaries (first and last voxels along each axis).

**Cause**: `compute_gradient_energy()` uses `np.roll` for finite differences, which wraps around: the last voxel is compared with the first. For non-periodic physical fields, this creates artificial energy contributions at the boundaries that all optimizers try to minimize.

**Diagnostic**: Plot error vs. z-layer (`spatial_error_distribution`) — boundary layers show elevated mean error.

**Mitigation**: Crop the outer layer of voxels after reconstruction. Or pad the field before optimization and discard the padding afterward.

## 7. Grid Anisotropy

**Symptom**: Sign optimization quality differs along different axes.

**Cause**: The gradient energy treats all axes equally ($\Delta x = \Delta y = \Delta z = 1$ voxel), but the physical voxel may be anisotropic (e.g., confocal z-step is typically larger than xy pixel size). A physical gradient of 1 rad/um appears as different voxel gradients along different axes.

**Diagnostic**: Check the voxel aspect ratio. If z-step is 3x the xy-step, the gradient energy over-penalizes z-variations by a factor of 9.

**Mitigation**: Scale the director field to isotropic voxels before optimization, or modify the energy function to account for voxel sizes (not yet implemented).

## 8. Oversmoothing by Graph Frustration

**Symptom**: The optimized field looks smoother than the true field, with reduced twist or splay amplitude.

**Cause**: When the sign optimization encounters frustrated regions (where no sign assignment can satisfy all neighbors simultaneously), it resolves the frustration by choosing signs that reduce the gradient energy — which may not correspond to the physically correct signs.

**Diagnostic**: Compare `sign_accuracy()` with `angular_error_nematic()`. If sign accuracy is high but angular error is also high, the directors themselves may have been corrupted (not just their signs).

**Mitigation**: This should not happen with sign-only optimization. If it does, the input directors may not be unit-normalized. Check `director.is_normalized()`.

## 9. Sample Drift Between Angles

**Symptom**: Systematic spatial offset in the reconstructed director, checkerboard-like artifacts, or anomalous cross-term values.

**Cause**: If the sample drifts between acquisition of different polarization angles, the four images are not co-registered. The algebraic inversion assumes they are registered.

**Diagnostic**: Look for spatial derivatives of the error that correlate with strong intensity gradients. Use `apply_slice_misregistration()` to simulate and compare.

**Mitigation**: Register the images before reconstruction. Or use `apply_slice_misregistration()` in sensitivity analysis to quantify the impact.
