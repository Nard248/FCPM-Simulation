# Identifiability in FCPM Reconstruction

This document describes what can and cannot be recovered from standard FCPM measurements, and how the library reports this information.

## What FCPM Measures

In standard FCPM, the fluorescence intensity at polarization angle $\alpha$ is:

$$I(\alpha) = [n_x \cos\alpha + n_y \sin\alpha]^4$$

where $(n_x, n_y, n_z)$ is the local director (unit vector). With 4 measurements at $\alpha = 0, \pi/4, \pi/2, 3\pi/4$, we can extract:

| Quantity | Determinable? | How |
|----------|:---:|-----|
| $n_x^2$ | Exactly | From $I(0)^{1/4}$ |
| $n_y^2$ | Exactly | From $I(\pi/2)^{1/4}$ |
| $n_x n_y$ (with sign) | Exactly | From $I(\pi/4)^{1/4} - I(3\pi/4)^{1/4}$ |
| $n_z^2$ | Exactly | From $1 - n_x^2 - n_y^2$ (unit constraint) |
| sign of $n_x n_z$ | **No** | Only $\|n_x\| \cdot \|n_z\|$ available |
| sign of $n_y n_z$ | **No** | Only $\|n_y\| \cdot \|n_z\|$ available |

## Q-Tensor Components

The Q-tensor $Q_{ij} = S(n_i n_j - \delta_{ij}/3)$ inherits this structure:

- **Observable** (exactly determinable): $Q_{xx}, Q_{yy}, Q_{xy}$
- **Ambiguous** (magnitude only, sign unknown): $Q_{xz}, Q_{yz}$
- **Derived** (from tracelessness): $Q_{zz} = -Q_{xx} - Q_{yy}$

The `ReconstructionResult` object returned by `reconstruct_via_qtensor()` explicitly records this:

```python
result = fcpm.reconstruct_via_qtensor(I_fcpm)
print(result.observable_components)  # ['Q_xx', 'Q_yy', 'Q_xy']
print(result.ambiguous_components)   # ['Q_xz', 'Q_yz']
```

## Three Identifiability Regimes

### Regime 1: Smooth director recoverable (up to global sign)

**When**: The field is smooth and has no half-integer topological defects. $n_z$ is not too large (sufficient in-plane signal).

**What the library returns**: A director field with consistent signs, confidence map near 1.0 everywhere.

**Example**: Cholesteric twist, uniform tilt, most soliton structures.

### Regime 2: No smooth director exists, but Q-tensor is smooth

**When**: The field contains half-integer disclination lines ($\pm 1/2$ charge). Any vectorial representation requires branch cuts — discontinuities that are a topological consequence, not an algorithmic failure.

**What the library returns**: A director field with branch cuts (detected by `compute_branch_cut_map()`), `ambiguity_mask=True` near the defect core.

**Example**: $+1/2$ or $-1/2$ disclination lines in thin films.

### Regime 3: Even Q-tensor recovery is uncertain

**When**: The in-plane signal is weak ($n_z \approx \pm 1$), the intensity drops as $\cos^4\beta \to 0$, and the in-plane angle becomes numerically unstable. Also occurs under strong noise, model mismatch, or insufficient measurements.

**What the library returns**: `confidence_map` values near 0.0 in affected regions, `ambiguity_mask=True`.

**Example**: Nearly vertical director (homeotropic alignment), low-signal regions in thick samples.

## Using the Diagnostics

### Confidence Map

The per-voxel confidence combines two signals:

1. **In-plane magnitude**: $n_x^2 + n_y^2$ (0 when $n_z = \pm 1$, 1 when in-plane)
2. **Q-tensor eigen-gap**: $(\lambda_1 - \lambda_2) / |\lambda_1|$ (low when the two largest eigenvalues are close, meaning the orientation is poorly defined)

```python
result = fcpm.reconstruct_via_qtensor(I_fcpm)

# Regions with low confidence should not be trusted
low_confidence = result.confidence_map < 0.3
print(f"Low-confidence voxels: {low_confidence.sum()} / {low_confidence.size}")
```

### Ambiguity Mask

Boolean mask flagging voxels where $|n_z| > 0.9$ (threshold configurable via `compute_ambiguity_mask(director, nz_threshold=0.9)`).

```python
print(f"Ambiguous voxels: {result.ambiguity_mask.sum()}")
```

### Branch Cut Detection

For fields with half-integer defects:

```python
branch_cuts = fcpm.compute_branch_cut_map(result.director)
print(f"Branch cut voxels: {branch_cuts.sum()}")
# These discontinuities are topologically required, not errors
```

## Reconstruction Modes

The `reconstruct_via_qtensor()` function supports 4 modes:

| Mode | Description | When to use |
|------|-------------|-------------|
| `'full'` | Reconstruct full Q-tensor, extract director | Default. Good for most cases |
| `'observed_Q'` | Only the 3 observable components ($Q_{xz} = Q_{yz} = 0$) | When you want to avoid sign-ambiguous components |
| `'line_field'` | Director without sign fixing | When you explicitly want a line field |
| `'director'` | Full pipeline with automatic sign optimization | Convenience mode |

```python
# Only trust the observable components
result_safe = fcpm.reconstruct_via_qtensor(I_fcpm, mode='observed_Q')

# Full pipeline with sign fixing
result_full = fcpm.reconstruct_via_qtensor(I_fcpm, mode='director')
```

## Recommendations

1. **Always check `confidence_map`** before trusting the director in a region.
2. **For half-integer defects**, compare Q-tensors instead of directors — `qtensor_frobenius_error()` is sign-invariant.
3. **For weak-signal regions**, use `mode='observed_Q'` to avoid introducing errors from the ambiguous $Q_{xz}, Q_{yz}$ components.
4. **For publication**, report both the director error *and* the fraction of voxels with `confidence > threshold` to give an honest assessment of reconstruction quality.
