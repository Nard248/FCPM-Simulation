# V2 Sign Optimization - Layer-by-Layer Energy Minimization

## Overview

V2 introduces a physics-informed, layer-by-layer approach to resolving the nematic sign ambiguity in FCPM reconstruction.

## The Problem

In nematic liquid crystals, the director **n** ≡ **-n** (head-tail equivalence). FCPM measures:
```
I(α) ∝ [nx·cos(α) + ny·sin(α)]⁴
```
which is sign-invariant. Q-tensor reconstruction gives us the director axis but not orientation.

## The Solution: Layer-by-Layer Energy Minimization

**Key Insight**: The director field should be smooth (minimize Frank elastic energy).

**Algorithm**:
1. **Layer Propagation**: Starting from z=0, for each subsequent layer z, align each voxel with its z-1 neighbor (minimize |n_z - n_{z-1}|²)
2. **Iterative Refinement**: Global passes to catch in-plane inconsistencies

This approach:
- Mirrors the physical z-stack acquisition of FCPM
- Is conceptually cleaner than BFS from arbitrary seed
- Is 4x faster than V1 on average
- Achieves same quality at moderate noise levels

## Usage

```python
from v2.sign_optimization_v2 import layer_then_refine

# After Q-tensor reconstruction
director_raw, Q, info = fcpm.reconstruct_via_qtensor(I_fcpm)

# Apply V2 sign optimization
result = layer_then_refine(director_raw, verbose=True)
director_fixed = result.director

print(f"Energy: {result.final_energy:.2f}")
print(f"Energy reduction: {result.initial_energy - result.final_energy:.2f}")
```

## Comparison with V1

| Metric | V1 (BFS + flip) | V2 (layer + refine) |
|--------|-----------------|---------------------|
| Average time | 0.12s | 0.03s |
| Energy (5% noise) | 60776 | 60776 |
| Energy (15% noise) | 76780 | 96430 |
| Conceptual clarity | Medium | High |

**Recommendation**: Use V2 for standard cases (noise ≤10%). Use V1 for very noisy data.

## Files

- `sign_optimization_v2.py` - Main implementation
- `test_v2_optimization.py` - Comprehensive benchmark
- `noise_sensitivity_comparison.png` - Generated plot
- `v1_v2_visual_comparison.png` - Generated visualization

## Key Functions

| Function | Description |
|----------|-------------|
| `layer_then_refine()` | **Recommended** - Layer propagation + refinement |
| `layer_by_layer_vectorized()` | Fast layer-only (no refinement) |
| `combined_v2_optimization()` | Wrapper for layer_then_refine |
| `compute_gradient_energy()` | Energy calculation |

## Integration with Main Library

To use V2 in the main fcpm library, the `combined_optimization` function could be updated to use V2 internally, or a new option could be added:

```python
# Option 1: Replace V1
from v2.sign_optimization_v2 import layer_then_refine
# Use instead of combined_optimization

# Option 2: Add as alternative
director, info = fcpm.reconstruct(I_fcpm, sign_method='v2')
```
