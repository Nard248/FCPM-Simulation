# Visualization

## Director Field Plots

```python
import matplotlib.pyplot as plt
import fcpm

# Quiver plot at a z-slice
fig, ax = plt.subplots()
fcpm.plot_director_slice(director, z_idx=16, step=2, ax=ax)
plt.show()

# RGB orientation map
fig = fcpm.plot_director_rgb(director, z_idx=16)
plt.show()

# Streamlines
fig = fcpm.plot_director_streamlines(director, z_idx=16)
plt.show()
```

## FCPM Intensities

```python
fig = fcpm.plot_fcpm_intensities(I_fcpm, z_idx=16)
plt.show()
```

## Comparison and Error Maps

```python
# Side-by-side comparison
fig = fcpm.compare_directors(director_gt, director_recon, z_idx=16)
plt.show()

# Angular error map
fig, ax = plt.subplots()
fcpm.plot_error_map(director_recon, director_gt, z_idx=16, ax=ax)
plt.show()

# Error histogram
fig = fcpm.plot_error_histogram(director_recon, director_gt)
plt.show()

# Error by depth
fig = fcpm.plot_error_by_depth(director_recon, director_gt)
plt.show()
```

## Q-tensor Visualization

```python
fig = fcpm.plot_qtensor_components(Q, z_idx=16)
plt.show()

fig = fcpm.plot_order_parameter(Q, z_idx=16)
plt.show()
```

## 3D Visualization

```python
# Matplotlib 3D (no extra dependencies)
fcpm.visualize_3d_matplotlib(director, subsample=4)

# Export for ParaView
fcpm.export_for_paraview(director, 'output.vtk')

# TopoVec (if available)
if fcpm.check_topovec_available():
    fcpm.visualize_topovec(director)
```
