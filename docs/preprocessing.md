# Preprocessing

## Cropping

```python
import fcpm

# Crop to specific region
director_crop = fcpm.crop_director(director,
    y_range=(100, 200), x_range=(100, 200), z_range=(10, 30))

# Crop from center
director_crop = fcpm.crop_director_center(director, size=(64, 64, 32))

# Same for FCPM data
I_crop = fcpm.crop_fcpm_center(I_fcpm, size=(64, 64, 32))
```

## Filtering

```python
# Gaussian smoothing
I_smooth = fcpm.gaussian_filter_fcpm(I_fcpm, sigma=1.0)

# Median filter (salt-and-pepper noise)
I_filtered = fcpm.median_filter_fcpm(I_fcpm, size=3)

# Background removal
I_clean = fcpm.remove_background_fcpm(I_fcpm, method='percentile')

# Normalization
I_norm = fcpm.normalize_fcpm(I_fcpm, method='global')
```

## Adding Noise

For simulation testing:

```python
# Realistic mixed noise
I_noisy = fcpm.add_fcpm_realistic_noise(
    I_fcpm, noise_model='mixed', gaussian_sigma=0.03, seed=42)

# Individual types
I_noisy = fcpm.add_gaussian_noise(I_fcpm, sigma=0.05)
I_noisy = fcpm.add_poisson_noise(I_fcpm, scale=100)
```

## Subsampling

```python
director_sub = fcpm.subsample_director(director, factor=2)
I_sub = fcpm.subsample_fcpm(I_fcpm, factor=2)
```
