"""
FCPM Preprocessing Module

Tools for preprocessing director fields and FCPM intensity data.

Submodules:
    cropping: ROI extraction, cropping, padding, subsampling
    filtering: Noise reduction, background removal, normalization
"""

from .cropping import (
    crop_director,
    crop_director_center,
    crop_fcpm,
    crop_fcpm_center,
    crop_qtensor,
    pad_director,
    extract_slice,
    subsample_director,
    subsample_fcpm,
)

from .filtering import (
    gaussian_filter_fcpm,
    median_filter_fcpm,
    bilateral_filter_fcpm,
    remove_background_fcpm,
    normalize_fcpm,
    clip_fcpm,
    smooth_director,
)

__all__ = [
    # Cropping
    'crop_director',
    'crop_director_center',
    'crop_fcpm',
    'crop_fcpm_center',
    'crop_qtensor',
    'pad_director',
    'extract_slice',
    'subsample_director',
    'subsample_fcpm',
    # Filtering
    'gaussian_filter_fcpm',
    'median_filter_fcpm',
    'bilateral_filter_fcpm',
    'remove_background_fcpm',
    'normalize_fcpm',
    'clip_fcpm',
    'smooth_director',
]
