"""
Minimal dataset module for Colab — only safe_normalize and Frangi.
No rasterio dependency (chips are loaded from .npy files directly).
"""

import numpy as np


def safe_normalize(arr, mean, std):
    arr = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
    return (arr - mean) / (std + 1e-8)


def compute_frangi_vesselness(image, sigmas=[1.0, 2.0, 4.0, 8.0],
                               alpha=0.5, beta=0.5):
    """
    Simplified Frangi vesselness using scipy.
    Returns vesselness map in [0, 1].
    """
    from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_dilation
    
    result = np.zeros_like(image, dtype=np.float32)
    
    for sigma in sigmas:
        # Smooth
        smoothed = gaussian_filter(image, sigma=sigma)
        
        # Compute Hessian (approximate with Sobel)
        from scipy.ndimage import sobel
        dxx = sobel(sobel(smoothed, axis=1), axis=1)
        dyy = sobel(sobel(smoothed, axis=0), axis=0)
        dxy = sobel(sobel(smoothed, axis=0), axis=1)
        
        # Eigenvalues (approximate)
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        # Vesselness (simplified)
        lambda1 = trace / 2 + np.sqrt(np.maximum(trace**2 / 4 - det, 0))
        lambda2 = trace / 2 - np.sqrt(np.maximum(trace**2 / 4 - det, 0))
        
        # Only respond to tube-like structures
        vesselness = np.zeros_like(image)
        mask = lambda2 < 0
        vesselness[mask] = np.exp(-lambda1[mask]**2 / (2 * alpha**2)) * \
                           (1 - np.exp(-lambda2[mask]**2 / (2 * beta**2)))
        
        result = np.maximum(result, vesselness)
    
    # Normalize to [0, 1]
    if result.max() > 0:
        result = result / result.max()
    
    return result
