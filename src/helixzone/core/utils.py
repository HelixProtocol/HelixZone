"""Utility functions for safe OpenCV operations."""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Any, Union, cast
import logging

def safe_gaussian_blur(
    img: NDArray[Any], 
    kernel_size: Tuple[int, int], 
    sigma: float, 
    border_type: int = cv2.BORDER_DEFAULT
) -> NDArray[Any]:
    """Apply Gaussian blur safely with proper error handling.
    
    Args:
        img: Input image
        kernel_size: Kernel size as (width, height)
        sigma: Standard deviation
        border_type: Border handling method
        
    Returns:
        Blurred image or original if operation fails
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_gaussian_blur: Empty input image")
            return img
            
        # Make sure kernel size is valid (odd and positive)
        ksize_x = max(1, kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1)
        ksize_y = max(1, kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1)
        adjusted_ksize = (ksize_x, ksize_y)
        
        if adjusted_ksize != kernel_size:
            logging.info(f"Adjusted kernel size from {kernel_size} to {adjusted_ksize}")
        
        # Create copy to avoid in-place modification
        img_copy = img.copy()
        
        # Apply blur based on type
        if img.dtype == np.float32:
            result = cv2.GaussianBlur(img_copy, adjusted_ksize, sigma, borderType=border_type)
        else:
            # Convert to float32 for better precision
            tmp_img = img_copy.astype(np.float32)
            result = cv2.GaussianBlur(tmp_img, adjusted_ksize, sigma, borderType=border_type)
            
            # Convert back to original type
            if img.dtype != np.float32:
                result = result.astype(img.dtype)
                
        return result
    except Exception as e:
        logging.error(f"Gaussian blur failed: {e}", exc_info=True)
        return img  # Return original on error
        
def safe_canny(
    img: NDArray[Any],
    threshold1: float,
    threshold2: float,
    aperture_size: int = 3,
    l2gradient: bool = False
) -> NDArray[np.uint8]:
    """Apply Canny edge detection safely with proper error handling.
    
    Args:
        img: Input image
        threshold1: First threshold
        threshold2: Second threshold
        aperture_size: Aperture size for Sobel operator
        l2gradient: Use L2 norm for gradient magnitude
        
    Returns:
        Edge map or empty array if operation fails
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_canny: Empty input image")
            return np.zeros((1, 1), dtype=np.uint8)
            
        # Make sure aperture size is valid (odd and between 3-7)
        aperture = max(3, min(7, aperture_size if aperture_size % 2 == 1 else aperture_size + 1))
        
        if aperture != aperture_size:
            logging.info(f"Adjusted aperture size from {aperture_size} to {aperture}")
        
        # Make sure image is in appropriate format (8-bit grayscale)
        if len(img.shape) > 2 and img.shape[2] > 1:
            # Convert color image to grayscale
            tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            tmp_img = img.copy()
            
        # Convert to 8-bit if needed
        if tmp_img.dtype != np.uint8:
            if np.issubdtype(tmp_img.dtype, np.floating):
                tmp_img = (tmp_img * 255).clip(0, 255).astype(np.uint8)
            else:
                tmp_img = tmp_img.astype(np.uint8)
        
        # Apply Canny
        return cv2.Canny(
            tmp_img,
            threshold1,
            threshold2,
            apertureSize=aperture,
            L2gradient=l2gradient
        )
    except Exception as e:
        logging.error(f"Canny edge detection failed: {e}", exc_info=True)
        if img is not None:
            return np.zeros(img.shape[:2], dtype=np.uint8)
        return np.zeros((1, 1), dtype=np.uint8)
        
def safe_resize(
    img: NDArray[Any],
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> NDArray[Any]:
    """Resize image safely with proper error handling.
    
    Args:
        img: Input image
        size: Target size as (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image or original if operation fails
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_resize: Empty input image")
            return img
            
        # Make sure size is valid
        if size[0] <= 0 or size[1] <= 0:
            logging.warning(f"Invalid target size: {size}")
            return img
            
        # Apply resize
        return cv2.resize(img, size, interpolation=interpolation)
    except Exception as e:
        logging.error(f"Image resize failed: {e}", exc_info=True)
        return img  # Return original on error
        
def safe_morphology(
    img: NDArray[Any],
    operation: int,
    kernel_size: Tuple[int, int],
    iterations: int = 1
) -> NDArray[Any]:
    """Apply morphological operations safely with proper error handling.
    
    Args:
        img: Input image
        operation: Morphological operation (cv2.MORPH_*)
        kernel_size: Kernel size as (width, height)
        iterations: Number of iterations
        
    Returns:
        Processed image or original if operation fails
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_morphology: Empty input image")
            return img
            
        # Make sure kernel size is valid
        ksize_x = max(1, kernel_size[0])
        ksize_y = max(1, kernel_size[1])
        kernel = np.ones((ksize_y, ksize_x), np.uint8)
        
        # Apply morphology
        return cv2.morphologyEx(img, operation, kernel, iterations=iterations)
    except Exception as e:
        logging.error(f"Morphological operation failed: {e}", exc_info=True)
        return img  # Return original on error
        
def safe_threshold(
    img: NDArray[Any],
    thresh: float,
    maxval: float,
    threshold_type: int = cv2.THRESH_BINARY
) -> Tuple[float, NDArray[Any]]:
    """Apply thresholding safely with proper error handling.
    
    Args:
        img: Input image
        thresh: Threshold value
        maxval: Maximum value
        threshold_type: Thresholding type
        
    Returns:
        Tuple of (threshold used, thresholded image)
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_threshold: Empty input image")
            return 0.0, img
            
        # Make sure image is in appropriate format
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                tmp_img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                tmp_img = img.astype(np.uint8)
        else:
            tmp_img = img
            
        # Apply threshold
        ret, thresh_img = cv2.threshold(tmp_img, thresh, maxval, threshold_type)
        
        # Convert back to original type if needed
        if img.dtype != np.uint8 and img.dtype != thresh_img.dtype:
            if np.issubdtype(img.dtype, np.floating):
                thresh_img = thresh_img.astype(img.dtype) / 255.0
            else:
                thresh_img = thresh_img.astype(img.dtype)
                
        return ret, thresh_img
    except Exception as e:
        logging.error(f"Thresholding failed: {e}", exc_info=True)
        return 0.0, img  # Return original on error
        
def safe_convert_color(
    img: NDArray[Any],
    conversion_code: int,
    dst_cn: int = 0
) -> NDArray[Any]:
    """Convert color space safely with proper error handling.
    
    Args:
        img: Input image
        conversion_code: Color conversion code
        dst_cn: Number of channels in output image
        
    Returns:
        Color-converted image or original if operation fails
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            logging.warning("safe_convert_color: Empty input image")
            return img
            
        # Make sure image has enough channels for the conversion
        if len(img.shape) < 3 and conversion_code in [cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR]:
            logging.warning("Cannot convert color on single-channel image with this conversion code")
            return img
            
        # Apply color conversion
        return cv2.cvtColor(img, conversion_code, dst_cn)
    except Exception as e:
        logging.error(f"Color conversion failed: {e}", exc_info=True)
        return img  # Return original on error 