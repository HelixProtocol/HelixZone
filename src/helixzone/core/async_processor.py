"""Asynchronous image processing operations.

This module provides async versions of image processing operations
that run in background threads to keep the UI responsive. It uses
the task manager and the tiled processor to efficiently process images.
"""

import logging
import time
import numpy as np
import cv2
from typing import Optional, Callable, Any, Dict, Tuple, Union, List

from .task_manager import run_in_background, report_progress, is_cancelled
from .tiled_processor import tiled_processor
from .memory_manager import memory_manager
from .image_processing import ColorSpaceConverter, ImageStats

# Configure logger
logger = logging.getLogger(__name__)


def _calculate_pixel_count(image: np.ndarray) -> int:
    """Calculate number of pixels in an image.
    
    Args:
        image: Input image
        
    Returns:
        Number of pixels
    """
    if image is None or image.size == 0:
        return 0
    return np.prod(image.shape[:2])  # height * width


def _get_tile_count(image: np.ndarray) -> int:
    """Get number of tiles for the image based on memory usage.
    
    Args:
        image: Input image
        
    Returns:
        Suggested number of tiles
    """
    if image is None or image.size == 0:
        return 1
        
    # Get image size in bytes
    element_size = image.itemsize
    channels = 1
    if len(image.shape) > 2:
        channels = image.shape[2]
    
    # Calculate total size
    total_size = image.shape[0] * image.shape[1] * channels * element_size
    
    # Get memory info
    available_memory = memory_manager.get_available_system_memory()
    
    # Each tile should use maximum 10% of available memory
    # We also need to account for intermediate results during processing
    tile_max_size = max(32 * 1024 * 1024, available_memory * 0.1)  # Min 32MB tiles
    
    # Calculate number of tiles needed
    tile_count = max(1, int(np.ceil(total_size * 4 / tile_max_size)))
    
    logger.debug(f"Image size: {total_size / (1024 * 1024):.2f}MB, "
                f"Using {tile_count} tiles for processing")
    
    return tile_count


def async_apply_filter(
    image: np.ndarray,
    filter_func: Callable,
    filter_args: tuple = (),
    filter_kwargs: Optional[Dict[str, Any]] = None,
    use_tiling: bool = True,
    task_name: str = "Applying Filter",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply a filter to an image asynchronously.
    
    This function runs the filter operation in a background thread and
    returns a task ID that can be used to track progress.
    
    Args:
        image: Input image
        filter_func: Filter function to apply
        filter_args: Additional positional arguments for the filter function
        filter_kwargs: Additional keyword arguments for the filter function
        use_tiling: Whether to use tiled processing for large images
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    if filter_kwargs is None:
        filter_kwargs = {}
    
    def process_function(cancellation_event=None, progress_callback=None):
        """Process function that will run in background thread."""
        start_time = time.time()
        
        # Handle empty images
        if image is None or image.size == 0:
            logger.warning("Empty image passed to async_apply_filter")
            return np.array([])
        
        # Make a copy of the image to avoid modifying the original
        img_copy = image.copy()
        
        # Check if this is a large image that should use tiling
        pixel_count = _calculate_pixel_count(img_copy)
        large_image = pixel_count > 2_000_000  # > 2 megapixels
        
        if use_tiling and large_image:
            # Create progress callback for tiled processing
            def tile_progress(tile_idx, total_tiles, tile_result=None):
                if cancellation_event and cancellation_event.is_set():
                    return False  # Stop processing
                
                percent = (tile_idx / total_tiles) * 100
                if progress_callback:
                    progress_callback(percent, f"Processing tile {tile_idx}/{total_tiles}")
                return True  # Continue processing
            
            # Use tiled processing
            tile_count = _get_tile_count(img_copy)
            result = tiled_processor.process_image_in_tiles(
                img_copy,
                filter_func,
                tile_count=tile_count,
                progress_callback=tile_progress,
                args=filter_args,
                kwargs=filter_kwargs
            )
        else:
            # Process the entire image at once
            if progress_callback:
                progress_callback(10.0, "Starting filter operation...")
            
            # Apply the filter
            result = filter_func(img_copy, *filter_args, **filter_kwargs)
            
            if progress_callback:
                progress_callback(90.0, "Finishing filter operation...")
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            logger.info("Filter operation cancelled")
            return None
        
        # Report completion time
        elapsed = time.time() - start_time
        logger.info(f"Filter {filter_func.__name__} completed in {elapsed:.2f}s")
        
        if progress_callback:
            progress_callback(100.0, "Filter operation completed")
        
        return result
    
    # Submit task to background thread
    task_id = run_in_background(
        task_name,
        process_function,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )
    
    return task_id


def async_apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    blend_mode: str = 'normal',
    opacity: float = 1.0,
    task_name: str = "Applying Mask",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply a mask to an image asynchronously.
    
    Args:
        image: Input image
        mask: Mask to apply (same dimensions as image)
        blend_mode: How to blend masked areas ('normal', 'overlay', etc.)
        opacity: Opacity of the mask (0.0-1.0)
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    def apply_mask_impl(img, msk, cancellation_event=None, progress_callback=None):
        """Mask application implementation."""
        if img is None or img.size == 0 or msk is None or msk.size == 0:
            logger.warning("Empty image or mask in apply_mask")
            return np.array([])
        
        # Report progress
        if progress_callback:
            progress_callback(10.0, "Checking dimensions...")
        
        # Convert mask to float if needed
        if msk.dtype != np.float32:
            msk = msk.astype(np.float32) / 255.0
        
        # Ensure mask has the correct dimensions
        if len(msk.shape) == 2 and len(img.shape) == 3:
            # Convert 2D mask to 3D
            msk = np.stack([msk] * img.shape[2], axis=2)
        
        # Apply opacity
        if opacity < 1.0:
            msk = msk * opacity
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
            
        # Report progress
        if progress_callback:
            progress_callback(30.0, "Applying mask...")
        
        # Apply mask based on blend mode
        result = img.copy()
        
        if blend_mode == 'normal':
            # Simple alpha blending
            result = img * (1.0 - msk)
        elif blend_mode == 'multiply':
            # Multiply blend mode
            result = img * (1.0 - msk + msk * img / 255.0)
        elif blend_mode == 'overlay':
            # Overlay blend mode
            result = np.where(
                img <= 127,
                (2 * img * msk / 255.0),
                (1.0 - 2 * (1.0 - img / 255.0) * (1.0 - msk))
            ) * 255.0
        else:
            # Default to normal
            result = img * (1.0 - msk)
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
            
        # Report progress
        if progress_callback:
            progress_callback(90.0, "Finalizing result...")
        
        # Ensure the result is the correct type
        if img.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    # Get filter args
    filter_kwargs = {
        'blend_mode': blend_mode,
        'opacity': opacity
    }
    
    # Submit task to background thread
    return async_apply_filter(
        image,
        apply_mask_impl,
        filter_args=(mask,),
        filter_kwargs=filter_kwargs,
        task_name=task_name,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )


def async_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
    task_name: str = "Gaussian Blur",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply Gaussian blur to an image asynchronously.
    
    Args:
        image: Input image
        kernel_size: Size of the blur kernel
        sigma: Standard deviation of the Gaussian kernel
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    def gaussian_blur_impl(img, cancellation_event=None, progress_callback=None):
        """Gaussian blur implementation."""
        if img is None or img.size == 0:
            return np.array([])
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            k_size = kernel_size + 1
        else:
            k_size = kernel_size
        
        # Apply blur
        if progress_callback:
            progress_callback(40.0, f"Applying {k_size}x{k_size} Gaussian blur...")
        
        result = cv2.GaussianBlur(img, (k_size, k_size), sigma)
        
        return result
    
    # Submit task to background thread
    return async_apply_filter(
        image,
        gaussian_blur_impl,
        filter_kwargs={'kernel_size': kernel_size, 'sigma': sigma},
        task_name=task_name,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )


def async_edge_detection(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    task_name: str = "Edge Detection",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply edge detection to an image asynchronously.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for the hysteresis procedure
        high_threshold: Higher threshold for the hysteresis procedure
        aperture_size: Aperture size for the Sobel operator
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    def edge_detection_impl(img, cancellation_event=None, progress_callback=None):
        """Edge detection implementation."""
        if img is None or img.size == 0:
            return np.array([])
        
        # Report progress
        if progress_callback:
            progress_callback(10.0, "Converting to grayscale...")
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
        
        # Convert to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Report progress
        if progress_callback:
            progress_callback(40.0, "Applying edge detection...")
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            gray,
            low_threshold,
            high_threshold,
            apertureSize=aperture_size
        )
        
        # Report progress
        if progress_callback:
            progress_callback(70.0, "Finalizing result...")
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
        
        # Create color image from edges
        if len(img.shape) == 3:
            result = np.zeros_like(img)
            result[edges > 0] = [255, 255, 255]
        else:
            result = edges
        
        return result
    
    # Submit task to background thread
    return async_apply_filter(
        image,
        edge_detection_impl,
        filter_kwargs={
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'aperture_size': aperture_size
        },
        task_name=task_name,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )


def async_histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    task_name: str = "Histogram Equalization",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply adaptive histogram equalization to an image asynchronously.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    def histogram_equalization_impl(img, cancellation_event=None, progress_callback=None):
        """Histogram equalization implementation."""
        if img is None or img.size == 0:
            return np.array([])
        
        # Report progress
        if progress_callback:
            progress_callback(10.0, "Preparing histogram equalization...")
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
        
        result = None
        
        # Apply CLAHE based on image type
        if len(img.shape) == 2:
            # Grayscale image
            if progress_callback:
                progress_callback(40.0, "Applying equalization to grayscale image...")
            result = clahe.apply(img)
        else:
            # Color image - need to process in LAB color space
            if progress_callback:
                progress_callback(30.0, "Converting to LAB color space...")
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                return None
            
            # Split the LAB image into channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            if progress_callback:
                progress_callback(50.0, "Equalizing luminance channel...")
            cl = clahe.apply(l)
            
            # Merge channels back
            if progress_callback:
                progress_callback(70.0, "Merging channels...")
            lab = cv2.merge((cl, a, b))
            
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                return None
            
            # Convert back to BGR
            if progress_callback:
                progress_callback(90.0, "Converting back to RGB color space...")
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    # Submit task to background thread
    return async_apply_filter(
        image,
        histogram_equalization_impl,
        filter_kwargs={
            'clip_limit': clip_limit,
            'tile_grid_size': tile_grid_size
        },
        task_name=task_name,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )


def async_threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255,
    threshold_type: int = cv2.THRESH_BINARY,
    task_name: str = "Thresholding",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_complete: Optional[Callable[[np.ndarray], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> str:
    """Apply thresholding to an image asynchronously.
    
    Args:
        image: Input image
        threshold_value: Threshold value
        max_value: Maximum value to use with threshold
        threshold_type: OpenCV threshold type
        task_name: Name of the task for progress reporting
        on_progress: Callback for progress updates
        on_complete: Callback when operation completes
        on_error: Callback when operation fails
        
    Returns:
        Task ID
    """
    def threshold_impl(img, cancellation_event=None, progress_callback=None):
        """Thresholding implementation."""
        if img is None or img.size == 0:
            return np.array([])
        
        # Report progress
        if progress_callback:
            progress_callback(10.0, "Preparing thresholding...")
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            if progress_callback:
                progress_callback(30.0, "Converting to grayscale...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return None
        
        # Apply threshold
        if progress_callback:
            progress_callback(60.0, f"Applying threshold at level {threshold_value}...")
        
        _, result = cv2.threshold(gray, threshold_value, max_value, threshold_type)
        
        # Convert back to color if original was color
        if len(img.shape) == 3:
            if progress_callback:
                progress_callback(80.0, "Converting result to color...")
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    # Submit task to background thread
    return async_apply_filter(
        image,
        threshold_impl,
        filter_kwargs={
            'threshold_value': threshold_value,
            'max_value': max_value,
            'threshold_type': threshold_type
        },
        task_name=task_name,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error
    )


# Export all async functions
__all__ = [
    'async_apply_filter',
    'async_apply_mask',
    'async_gaussian_blur',
    'async_edge_detection',
    'async_histogram_equalization',
    'async_threshold'
] 