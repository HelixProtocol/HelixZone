"""Type checking utilities for HelixZone with GPU acceleration support."""

from typing import Union, Dict, Any, Optional, TypeVar, Protocol, runtime_checkable, cast, TYPE_CHECKING
from typing_extensions import TypeAlias
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
import cv2
import logging
import os
import time
from functools import wraps
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    
try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(log_dir, exist_ok=True)

# File handler for errors
error_handler = logging.FileHandler(os.path.join(log_dir, "errors.log"))
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)

# Console handler for debug info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(error_handler)
logger.addHandler(console_handler)

if TYPE_CHECKING:
    # Define full types for type checking
    ImageArray = NDArray[np.uint8]
else:
    # Use string literal for runtime to avoid type expression issues
    ImageArray = "NDArray[np.uint8]"  # type: ignore

# Type aliases
ImageTypes = Union[str, Path, QImage, "NDArray[np.uint8]"]  # type: ignore

@dataclass
class ProcessingParams:
    """Parameters for image processing."""
    sigma: float = 1.0
    threshold: float = 0.5
    kernel_size: int = 3
    iterations: int = 1
    normalize: bool = True
    threshold1: float = 100.0  # Lower threshold for Canny
    threshold2: float = 200.0  # Upper threshold for Canny
    aperture_size: int = 3     # Aperture size for Canny
    l2_gradient: bool = False  # Use L2 gradient for Canny

class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

# Performance monitoring
class PerformanceMetrics:
    """Track performance metrics for image processing operations."""
    def __init__(self):
        self.operation_times: Dict[str, list[float]] = {}
        self.memory_usage: Dict[str, list[int]] = {}
        self.gpu_usage: Dict[str, list[Optional[int]]] = {}
        
    def record_operation(self, operation: str, duration: float, memory_used: int, gpu_memory: Optional[int] = None):
        """Record metrics for an operation."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.memory_usage[operation] = []
            self.gpu_usage[operation] = []
            
        self.operation_times[operation].append(duration)
        self.memory_usage[operation].append(memory_used)
        self.gpu_usage[operation].append(gpu_memory)
        
    def get_average_metrics(self, operation: str) -> Dict[str, Optional[float]]:
        """Get average metrics for an operation."""
        if operation not in self.operation_times:
            return {}
            
        return {
            'avg_time': float(np.mean(self.operation_times[operation])),
            'avg_memory': float(np.mean(self.memory_usage[operation])),
            'avg_gpu_memory': float(np.mean([x for x in self.gpu_usage[operation] if x is not None])) if any(x is not None for x in self.gpu_usage[operation]) else None
        }

# Initialize performance metrics
metrics = PerformanceMetrics()

def track_performance(func):
    """Decorator to track performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = 0  # TODO: Add memory tracking
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            end_memory = 0  # TODO: Add memory tracking
            gpu_memory = None
            if HAS_GPU:
                gpu_memory = cp.get_default_memory_pool().used_bytes()
            metrics.record_operation(func.__name__, duration, end_memory - start_memory, gpu_memory)
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            raise
    return wrapper

def validate_image(image: Any) -> bool:
    """Validate that an object is a valid image type."""
    logger.debug(f"Validating image of type: {type(image)}")
    
    if isinstance(image, (str, Path)):
        path = Path(image)
        is_valid = path.exists() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        if not is_valid:
            logger.warning(f"Invalid image path: {path}")
        return is_valid
    
    if isinstance(image, QImage):
        is_valid = not image.isNull()
        if not is_valid:
            logger.warning("QImage is null")
        return is_valid
    
    if isinstance(image, np.ndarray):
        is_valid_dtype = image.dtype == np.uint8
        is_valid_dims = image.ndim in (2, 3)
        is_valid_channels = (image.ndim == 2 or 
                           (image.ndim == 3 and image.shape[-1] in (1, 3, 4)))
        is_valid = is_valid_dtype and is_valid_dims and is_valid_channels
        if not is_valid:
            logger.warning(f"Invalid numpy array: dtype={image.dtype}, dims={image.ndim}, shape={image.shape}")
        return is_valid
    
    logger.warning(f"Unsupported image type: {type(image)}")
    return False

@track_performance
def ensure_numpy_array(image: ImageTypes) -> "NDArray[np.uint8]":  # type: ignore
    """Convert various image types to a numpy array with GPU acceleration if available."""
    logger.debug(f"Converting image of type {type(image)} to numpy array")
    
    try:
        # Convert to numpy array first
        if isinstance(image, (str, Path)):
            logger.debug(f"Loading image from path: {image}")
            qimage = QImage(str(image))
            if qimage.isNull():
                error_msg = f"Failed to load image from {image}"
                logger.error(error_msg)
                raise ProcessingError(error_msg)
            arr = qimage_to_numpy(qimage)
        elif isinstance(image, QImage):
            arr = qimage_to_numpy(image)
        elif isinstance(image, np.ndarray):
            if not validate_image(image):
                error_msg = "Invalid numpy array format for image"
                logger.error(error_msg)
                raise ProcessingError(error_msg)
            arr = image
        else:
            error_msg = f"Unsupported image type: {type(image)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
            
        # Try GPU acceleration if available
        if HAS_GPU and arr.size > 1_000_000:  # Only use GPU for larger images
            try:
                logger.debug("Attempting GPU acceleration")
                gpu_arr = cp.asarray(arr)
                # Perform any GPU operations here
                arr = cp.asnumpy(gpu_arr)
                logger.debug("GPU acceleration successful")
            except Exception as e:
                logger.warning(f"GPU acceleration failed: {str(e)}")
                
        return cast("NDArray[np.uint8]", arr)  # type: ignore
        
    except Exception as e:
        logger.exception("Error in ensure_numpy_array")
        raise ProcessingError(f"Failed to convert image: {str(e)}") from e

@track_performance
def qimage_to_numpy(qimage: QImage) -> "NDArray[np.uint8]":  # type: ignore
    """Convert QImage to numpy array with optional GPU acceleration."""
    logger.debug("Converting QImage to numpy array")
    
    try:
        if qimage.isNull():
            error_msg = "Cannot convert null QImage"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
            
        # Get image dimensions
        width = qimage.width()
        height = qimage.height()
        logger.debug(f"Image dimensions: {width}x{height}")
        
        # Convert to RGBA format for consistent handling
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)  # type: ignore
        
        # Create a byte array to store the image data
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        qimage.save(buffer, "PNG")
        
        # Convert byte array to numpy array
        arr = np.frombuffer(byte_array.data(), np.uint8)  # type: ignore
        
        # Try GPU acceleration for decoding if available
        if HAS_GPU and arr.size > 1_000_000:
            try:
                logger.debug("Attempting GPU decoding")
                gpu_arr = cp.asarray(arr)
                # Use OpenCV GPU module if available
                if hasattr(cv2, 'cuda'):
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(arr)
                    arr = gpu_mat.download()
                else:
                    arr = cp.asnumpy(gpu_arr)
                logger.debug("GPU decoding successful")
            except Exception as e:
                logger.warning(f"GPU decoding failed: {str(e)}")
                arr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            arr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        
        # Convert RGBA to RGB if alpha channel is not needed
        if arr.ndim == 3 and arr.shape[2] == 4 and np.all(arr[:, :, 3] == 255):
            logger.debug("Converting RGBA to RGB")
            arr = arr[:, :, :3]
            
        return cast("NDArray[np.uint8]", arr)  # type: ignore
        
    except Exception as e:
        logger.exception("Error in qimage_to_numpy")
        raise ProcessingError(f"Failed to convert QImage to numpy array: {str(e)}") from e

def is_valid_image(image: Any) -> bool:
    """Alias for validate_image for backwards compatibility."""
    return validate_image(image)

def handle_processing_errors(func):
    """Decorator to handle processing errors."""
    def wrapper(*args, **kwargs):
        try:
            logger.debug(f"Executing {func.__name__}")
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Processing error in {func.__name__}: {str(e)}"
            logger.exception(error_msg)
            raise ProcessingError(error_msg) from e
    return wrapper

def get_performance_report() -> Dict[str, Dict[str, Optional[float]]]:
    """Get a comprehensive performance report."""
    return {
        op: metrics.get_average_metrics(op)
        for op in metrics.operation_times.keys()
    }

# Update __all__ to include new functions
__all__ = [
    'ImageTypes',
    'ImageArray',
    'ProcessingParams',
    'ProcessingError',
    'validate_image',
    'is_valid_image',
    'ensure_numpy_array',
    'qimage_to_numpy',
    'handle_processing_errors',
    'get_performance_report',
    'PerformanceMetrics',
    'track_performance'
] 