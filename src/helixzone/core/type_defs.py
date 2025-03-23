# ruff: noqa: F401
"""Type definitions for core functionality."""

from __future__ import annotations

from typing import (
    TypeVar, Union, Optional, List, Tuple, Dict, Any,
    TYPE_CHECKING, Protocol, runtime_checkable, overload, cast, Literal
)
from typing_extensions import TypeAlias
import numpy as np
from numpy.typing import NDArray
import cv2

# Type variables for image types
_TImg = TypeVar('_TImg', np.uint8, np.float32)
_TScalar = TypeVar('_TScalar', np.uint8, np.float32, np.bool_)

# OpenCV Mat type aliases
Mat: TypeAlias = Union[cv2.Mat, cv2.UMat]  # type: ignore
CVMat: TypeAlias = cv2.UMat  # type: ignore
CVMatLike: TypeAlias = Union[cv2.UMat, NDArray[Any]]  # type: ignore
MatLike: TypeAlias = Union[Mat, NDArray[Any]]
GaussianBlurInput: TypeAlias = Union[NDArray[np.uint8], NDArray[np.float32]]

# Image type aliases
ImageFloat: TypeAlias = NDArray[np.float32]
ImageUInt8: TypeAlias = NDArray[np.uint8]
ImageBool: TypeAlias = NDArray[np.bool_]
ImageArray: TypeAlias = Union[ImageFloat, ImageUInt8]
GrayImage: TypeAlias = Union[NDArray[np.uint8], NDArray[np.float32]]
FloatArray: TypeAlias = NDArray[np.float32]
Array: TypeAlias = NDArray[Any]

# Source for images - path, array, or matrix
ImageSource: TypeAlias = Union[str, ImageArray, Mat]

# GPU types
GPUBackend = Literal['cuda', 'opencl', 'cpu']
ColorMode = Literal['rgb', 'bgr', 'grayscale']
EdgeDetectionParams = Dict[str, Any]
GPUMetrics = Dict[str, float]
MetricsDict = Dict[str, Any]
BatchSizerConfig = Dict[str, Any]
ProcessingResults = Dict[str, Any]
GPUDevice = Dict[str, Any]

# Constants
MEMORY_CRITICAL_THRESHOLD: float = 95.0
MEMORY_WARNING_THRESHOLD: float = 85.0

# Exceptions
class GPUMemoryError(Exception):
    """Raised when GPU memory is critically low."""
    pass

# Protocol for array-like objects
@runtime_checkable
class ArrayLike(Protocol[_TScalar]):
    """Protocol for array-like objects."""
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[_TScalar]: ...
    def astype(self, dtype: Any, copy: bool = True) -> NDArray[Any]: ...

# Image type checking functions
def is_image_array(img: Any) -> bool:
    """Check if object is a valid image array (numpy array or OpenCV Mat)."""
    if isinstance(img, (cv2.Mat, cv2.UMat)):  # type: ignore
        return True
    if isinstance(img, np.ndarray):
        if img.ndim in (2, 3) and img.dtype in (np.uint8, np.float32):
            return True
    return False

def is_gray_image(img: Any) -> bool:
    """Check if object is a grayscale image."""
    if not is_image_array(img):
        return False
    if isinstance(img, np.ndarray):
        return img.ndim == 2
    return len(img.shape) == 2  # type: ignore

def is_color_image(img: Any) -> bool:
    """Check if object is a color image."""
    if not is_image_array(img):
        return False
    if isinstance(img, np.ndarray):
        return img.ndim == 3 and img.shape[2] in (3, 4)
    return len(img.shape) == 3 and img.shape[2] in (3, 4)  # type: ignore

@overload
def to_mat(img: NDArray[np.uint8]) -> Mat: ...

@overload
def to_mat(img: NDArray[np.float32]) -> Mat: ...

@overload
def to_mat(img: Mat) -> Mat: ...

def to_mat(img: Union[NDArray[Any], Mat]) -> Mat:
    """Convert numpy array to OpenCV Mat."""
    if isinstance(img, (cv2.Mat, cv2.UMat)):  # type: ignore
        return img
    return cv2.UMat(img) if cv2.ocl.useOpenCL() else cv2.Mat(img)  # type: ignore

@overload
def to_float_img(img: NDArray[np.uint8]) -> ImageFloat: ...

@overload
def to_float_img(img: NDArray[np.float32]) -> ImageFloat: ...

@overload
def to_float_img(img: Mat) -> ImageFloat: ...

def to_float_img(img: Union[NDArray[Any], Mat]) -> ImageFloat:
    """Convert image to float32 numpy array."""
    if isinstance(img, (cv2.Mat, cv2.UMat)):  # type: ignore
        img = img.get() if isinstance(img, cv2.UMat) else np.asarray(img)  # type: ignore
    return img.astype(np.float32)

@overload
def to_uint8_img(img: NDArray[np.uint8]) -> ImageUInt8: ...

@overload
def to_uint8_img(img: NDArray[np.float32]) -> ImageUInt8: ...

@overload
def to_uint8_img(img: Mat) -> ImageUInt8: ...

def to_uint8_img(img: Union[NDArray[Any], Mat]) -> ImageUInt8:
    """Convert image to uint8 numpy array."""
    if isinstance(img, (cv2.Mat, cv2.UMat)):  # type: ignore
        img = img.get() if isinstance(img, cv2.UMat) else np.asarray(img)  # type: ignore
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

@overload
def ensure_mat(img: NDArray[np.uint8]) -> Mat: ...

@overload
def ensure_mat(img: NDArray[np.float32]) -> Mat: ...

@overload
def ensure_mat(img: Mat) -> Mat: ...

def ensure_mat(img: Union[NDArray[Any], Mat]) -> Mat:
    """Convert numpy array to OpenCV Mat if needed."""
    return to_mat(img)

@overload
def ensure_array(img: NDArray[np.uint8], dtype: None = None) -> NDArray[np.uint8]: ...

@overload
def ensure_array(img: NDArray[np.float32], dtype: None = None) -> NDArray[np.float32]: ...

@overload
def ensure_array(img: Mat, dtype: None = None) -> NDArray[Any]: ...

@overload
def ensure_array(img: Union[NDArray[Any], Mat], dtype: type[np.uint8]) -> NDArray[np.uint8]: ...

@overload
def ensure_array(img: Union[NDArray[Any], Mat], dtype: type[np.float32]) -> NDArray[np.float32]: ...

def ensure_array(img: Union[NDArray[Any], Mat], dtype: Any = None) -> NDArray[Any]:
    """Convert OpenCV Mat to numpy array if needed."""
    if isinstance(img, (cv2.Mat, cv2.UMat)):  # type: ignore
        img = img.get() if isinstance(img, cv2.UMat) else np.asarray(img)  # type: ignore
    return img.astype(dtype) if dtype is not None else img

@overload
def ensure_float32(img: NDArray[np.uint8]) -> ImageFloat: ...

@overload
def ensure_float32(img: NDArray[np.float32]) -> ImageFloat: ...

@overload
def ensure_float32(img: Mat) -> ImageFloat: ...

def ensure_float32(img: Union[NDArray[Any], Mat]) -> ImageFloat:
    """Ensure image is float32 numpy array."""
    return to_float_img(img)

@overload
def ensure_uint8(img: NDArray[np.uint8]) -> ImageUInt8: ...

@overload
def ensure_uint8(img: NDArray[np.float32]) -> ImageUInt8: ...

@overload
def ensure_uint8(img: Mat) -> ImageUInt8: ...

def ensure_uint8(img: Union[NDArray[Any], Mat]) -> ImageUInt8:
    """Ensure image is uint8 numpy array."""
    return to_uint8_img(img)

__all__ = [
    'Mat',
    'CVMat',
    'CVMatLike',
    'MatLike',
    'GaussianBlurInput',
    'ImageFloat',
    'ImageUInt8',
    'ImageBool',
    'ImageArray',
    'GrayImage',
    'FloatArray',
    'Array',
    'ImageSource',
    'GPUBackend',
    'ColorMode',
    'EdgeDetectionParams',
    'GPUMetrics',
    'MetricsDict',
    'BatchSizerConfig',
    'ProcessingResults',
    'GPUDevice',
    'GPUMemoryError',
    'MEMORY_CRITICAL_THRESHOLD',
    'MEMORY_WARNING_THRESHOLD',
    'ArrayLike',
    'is_image_array',
    'is_gray_image',
    'is_color_image',
    'to_mat',
    'to_float_img',
    'to_uint8_img',
    'ensure_mat',
    'ensure_array',
    'ensure_float32',
    'ensure_uint8',
] 