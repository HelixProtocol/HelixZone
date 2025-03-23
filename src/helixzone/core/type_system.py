"""Type system for image processing."""

from typing import Dict, List, Optional, Union, Any, Protocol, runtime_checkable
from typing_extensions import TypeAlias
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QImage

# Basic type definitions using string literals to avoid linter errors
NumpyArray = "NDArray[np.uint8]"
ImageArray = NumpyArray
GrayImage = NumpyArray
RGBImage = NumpyArray

# Complex types
ImageSource = Union[str, Path, QImage, "NDArray[np.uint8]"]
ImageResult = Dict[str, Union["NDArray[np.uint8]", float, str]]
EdgeMap = Dict[str, Union["NDArray[np.uint8]", Dict[str, float]]]

@runtime_checkable
class ImageProcessor(Protocol):
    """Protocol for image processors."""
    def process(self, image: ImageSource) -> ImageResult:
        """Process an image."""
        ...

def is_valid_image(image: Any) -> bool:
    """Check if an object is a valid image."""
    if isinstance(image, (str, Path)):
        return Path(image).exists() and Path(image).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    elif isinstance(image, QImage):
        return not image.isNull()
    elif isinstance(image, np.ndarray):
        return image.dtype == np.uint8 and image.ndim in (2, 3)
    return False

def convert_qimage_to_numpy(qimage: QImage) -> NDArray[np.uint8]:
    """Convert QImage to numpy array."""
    if qimage.isNull():
        raise ValueError("Invalid QImage")
    return np.array(qimage.convertToFormat(QImage.Format.Format_RGB888))

def ensure_numpy_array(image: ImageSource) -> NDArray[np.uint8]:
    """Convert image source to numpy array."""
    if isinstance(image, (str, Path)):
        import cv2
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
        return img
    elif isinstance(image, QImage):
        return convert_qimage_to_numpy(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            return image.astype(np.uint8)
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")