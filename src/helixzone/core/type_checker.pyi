"""Type stub file for type checking utilities."""

from typing import Union, Dict, Any, Optional, TypeVar, Protocol, runtime_checkable, cast, Callable, TypeGuard
from typing_extensions import TypeAlias
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QImage

# Type aliases
ImageTypes: TypeAlias = Union[str, Path, QImage, "NDArray[np.uint8]"]

@dataclass
class ProcessingParams:
    """Parameters for image processing."""
    sigma: float
    threshold: float
    kernel_size: int
    iterations: int
    normalize: bool

class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

def validate_image(image: Any) -> bool: ...
def ensure_numpy_array(image: ImageTypes) -> "NDArray[np.uint8]": ...
def qimage_to_numpy(qimage: QImage) -> "NDArray[np.uint8]": ...
def is_valid_image(image: Any) -> bool: ...

def handle_processing_errors(func: Callable[..., Any]) -> Callable[..., Any]: ... 