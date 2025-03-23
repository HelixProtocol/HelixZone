"""Type definitions for the GUI module."""

from __future__ import annotations

from typing import (
    TypeVar, Union, Optional, List, Tuple, cast, Dict, Any, Literal,
    TYPE_CHECKING, TypedDict, Protocol, runtime_checkable, Callable
)
from typing_extensions import TypeAlias, TypeGuard, Final
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from PyQt6.QtCore import QPoint, QPointF, pyqtSignal, pyqtSlot, Qt, QObject
from PyQt6.QtGui import (
    QImage, QResizeEvent, QPaintEvent, QAction, QKeySequence,
    QIcon, QColor
)
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QWidget
)
import os
from pathlib import Path
from PyQt6 import sip
from PyQt6.sip import voidptr
import cv2

from ..core.type_defs import EdgeDetectionParams

# Type checking configuration
if TYPE_CHECKING:
    from numpy import dtype, uint8, float32, bool_
    from PyQt6.QtGui import QImage
    from PyQt6.QtCore import pyqtBoundSignal

# Type aliases for PyQt types
SlotType = Union[Callable[..., Any], 'pyqtBoundSignal']

# Type variables with bounds
T = TypeVar('T', bound=np.generic)
ImageType = TypeVar('ImageType', bound=np.ndarray)
MaskType = TypeVar('MaskType', bound=np.ndarray)

# Type aliases with explicit types
MousePosition: TypeAlias = Union[QPoint, QPointF]
MouseDelta: TypeAlias = Union[QPoint, QPointF]

# Type aliases for numpy arrays with specific dtypes
RGBImage: TypeAlias = NDArray[np.uint8]  # Shape: (H, W, 3)
GrayscaleImage: TypeAlias = NDArray[np.uint8]  # Shape: (H, W)
ImageArray: TypeAlias = Union[RGBImage, GrayscaleImage]
FloatArray: TypeAlias = NDArray[np.float32]
MaskArray: TypeAlias = NDArray[np.bool_]
Array2D: TypeAlias = NDArray[np.float32]
Array3D: TypeAlias = NDArray[np.float32]
Array4D: TypeAlias = NDArray[np.float32]

# Protocol for array operations
class ArrayProtocol(Protocol):
    """Protocol for array operations."""
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...  # type: ignore
    def astype(self, dtype: Any, copy: bool = True) -> Any: ...  # type: ignore

# Custom type for numpy shape
Shape: TypeAlias = Tuple[int, ...]

# Feature types
Coordinates: TypeAlias = List[Tuple[int, int]]  # List of coordinate tuples

# Type alias for processing results with explicit key type
class ProcessingResults(TypedDict):
    """Type definition for image processing results.
    
    Attributes:
        original: Original input RGB image as numpy array
        processed: Processed output image as numpy array
        parameters: Dictionary of processing parameters
    """
    original: np.ndarray  # type: ignore
    processed: np.ndarray  # type: ignore
    parameters: Dict[str, Any]

# Constants for file dialogs
IMAGE_FILTER: Final[str] = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
)
SAVE_FILTER: Final[str] = (
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
)

# Type guards for runtime type checking
def is_rgb_image(arr: NDArray[Any]) -> TypeGuard[NDArray[np.uint8]]:
    """Type guard to check if array is a valid RGB image."""
    return (isinstance(arr.dtype, np.dtype) and arr.dtype.type == np.uint8 and 
            isinstance(arr.ndim, int) and arr.ndim >= 2 and
            isinstance(arr.shape[-1], int) and arr.shape[-1] == 3)

def is_float_mask(arr: NDArray[Any]) -> TypeGuard[NDArray[np.float32]]:
    """Type guard to check if array is a valid float mask."""
    return isinstance(arr.dtype, np.dtype) and arr.dtype.type == np.float32

@dataclass
class CanvasState:
    """State of the canvas widget."""
    scale_factor: float = 1.0
    last_pan: QPointF = field(default_factory=lambda: QPointF(0, 0))
    pan_start: QPointF = field(default_factory=lambda: QPointF(0, 0))
    panning: bool = False
    selection_mask: Optional[NDArray[np.bool_]] = field(default=None)
    current_image: Optional[NDArray[np.uint8]] = None

def create_empty_image(width: int, height: int) -> NDArray[np.uint8]:
    """Create an empty RGB image with the specified dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    return np.zeros((height, width, 3), dtype=np.uint8)

def create_empty_mask(width: int, height: int) -> NDArray[np.bool_]:
    """Create an empty boolean mask with the specified dimensions."""
    return np.zeros((height, width), dtype=np.bool_)

@runtime_checkable
class Buffer(Protocol):
    """Protocol for buffer objects."""
    def __buffer__(self) -> memoryview: ...

def convert_qimage_to_numpy(qimage: QImage) -> RGBImage:
    """Convert QImage to numpy array."""
    if qimage.isNull():
        raise ValueError("Cannot convert null QImage")
        
    # Convert to RGB888 format if needed
    if qimage.format() != QImage.Format.Format_RGB888:
        # Use explicit type for format conversion
        format_rgb = QImage.Format.Format_RGB888
        flags = Qt.ImageConversionFlag.AutoColor
        qimage = qimage.convertToFormat(format_rgb, flags)
        
    # Get dimensions
    width = qimage.width()
    height = qimage.height()
    
    # Get image data using sip
    ptr = qimage.constBits()
    if ptr is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create numpy array from image data
    ptr.setsize(height * width * 3)  # 3 channels for RGB
    # Use sip.voidptr directly as a buffer
    arr = np.array(ptr, copy=True).reshape(height, width, 3)
    
    # Ensure array is contiguous
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
        
    return arr

def convert_numpy_to_qimage(arr: NDArray[np.uint8]) -> QImage:
    """Convert a numpy array to QImage."""
    if not is_rgb_image(arr):
        raise ValueError("Array must be a valid RGB image")
    height, width, channels = arr.shape
    bytes_per_line = channels * width
    return QImage(arr.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

class MainWindow(QMainWindow):
    """Main window class with properly typed signals and slots."""
    
    # Define signals with proper types
    image_loaded = pyqtSignal(np.ndarray)
    processing_finished = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.current_image: Optional[RGBImage] = None
        self.current_results: Optional[ProcessingResults] = None
        
        # Initialize actions
        self.open_action = QAction("&Open", self)
        self.save_action = QAction("&Save", self)
        
        self.setup_ui()
        self.setup_actions()
        self.setup_connections()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        pass  # To be implemented by subclasses

    def setup_actions(self) -> None:
        """Set up menu actions with proper typing."""
        # Configure open action
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setStatusTip("Open an image file")
        self.open_action.triggered.connect(
            cast(SlotType, self._on_open_triggered)
        )
        
        # Configure save action
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setStatusTip("Save the current image")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(
            cast(SlotType, self._on_save_triggered)
        )

    def setup_connections(self) -> None:
        """Connect signals with proper type hints."""
        self.image_loaded.connect(
            cast(SlotType, self.on_image_loaded)
        )
        self.processing_finished.connect(
            cast(SlotType, self.on_processing_finished)
        )

    def _show_error(self, title: str, message: str) -> None:
        """Show error message box."""
        QMessageBox.critical(self, title, message)

    @pyqtSlot()
    def _on_open_triggered(self) -> None:
        """Handle open action triggered signal."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", IMAGE_FILTER
        )
        
        if file_path:
            try:
                image = QImage(file_path)
                if not image.isNull():
                    arr = convert_qimage_to_numpy(image)
                    self.image_loaded.emit(arr)
                else:
                    self._show_error(
                        "Error",
                        f"Failed to load image: {Path(file_path).name}"
                    )
            except Exception as e:
                self._show_error("Error", f"Error loading image: {str(e)}")

    @pyqtSlot()
    def _on_save_triggered(self) -> None:
        """Handle save action triggered signal."""
        if self.current_image is None:
            return
            
        file_path, filter_used = QFileDialog.getSaveFileName(
            self, "Save Image", "", SAVE_FILTER
        )
        
        if file_path:
            try:
                image = convert_numpy_to_qimage(self.current_image)
                if not image.save(file_path):
                    self._show_error(
                        "Error",
                        f"Failed to save image: {Path(file_path).name}"
                    )
            except Exception as e:
                self._show_error("Error", f"Error saving image: {str(e)}")

    @pyqtSlot(np.ndarray)
    def on_image_loaded(self, image: RGBImage) -> None:
        """Handle loaded image with proper typing."""
        if not is_rgb_image(image):
            self._show_error("Error", "Invalid image format")
            return
        self.current_image = image
        self.save_action.setEnabled(True)
        self.update_display()

    @pyqtSlot(dict)
    def on_processing_finished(self, results: ProcessingResults) -> None:
        """Handle processing results with proper typing.
        
        Args:
            results: Dictionary containing original image, processed result and parameters
        """
        if not isinstance(results, dict):
            self._show_error("Error", "Invalid processing results format")
            return
            
        if not all(key in results for key in ('original', 'processed', 'parameters')):
            self._show_error("Error", "Missing required keys in processing results")
            return
            
        if not (is_rgb_image(results['original']) and is_rgb_image(results['processed'])):
            self._show_error("Error", "Invalid image format in processing results")
            return
            
        self.current_results = results
        self.update_display()

    def process_single_image(
        self,
        image: np.ndarray,
        params: EdgeDetectionParams
    ) -> ProcessingResults:
        """Process single image with proper typing.
        
        Args:
            image: Input RGB image as numpy array
            params: Edge detection parameters
            
        Returns:
            Dictionary containing original image, processed result and parameters
            
        Raises:
            ValueError: If input image is not a valid RGB image
        """
        if not is_rgb_image(image):
            raise ValueError("Input must be an RGB image")
            
        results: ProcessingResults = {
            'original': image,
            'processed': np.copy(image),  # Placeholder for actual processing
            'parameters': params.__dict__
        }
        return results

    def update_display(self) -> None:
        """Update the display with current image and results."""
        pass  # To be implemented by subclasses
        
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle window resize events."""
        super().resizeEvent(event)
        self.update_display()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Handle paint events."""
        super().paintEvent(event)
        self.update_display()

__all__ = [
    'MousePosition',
    'MouseDelta',
    'ImageArray',
    'FloatArray',
    'MaskArray',
    'Shape',
    'CanvasState',
    'ImageType',
    'MaskType',
    'Coordinates',
    'ProcessingResults',
    'MainWindow',
    'create_empty_image',
    'create_empty_mask',
    'convert_qimage_to_numpy',
    'convert_numpy_to_qimage',
    'is_rgb_image',
    'is_float_mask',
] 