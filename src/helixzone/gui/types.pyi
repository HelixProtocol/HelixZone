"""Type stub file for the GUI module."""

from __future__ import annotations

from typing import (
    TypeVar, Union, Optional, List, Tuple, Dict, Any, Literal,
    TYPE_CHECKING, TypedDict, Protocol, runtime_checkable, overload,
    Callable, ClassVar, Generic
)
from typing_extensions import TypeAlias, TypeGuard, Final
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
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

from ..core.type_defs import EdgeDetectionParams

# Type variables with bounds
T = TypeVar('T', bound=np.generic)
ImageType = TypeVar('ImageType', bound=np.ndarray)
MaskType = TypeVar('MaskType', bound=np.ndarray)

# Type aliases with explicit types
MousePosition: TypeAlias = Union[QPoint, QPointF]
MouseDelta: TypeAlias = Union[QPoint, QPointF]

# Type aliases for numpy arrays with specific dtypes
ImageArray: TypeAlias = "NDArray[np.uint8]"  # RGB image array (height, width, 3)
FloatArray: TypeAlias = "NDArray[np.float32]"  # Float array
MaskArray: TypeAlias = "NDArray[np.bool_]"   # Boolean mask array
Array2D: TypeAlias = "NDArray[np.float32]"  # 2D array (height, width)
Array3D: TypeAlias = "NDArray[np.uint8]"  # 3D array (height, width, channels)
Array4D: TypeAlias = "NDArray[np.float32]"  # 4D array (batch, height, width, channels)

# Protocol for array operations
@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for array operations."""
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> "np.dtype[Any]": ...
    def astype(self, dtype: "np.dtype[Any]", copy: bool = True) -> "NDArray[Any]": ...

# Custom type for numpy shape
Shape: TypeAlias = Tuple[int, ...]

# Feature types
Coordinates: TypeAlias = List[Tuple[int, int]]  # List of coordinate tuples

# Type alias for processing results with explicit key type
class ProcessingResults(TypedDict):
    """Type definition for image processing results."""
    original: "NDArray[np.uint8]"
    processed: "NDArray[np.uint8]"
    parameters: Dict[str, Any]

# Constants for file dialogs
IMAGE_FILTER: Final[str] = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
SAVE_FILTER: Final[str] = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"

# Type guards for runtime type checking
def is_rgb_image(arr: "NDArray[Any]") -> TypeGuard["NDArray[np.uint8]"]: ...
def is_float_mask(arr: "NDArray[Any]") -> TypeGuard["NDArray[np.float32]"]: ...

@dataclass
class CanvasState:
    """State of the canvas widget."""
    scale_factor: float
    last_pan: QPointF
    pan_start: QPointF
    panning: bool
    selection_mask: Optional["NDArray[np.bool_]"]

def create_empty_image(width: int, height: int) -> "NDArray[np.uint8]": ...
def convert_qimage_to_numpy(qimage: QImage) -> "NDArray[np.uint8]": ...
def convert_numpy_to_qimage(array: "NDArray[np.uint8]") -> QImage: ...

class MainWindow(QMainWindow):
    """Main window class with properly typed signals and slots."""
    
    # Define signals with proper types
    image_loaded: ClassVar[pyqtSignal]
    processing_finished: ClassVar[pyqtSignal]
    
    current_image: Optional["NDArray[np.uint8]"]
    current_results: Optional[ProcessingResults]
    open_action: QAction
    save_action: QAction
    
    def __init__(self, parent: Optional[QWidget] = None) -> None: ...
    def setup_ui(self) -> None: ...
    def setup_actions(self) -> None: ...
    def setup_connections(self) -> None: ...
    def _show_error(self, title: str, message: str) -> None: ...
    
    @pyqtSlot()
    def _on_open_triggered(self) -> None: ...
    
    @pyqtSlot()
    def _on_save_triggered(self) -> None: ...
    
    @pyqtSlot(np.ndarray)
    def on_image_loaded(self, image: "NDArray[np.uint8]") -> None: ...
    
    @pyqtSlot(dict)
    def on_processing_finished(self, results: ProcessingResults) -> None: ...
    
    def process_single_image(
        self,
        image: "NDArray[np.uint8]",
        params: EdgeDetectionParams
    ) -> ProcessingResults: ...
    
    def update_display(self) -> None: ...
    def resizeEvent(self, event: QResizeEvent) -> None: ...
    def paintEvent(self, event: QPaintEvent) -> None: ... 