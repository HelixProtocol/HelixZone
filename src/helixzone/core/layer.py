from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..gui.canvas import Canvas

class Layer(QObject):
    """A single layer in the image editor."""
    
    # Signals for layer changes
    changed = pyqtSignal()  # Emitted when layer content changes
    properties_changed = pyqtSignal()  # Emitted when properties (opacity, visibility) change
    
    def __init__(self, name: str = "Layer", size: tuple[int, int] = (800, 600)):
        super().__init__()
        self.name = name
        self.visible = True
        self.opacity = 1.0  # 0.0 to 1.0
        self.blend_mode = "normal"
        
        # Create transparent image for the layer
        self.image = QImage(size[0], size[1], QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.transparent)
    
    def set_image(self, image: QImage | np.ndarray) -> None:
        """Set the layer's image content."""
        if isinstance(image, QImage):
            self.image = image
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            if len(image.shape) == 2:  # Grayscale
                image = np.stack((image,) * 3, axis=-1)
            bytes_per_line = 3 * width
            self.image = QImage(image.data, width, height,
                              bytes_per_line, QImage.Format.Format_RGB888)
        self.changed.emit()
    
    def get_image(self) -> QImage:
        """Get the layer's image content."""
        return self.image
    
    def set_opacity(self, opacity: float) -> None:
        """Set layer opacity (0.0 to 1.0)."""
        self.opacity = max(0.0, min(1.0, opacity))
        self.properties_changed.emit()
    
    def set_visible(self, visible: bool) -> None:
        """Set layer visibility."""
        self.visible = visible
        self.properties_changed.emit()
    
    def set_blend_mode(self, mode: str) -> None:
        """Set layer blend mode."""
        self.blend_mode = mode
        self.properties_changed.emit()
    
    def clear(self) -> None:
        """Clear the layer to transparency."""
        self.image.fill(Qt.GlobalColor.transparent)
        self.changed.emit()
    
    def resize(self, width: int, height: int) -> None:
        """Resize the layer."""
        self.image = self.image.scaled(width, height, 
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.changed.emit()

class LayerStack:
    """Manages a stack of layers."""
    
    def __init__(self):
        self.layers: list[Layer] = []
        self.active_layer_index: int = -1
        self._canvas: Optional['Canvas'] = None
    
    @property
    def canvas(self) -> Optional['Canvas']:
        """Get the associated canvas."""
        return self._canvas
    
    @canvas.setter
    def canvas(self, value: Optional['Canvas']) -> None:
        """Set the associated canvas."""
        self._canvas = value
    
    def add_layer(self, layer: Optional[Layer] = None, name: Optional[str] = None, size: tuple[int, int] = (800, 600)) -> Layer:
        """Add a new layer to the stack."""
        if layer is None:
            layer = Layer(name or f"Layer {len(self.layers) + 1}", size)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        return layer
    
    def remove_layer(self, index: int) -> None:
        """Remove a layer from the stack."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self.active_layer_index = min(self.active_layer_index,
                                        len(self.layers) - 1)
    
    def move_layer(self, from_index: int, to_index: int) -> None:
        """Move a layer to a new position in the stack."""
        if 0 <= from_index < len(self.layers) and 0 <= to_index < len(self.layers):
            layer = self.layers.pop(from_index)
            self.layers.insert(to_index, layer)
            if self.active_layer_index == from_index:
                self.active_layer_index = to_index
    
    def get_active_layer(self) -> Optional[Layer]:
        """Get the currently active layer."""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
    
    def set_active_layer(self, index: int) -> None:
        """Set the active layer by index."""
        if 0 <= index < len(self.layers):
            self.active_layer_index = index
    
    def merge_visible(self) -> Optional[QImage]:
        """Merge all visible layers into a single image."""
        if not self.layers:
            return None
            
        # Create a new image with the same size as the first layer
        result = QImage(self.layers[0].image.size(), QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Paint each visible layer
        for layer in self.layers:
            if layer.visible:
                painter.setOpacity(layer.opacity)
                painter.drawImage(0, 0, layer.image)
        
        painter.end()
        return result 