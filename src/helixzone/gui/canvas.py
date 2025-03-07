from PyQt6.QtWidgets import QWidget, QScrollArea
from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QTransform
from ..core.layer import LayerStack
from ..core.tools import ToolManager
from ..core.commands import CommandStack, DrawCommand
import numpy as np

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize command stack first
        self.command_stack = CommandStack()
        
        # Initialize layer stack with reference to self
        self.layer_stack = LayerStack()
        self.layer_stack.canvas = self  # Add reference to canvas
        
        # Add initial background layer
        self.layer_stack.add_layer(name="Background")
        
        # Initialize tool manager
        self.tool_manager = ToolManager(self)
        
        self.scale_factor = 1.0
        self.pan_start = QPoint()
        self.last_pan = QPoint()
        self.panning = False
        
        # Current drawing command (for continuous strokes)
        self.current_draw_command = None
        
        # Enable mouse tracking for smooth panning and drawing
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def set_image(self, image):
        """Set image to the active layer."""
        active_layer = self.layer_stack.get_active_layer()
        if active_layer:
            if isinstance(image, QImage):
                active_layer.set_image(image)
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack((image,) * 3, axis=-1)
                bytes_per_line = 3 * width
                qimage = QImage(image.data, width, height,
                              bytes_per_line, QImage.Format.Format_RGB888)
                active_layer.set_image(qimage)
            elif isinstance(image, str):
                active_layer.set_image(QImage(image))
            self.update()
        
    def get_image(self):
        """Get the merged image of all layers."""
        return self.layer_stack.merge_visible()
        
    def get_transformed_pos(self, pos):
        """Convert screen coordinates to image coordinates."""
        # Create inverse transform
        transform = QTransform()
        transform.scale(self.scale_factor, self.scale_factor)
        transform.translate(self.last_pan.x() / self.scale_factor,
                          self.last_pan.y() / self.scale_factor)
        inverse_transform, invertible = transform.inverted()
        
        if invertible:
            return inverse_transform.map(pos)
        return pos
        
    def paintEvent(self, event):
        painter = QPainter(self)
        # Enable antialiasing for smoother rendering
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Apply transformations
        transform = QTransform()
        transform.scale(self.scale_factor, self.scale_factor)
        transform.translate(self.last_pan.x() / self.scale_factor,
                          self.last_pan.y() / self.scale_factor)
        painter.setTransform(transform)
        
        # Draw the merged layers
        merged_image = self.layer_stack.merge_visible()
        if merged_image:
            painter.drawImage(0, 0, merged_image)
        
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel."""
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.scale_factor *= factor
        # Limit zoom range
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Handle panning
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            # Start new draw command
            active_layer = self.layer_stack.get_active_layer()
            if active_layer:
                self.current_draw_command = DrawCommand(active_layer, active_layer.image)
            
            # Handle tool
            transformed_pos = self.get_transformed_pos(event.pos())
            self.tool_manager.get_current_tool().mouse_press(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Handle panning
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Handle tool
            transformed_pos = self.get_transformed_pos(event.pos())
            self.tool_manager.get_current_tool().mouse_release(event)
            
            # Finish draw command
            if self.current_draw_command:
                self.current_draw_command.execute()
                self.command_stack.push(self.current_draw_command)
                self.current_draw_command = None
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.panning:
            # Handle panning
            delta = event.pos() - self.pan_start
            self.last_pan += delta
            self.pan_start = event.pos()
            self.update()
        else:
            # Handle tool
            transformed_pos = self.get_transformed_pos(event.pos())
            self.tool_manager.get_current_tool().mouse_move(event)
            
    def sizeHint(self):
        """Suggest a default size."""
        return QSize(800, 600)
    
    def undo(self):
        """Undo the last command."""
        self.command_stack.undo()
        self.update()
    
    def redo(self):
        """Redo the last undone command."""
        self.command_stack.redo()
        self.update()

class CanvasView(QScrollArea):
    """A scroll area container for the canvas widget."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = Canvas()
        self.setWidget(self.canvas)
        self.setWidgetResizable(True)
        # Enable scroll bars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded) 