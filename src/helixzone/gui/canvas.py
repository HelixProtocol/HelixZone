from PyQt6.QtWidgets import QWidget, QScrollArea
from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QImage, QPixmap, QTransform, QPen, QColor, QPainterPath
from ..core.layer import LayerStack
from ..core.tool_manager import ToolManager
from ..core.commands import CommandStack, DrawCommand
import numpy as np
import cv2
from typing import Optional, cast, Union

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
        
        # Selection mask
        self._selection_mask = None
        
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
        
    def get_transformed_pos(self, pos: Union[QPoint, QPointF]) -> QPointF:
        """Convert screen coordinates to image coordinates."""
        try:
            # Convert to QPointF if needed
            if isinstance(pos, QPoint):
                pos = QPointF(pos)
            elif not isinstance(pos, QPointF):
                return QPointF(0, 0)  # Return safe default if pos is invalid
            
            # Create inverse transform
            transform = QTransform()
            transform.scale(self.scale_factor, self.scale_factor)
            transform.translate(self.last_pan.x() / self.scale_factor,
                            self.last_pan.y() / self.scale_factor)
            inverse_transform, invertible = transform.inverted()
            
            if invertible:
                transformed_pos = inverse_transform.map(pos)
                # Ensure the transformed position is within valid bounds
                x = max(0.0, min(transformed_pos.x(), float(self.width() - 1)))
                y = max(0.0, min(transformed_pos.y(), float(self.height() - 1)))
                return QPointF(x, y)
            return QPointF(pos)  # Return safe copy if transform not invertible
        except Exception as e:
            print(f"Error in get_transformed_pos: {e}")
            return QPointF(0, 0)
        
    def get_selection(self) -> Optional[np.ndarray]:
        """Get the current selection mask."""
        return self._selection_mask

    def set_selection(self, mask: np.ndarray) -> None:
        """Set the selection mask."""
        self._selection_mask = mask.astype(np.float32)
        self.update()

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self._selection_mask = None
        self.update()

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
        
        # Draw selection overlay if exists
        if self._selection_mask is not None:
            # Create a semi-transparent overlay
            overlay = QImage(self.width(), self.height(), QImage.Format.Format_ARGB32)
            overlay.fill(Qt.GlobalColor.transparent)
            
            # Draw selection outline
            overlay_painter = QPainter(overlay)
            overlay_painter.setPen(QPen(QColor(0, 120, 215, 128), 1))  # Semi-transparent blue
            overlay_painter.setBrush(QColor(0, 120, 215, 32))  # Very transparent blue
            
            # Convert mask to path
            path = QPainterPath()
            contours = cv2.findContours(
                (self._selection_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[0]
            
            for contour in contours:
                # Convert contour points to QPointF
                points = []
                contour_array = cast(np.ndarray, contour)
                for point in contour_array:
                    x = float(point[0][0])
                    y = float(point[0][1])
                    points.append(QPointF(x, y))
                
                if points:
                    path.moveTo(points[0])
                    for point in points[1:]:
                        path.lineTo(point)
                    path.lineTo(points[0])  # Close the path
            
            overlay_painter.drawPath(path)
            overlay_painter.end()
            
            # Draw the overlay
            painter.drawImage(0, 0, overlay)
        
        # Draw tool preview (e.g., selection outlines)
        current_tool = self.tool_manager.get_current_tool()
        if current_tool:
            current_tool.draw_preview(painter)
        
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
            # Handle tool
            self.tool_manager.get_current_tool().mouse_press(event)
            self.update()  # Request repaint to show preview
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Handle panning
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Handle tool
            self.tool_manager.get_current_tool().mouse_release(event)
            self.update()  # Request repaint to show final result
            
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
            self.tool_manager.get_current_tool().mouse_move(event)
            self.update()  # Request repaint to show preview
            
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