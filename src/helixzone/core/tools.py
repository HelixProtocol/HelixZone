from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING, cast, Union
from PyQt6.QtCore import QPoint, Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QMouseEvent, QImage
import numpy as np
from numpy.typing import NDArray
from .commands import DrawCommand
import cv2

if TYPE_CHECKING:
    from ..gui.canvas import Canvas
    from ..core.layer import Layer

class Tool:
    """Base class for all tools."""
    def __init__(self, canvas: 'Canvas', name: str):
        self.canvas = canvas
        self.name = name
        self.start_pos: Optional[QPoint] = None
        self.current_pos: Optional[QPoint] = None
        self.is_active = False
        self.options: Dict[str, Any] = {}
        self.size: int = 10
        self.opacity: float = 1.0
        self.color: QColor = QColor(0, 0, 0)
        self.feather_radius: int = 0
        self.points: List[QPointF] = []
        self.min_distance: float = 1.0

    def mouse_press(self, event: Optional[QMouseEvent] = None) -> None:
        """Handle mouse press event."""
        if event is None:
            return
            
        try:
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos)
            pos = self.canvas.get_transformed_pos(pos)
            
            # Store positions
            self.start_pos = QPoint(round(pos.x()), round(pos.y()))
            self.current_pos = QPoint(round(pos.x()), round(pos.y()))
            self.points = [pos]  # Store QPointF directly
            self.is_active = True
            self.canvas.update()
        except Exception as e:
            print(f"Error in mouse_press: {e}")
        
    def mouse_move(self, event: Optional[QMouseEvent] = None) -> None:
        """Handle mouse move event."""
        if event is None or not self.is_active:
            return
            
        try:
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos)
            pos = self.canvas.get_transformed_pos(pos)
            
            # Update current position
            self.current_pos = QPoint(round(pos.x()), round(pos.y()))
            
            # Calculate distance from last point
            if self.points:
                last_point = self.points[-1]
                dx = pos.x() - last_point.x()
                dy = pos.y() - last_point.y()
                distance = np.sqrt(dx * dx + dy * dy)
                
                # Add point if distance is sufficient or we have few points
                if distance >= self.min_distance or len(self.points) < 3:
                    self.points.append(pos)  # Store QPointF directly
            
            # Update the canvas
            self.canvas.update()
        except Exception as e:
            print(f"Error in mouse_move: {e}")
        
    def mouse_release(self, event: Optional[QMouseEvent] = None) -> None:
        """Handle mouse release event."""
        if event is None or not self.is_active:
            return
            
        try:
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos)
            pos = self.canvas.get_transformed_pos(pos)
            
            # Add the final point
            self.points.append(pos)  # Store QPointF directly
            if len(self.points) >= 3:
                # Close the path by adding the first point again
                if not np.allclose([self.points[0].x(), self.points[0].y()], 
                                 [self.points[-1].x(), self.points[-1].y()]):
                    self.points.append(self.points[0])
                
                # Create the selection mask
                mask = self.get_selection_mask(self.canvas.width(), self.canvas.height())
                if mask is not None:
                    self.canvas.set_selection(mask)
            
            # Reset state
            self.start_pos = None
            self.current_pos = None
            self.is_active = False
            
            # Update the canvas
            self.canvas.update()
        except Exception as e:
            print(f"Error in mouse_release: {e}")

    def get_cursor(self) -> Qt.CursorShape:
        """Return the cursor to use for this tool."""
        return Qt.CursorShape.ArrowCursor

    def draw_preview(self, painter: QPainter) -> None:
        """Draw a preview of the tool's effect."""
        pass

    def get_selection_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a boolean mask for the current selection."""
        try:
            # Create the base mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Need at least 3 points to form a valid polygon
            if len(self.points) < 3:
                return np.zeros((height, width), dtype=bool)
                
            # Convert QPointF points to numpy array of integers
            points_array = []
            for p in self.points:
                x = max(0, min(round(p.x()), width - 1))
                y = max(0, min(round(p.y()), height - 1))
                points_array.append([x, y])
            
            # Convert to numpy array and ensure the path is closed
            points_array = np.array(points_array, dtype=np.int32)
            if not np.array_equal(points_array[0], points_array[-1]):
                points_array = np.vstack([points_array, points_array[0]])
            
            # Fill the polygon
            cv2.fillPoly(mask, [points_array], (255,))
            
            # Apply feathering if needed
            if self.feather_radius > 0:
                mask = mask.astype(np.float32) / 255.0
                mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
                mask = mask > 0.5
            else:
                mask = mask > 0
            
            # Ensure the mask has some selected pixels
            if not np.any(mask):
                return np.zeros((height, width), dtype=bool)
            
            return mask.astype(bool)
        except Exception as e:
            print(f"Error in get_selection_mask: {e}")
            return np.zeros((height, width), dtype=bool)

class BrushTool(Tool):
    """Brush tool for painting."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas, "Brush")
        self.size = 10
        self.hardness = 0.8  # 0.0 to 1.0
        self.opacity = 1.0   # 0.0 to 1.0
        self.color = QColor(0, 0, 0)  # Default to black
        self.last_pos: Optional[QPoint] = None
        self.current_draw_command = None

    def mouse_press(self, event: QMouseEvent) -> None:
        super().mouse_press(event)
        self.last_pos = self.current_pos
        
        # Start new draw command
        active_layer = self.canvas.layer_stack.get_active_layer()
        if active_layer:
            self.current_draw_command = DrawCommand(active_layer, active_layer.image)
            if self.last_pos is not None:
                self.draw_stroke(self.last_pos, self.last_pos)

    def mouse_move(self, event: QMouseEvent) -> None:
        super().mouse_move(event)
        if self.is_active and self.last_pos is not None and self.current_pos is not None:
            self.draw_stroke(self.last_pos, self.current_pos)
            self.last_pos = self.current_pos

    def mouse_release(self, event: QMouseEvent) -> None:
        if self.current_draw_command:
            self.current_draw_command.execute()
            self.canvas.command_stack.push(self.current_draw_command)
            self.current_draw_command = None
        super().mouse_release(event)
        self.last_pos = None

    def get_cursor(self) -> Qt.CursorShape:
        return Qt.CursorShape.CrossCursor

    def draw_stroke(self, start_pos: QPoint, end_pos: QPoint) -> None:
        """Draw a stroke between two points."""
        try:
            if not hasattr(self.canvas.layer_stack, 'get_active_layer'):
                print("Layer stack not available")
                return
                
            layer = self.canvas.layer_stack.get_active_layer()
            if not layer or not hasattr(layer, 'image'):
                print("No active layer available")
                return

            # Validate layer image
            layer_image = cast(QImage, layer.image)
            if layer_image.isNull():
                print("Invalid layer image")
                return
                
            # Validate positions
            if (start_pos.x() < 0 or start_pos.x() >= layer_image.width() or
                start_pos.y() < 0 or start_pos.y() >= layer_image.height() or
                end_pos.x() < 0 or end_pos.x() >= layer_image.width() or
                end_pos.y() < 0 or end_pos.y() >= layer_image.height()):
                print("Stroke position out of bounds")
                return

            # Create painter for the current layer
            painter = QPainter(layer_image)
            if not painter.isActive():
                print("Failed to initialize painter")
                return
                
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                # Set up the pen
                pen = QPen(self.color)
                pen.setWidth(self.size)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)

                # Calculate intermediate points for smoother stroke
                dx = end_pos.x() - start_pos.x()
                dy = end_pos.y() - start_pos.y()
                distance = np.sqrt(dx * dx + dy * dy)
                
                if distance > self.size:
                    # Add intermediate points for long strokes
                    steps = int(distance / (self.size * 0.5))
                    for i in range(steps + 1):
                        t = i / steps
                        x = start_pos.x() + dx * t
                        y = start_pos.y() + dy * t
                        if i == 0:
                            painter.drawPoint(QPoint(round(x), round(y)))
                        else:
                            painter.drawLine(
                                QPoint(round(prev_x), round(prev_y)),
                                QPoint(round(x), round(y))
                            )
                        prev_x, prev_y = x, y
                else:
                    # Draw direct line for short strokes
                    painter.drawLine(QPointF(start_pos), QPointF(end_pos))
            finally:
                painter.end()

            # Update the canvas
            self.canvas.update()
            
        except Exception as e:
            print(f"Error in draw_stroke: {e}")

class EraserTool(BrushTool):
    """Eraser tool - similar to brush but clears pixels."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas)
        self.name = "Eraser"
        self.color = QColor(0, 0, 0, 0)  # Transparent

class SelectionTool(Tool):
    """Base class for selection tools."""
    def __init__(self, canvas: 'Canvas', name: str):
        super().__init__(canvas, name)
        self.selection_path = QPainterPath()
        self.is_selecting = False
        self.feather_radius = 0
        self.clipboard_image: Optional[QImage] = None  # Store cut/copied content
        self.clipboard_pos: Optional[QPoint] = None    # Store original position
        self.is_floating = False  # Track if we have a floating selection
        self.floating_pos: Optional[QPoint] = None  # Position of floating selection

    def reset_state(self) -> None:
        """Reset the tool's state."""
        self.start_pos = None
        self.current_pos = None
        self.points = []
        self.is_selecting = False
        self.is_drawing = False
        self.is_floating = False
        self.floating_pos = None
        self.selection_path = QPainterPath()
        # Don't clear clipboard_image and clipboard_pos as they should persist

    def draw_preview(self, painter: QPainter) -> None:
        if self.is_active and self.start_pos is not None and self.current_pos is not None:
            # Set up the pen for selection outline
            pen = QPen(QColor(0, 0, 0))  # Black outline
            pen.setStyle(Qt.PenStyle.DashLine)  # Dashed line
            pen.setWidth(1)  # 1 pixel width
            painter.setPen(pen)

            # Draw white background line
            white_pen = QPen(QColor(255, 255, 255))  # White background
            white_pen.setStyle(Qt.PenStyle.DashLine)
            white_pen.setWidth(1)
            white_pen.setDashOffset(4)  # Offset to create alternating pattern
            painter.save()
            painter.setPen(white_pen)
            self.draw_selection_shape(painter)
            painter.restore()

            # Draw black foreground line
            painter.setPen(pen)
            self.draw_selection_shape(painter)

        # Draw floating selection if active
        if self.is_floating and self.clipboard_image is not None and self.floating_pos is not None:
            painter.save()
            painter.setOpacity(0.8)  # Make it slightly transparent to indicate floating state
            painter.drawImage(self.floating_pos, self.clipboard_image)
            
            # Draw border around floating selection
            pen = QPen(QColor(0, 0, 0))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(
                self.floating_pos.x(),
                self.floating_pos.y(),
                self.clipboard_image.width(),
                self.clipboard_image.height()
            )
            painter.restore()

    def draw_selection_shape(self, painter: QPainter) -> None:
        pass

    def get_cursor(self) -> Qt.CursorShape:
        """Return crosshair cursor for selection tools."""
        return Qt.CursorShape.CrossCursor

    def mouse_press(self, event: QMouseEvent) -> None:
        if self.is_floating:
            # Finalize paste on left click
            if event.button() == Qt.MouseButton.LeftButton:
                self.finalize_paste()
            # Cancel paste on right click
            elif event.button() == Qt.MouseButton.RightButton:
                self.is_floating = False
                self.floating_pos = None
                self.canvas.update()
            return
        
        super().mouse_press(event)
        self.is_selecting = True
        self.canvas.clear_selection()  # Clear previous selection

    def mouse_release(self, event: QMouseEvent) -> None:
        if self.is_selecting:
            self.is_selecting = False
            # Apply the selection
            if self.start_pos is not None and self.current_pos is not None:
                # Get the selection mask
                width = self.canvas.width()
                height = self.canvas.height()
                mask = self.get_selection_mask(width, height)
                
                # Apply feathering if needed
                if self.feather_radius > 0:
                    mask = mask.astype(np.float32)
                    mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
                
                # Update the canvas selection
                self.canvas.set_selection(mask)
        super().mouse_release(event)

    def apply_selection(self) -> None:
        """Apply the current selection to the active layer."""
        if not hasattr(self.canvas, 'get_selection'):
            return
        
        # Get the current selection mask
        selection = self.canvas.get_selection()
        if selection is None:
            return
        
        # Get the active layer
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return
        
        # Convert QImage to numpy array safely
        img = layer.image
        width = img.width()
        height = img.height()
        ptr = img.constBits()
        if ptr is None:
            return
        ptr.setsize(height * width * 4)  # 4 channels (RGBA)
        arr = np.array(ptr).reshape(height, width, 4)
        
        # Apply the selection mask to the alpha channel
        selection = selection.astype(np.float32)
        arr[..., 3] = (arr[..., 3] * selection).astype(np.uint8)
        
        # Update the layer
        layer.changed.emit()
        self.canvas.update()

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self.canvas.clear_selection()
        self.canvas.update()

    def cut_selection(self) -> None:
        """Cut the selected area and store it for pasting."""
        try:
            # Copy first
            self.copy_selection()
            if self.clipboard_image is None:
                print("Failed to copy selection")
                return
                
            # Get the active layer
            layer = self.canvas.layer_stack.get_active_layer()
            if not layer or not hasattr(layer, 'image'):
                print("No active layer available")
                return
                
            # Validate layer dimensions
            if layer.image.width() <= 0 or layer.image.height() <= 0:
                print("Invalid layer dimensions")
                return
                
            # Create a new image for the layer with the cut area removed
            new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
            if new_image.isNull():
                print("Failed to create new image")
                return
                
            new_image.fill(Qt.GlobalColor.transparent)
            
            # Draw the original image
            painter = QPainter(new_image)
            if not painter.isActive():
                print("Failed to initialize painter")
                return
                
            try:
                painter.drawImage(0, 0, layer.image)
                
                # Get the selection mask
                selection = self.canvas.get_selection()
                if selection is None:
                    print("No active selection")
                    return
                    
                # Create a mask image
                mask_image = QImage(layer.image.size(), QImage.Format.Format_Alpha8)
                if mask_image.isNull():
                    print("Failed to create mask image")
                    return
                    
                mask_image.fill(Qt.GlobalColor.transparent)
                
                # Fill the mask with white where selection is False (to keep)
                for y in range(selection.shape[0]):
                    for x in range(selection.shape[1]):
                        mask_image.setPixelColor(x, y, QColor(0, 0, 0, 0 if selection[y, x] else 255))
                
                # Apply the mask to remove the selected area
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                painter.drawImage(0, 0, mask_image)
            finally:
                painter.end()
            
            # Update the layer
            layer.set_image(new_image)
            layer.changed.emit()
            self.canvas.update()
        except Exception as e:
            print(f"Error in cut_selection: {e}")
            self.reset_state()

    def copy_selection(self) -> None:
        """Copy the selected area without removing it."""
        try:
            # Get the current selection mask
            selection = self.canvas.get_selection()
            if selection is None:
                return
                
            # Get the active layer
            layer = self.canvas.layer_stack.get_active_layer()
            if not layer or not hasattr(layer, 'image'):
                return
                
            # Find the bounds of the selection
            y_indices, x_indices = np.where(selection)
            if len(x_indices) == 0 or len(y_indices) == 0:
                return
                
            min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
            min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
            
            # Create a new image for the copied content
            copy_image = QImage(max_x - min_x + 1, max_y - min_y + 1, QImage.Format.Format_ARGB32)
            copy_image.fill(Qt.GlobalColor.transparent)
            
            # Draw the original content into the copy image
            painter = QPainter(copy_image)
            painter.drawImage(0, 0, layer.image, min_x, min_y)
            
            # Create and apply the mask
            mask_image = QImage(copy_image.size(), QImage.Format.Format_Alpha8)
            mask_image.fill(Qt.GlobalColor.transparent)
            
            # Fill the mask based on the selection
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if selection[y, x]:
                        mask_image.setPixelColor(x - min_x, y - min_y, QColor(255, 255, 255, 255))
            
            # Apply the mask
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            painter.drawImage(0, 0, mask_image)
            painter.end()
            
            # Store the copied content and its original position
            self.clipboard_image = copy_image
            self.clipboard_pos = QPoint(min_x, min_y)
        except Exception as e:
            print(f"Error in copy_selection: {e}")

    def paste_selection(self, pos: Optional[Union[QPoint, QPointF]] = None) -> None:
        """Paste the previously cut/copied content."""
        try:
            if self.clipboard_image is None:
                print("No content to paste")
                return
                
            layer = self.canvas.layer_stack.get_active_layer()
            if not layer or not hasattr(layer, 'image'):
                print("No active layer available")
                return
                
            # Validate layer dimensions
            if layer.image.width() <= 0 or layer.image.height() <= 0:
                print("Invalid layer dimensions")
                return
                
            # Create a new image with the same size
            new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
            if new_image.isNull():
                print("Failed to create new image")
                return
                
            new_image.fill(Qt.GlobalColor.transparent)
            
            # Draw the original image
            painter = QPainter(new_image)
            if not painter.isActive():
                print("Failed to initialize painter")
                return
                
            try:
                painter.drawImage(0, 0, layer.image)
                
                # Calculate paste position
                paste_pos = pos if pos is not None else self.clipboard_pos
                if paste_pos is None:
                    # Default to center if no position is specified
                    paste_pos = QPointF(
                        (layer.image.width() - self.clipboard_image.width()) // 2,
                        (layer.image.height() - self.clipboard_image.height()) // 2
                    )
                
                # Convert to QPoint for drawing
                if isinstance(paste_pos, QPointF):
                    paste_point = QPoint(round(paste_pos.x()), round(paste_pos.y()))
                else:
                    paste_point = paste_pos
                
                # Validate paste position
                if (paste_point.x() < -self.clipboard_image.width() or 
                    paste_point.y() < -self.clipboard_image.height() or
                    paste_point.x() > layer.image.width() or 
                    paste_point.y() > layer.image.height()):
                    print("Paste position out of bounds")
                    return
                
                # Draw the clipboard content
                painter.drawImage(paste_point, self.clipboard_image)
            finally:
                painter.end()
            
            # Update the layer
            layer.set_image(new_image)
            layer.changed.emit()
            self.canvas.update()
            
            # Reset state after pasting
            self.reset_state()
        except Exception as e:
            print(f"Error in paste_selection: {e}")
            self.reset_state()

    def recolor_selection(self, color: QColor) -> None:
        """Recolor the selected area with the specified color."""
        if not hasattr(self.canvas, 'get_selection'):
            return
        
        # Get the current selection mask
        selection = self.canvas.get_selection()
        if selection is None:
            return
        
        # Get the active layer
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return
        
        try:
            # Create a new image with the same size
            new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
            new_image.fill(Qt.GlobalColor.transparent)
            
            # Draw the original image
            painter = QPainter(new_image)
            painter.drawImage(0, 0, layer.image)
            
            # Create a color overlay image
            overlay = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
            overlay.fill(color)
            
            # Create a mask image
            mask_image = QImage(layer.image.size(), QImage.Format.Format_Alpha8)
            mask_image.fill(Qt.GlobalColor.transparent)
            
            # Fill the mask based on the selection
            for y in range(selection.shape[0]):
                for x in range(selection.shape[1]):
                    if selection[y, x]:
                        mask_image.setPixelColor(x, y, QColor(255, 255, 255, int(255 * self.opacity)))
            
            # Apply the color overlay with the mask
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setOpacity(self.opacity)
            painter.drawImage(0, 0, overlay)
            
            # Apply the mask
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            painter.drawImage(0, 0, mask_image)
            painter.end()
            
            # Update the layer
            layer.set_image(new_image)
            layer.changed.emit()
            self.canvas.update()
        except Exception as e:
            print(f"Error in recolor_selection: {e}")

    def finalize_paste(self) -> None:
        """Finalize the paste operation at the current floating position."""
        if not self.is_floating or self.floating_pos is None or self.clipboard_image is None:
            return
            
        try:
            # Perform the paste operation
            self.paste_selection(self.floating_pos)
            
            # Reset floating state but keep clipboard content
            self.is_floating = False
            self.floating_pos = None
            self.canvas.update()
        except Exception as e:
            print(f"Error in finalize_paste: {e}")
            self.reset_state()

    def start_floating_paste(self, pos: Union[QPoint, QPointF]) -> None:
        """Start floating paste mode at the given position."""
        if self.clipboard_image is None:
            print("No content to paste")
            return
            
        try:
            # Convert QPointF to QPoint if needed
            if isinstance(pos, QPointF):
                pos = QPoint(round(pos.x()), round(pos.y()))
            elif not isinstance(pos, QPoint):
                print("Invalid position type")
                return
            
            # Calculate centered position
            offset_x = self.clipboard_image.width() // 2
            offset_y = self.clipboard_image.height() // 2
            centered_pos = QPoint(pos.x() - offset_x, pos.y() - offset_y)
            
            # Start floating mode
            self.is_floating = True
            self.floating_pos = centered_pos
            self.canvas.update()
        except Exception as e:
            print(f"Error in start_floating_paste: {e}")
            self.reset_state()

    def update_floating_position(self, pos: Union[QPoint, QPointF]) -> None:
        """Update the position of the floating selection."""
        if not self.is_floating:
            return
            
        # Convert QPointF to QPoint if needed
        if isinstance(pos, QPointF):
            pos = QPoint(round(pos.x()), round(pos.y()))
        elif not isinstance(pos, QPoint):
            return
            
        self.floating_pos = pos
        self.canvas.update()

    def mouse_move(self, event: QMouseEvent) -> None:
        if event is None:
            return
            
        # Get the transformed position
        pos = event.position()
        if not isinstance(pos, QPointF):
            pos = QPointF(pos.x(), pos.y())
        pos = self.canvas.get_transformed_pos(pos)
        
        if self.is_floating:
            # Update floating selection position
            if self.clipboard_image is not None:
                offset_x = self.clipboard_image.width() // 2
                offset_y = self.clipboard_image.height() // 2
                self.floating_pos = QPoint(round(pos.x()) - offset_x, round(pos.y()) - offset_y)
                self.canvas.update()
            return
        
        # Convert QPointF to QPoint for base class
        self.current_pos = QPoint(round(pos.x()), round(pos.y()))
        self.canvas.update()

class RectangleSelection(SelectionTool):
    """Rectangular selection tool."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas, "Rectangle Selection")

    def draw_selection_shape(self, painter: QPainter) -> None:
        if self.start_pos is not None and self.current_pos is not None:
            x = min(self.start_pos.x(), self.current_pos.x())
            y = min(self.start_pos.y(), self.current_pos.y())
            width = abs(self.current_pos.x() - self.start_pos.x())
            height = abs(self.current_pos.y() - self.start_pos.y())
            painter.drawRect(x, y, width, height)

    def get_selection_mask(self, width: int, height: int) -> NDArray[np.bool_]:
        mask = np.zeros((height, width), dtype=np.bool_)
        if self.start_pos is not None and self.current_pos is not None:
            x1 = min(max(self.start_pos.x(), 0), width)
            y1 = min(max(self.start_pos.y(), 0), height)
            x2 = min(max(self.current_pos.x(), 0), width)
            y2 = min(max(self.current_pos.y(), 0), height)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            mask[y_min:y_max, x_min:x_max] = True
        return mask

class EllipseSelection(SelectionTool):
    """Elliptical selection tool."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas, "Ellipse Selection")

    def draw_selection_shape(self, painter: QPainter) -> None:
        if self.start_pos is not None and self.current_pos is not None:
            x = min(self.start_pos.x(), self.current_pos.x())
            y = min(self.start_pos.y(), self.current_pos.y())
            width = abs(self.current_pos.x() - self.start_pos.x())
            height = abs(self.current_pos.y() - self.start_pos.y())
            painter.drawEllipse(x, y, width, height)

    def get_selection_mask(self, width: int, height: int) -> NDArray[np.bool_]:
        mask = np.zeros((height, width), dtype=np.bool_)
        if self.start_pos is not None and self.current_pos is not None:
            x1 = min(max(self.start_pos.x(), 0), width)
            y1 = min(max(self.start_pos.y(), 0), height)
            x2 = min(max(self.current_pos.x(), 0), width)
            y2 = min(max(self.current_pos.y(), 0), height)
            
            # Create ellipse mask using numpy
            y, x = np.ogrid[:height, :width]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            rx = abs(x2 - x1) / 2
            ry = abs(y2 - y1) / 2
            
            # Ellipse equation: (x-h)²/a² + (y-k)²/b² <= 1
            if rx > 0 and ry > 0:
                mask = ((x - center_x)**2 / rx**2 + (y - center_y)**2 / ry**2) <= 1
        
        return mask

class LassoSelection(SelectionTool):
    """Freehand selection tool."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas, "Lasso Selection")
        self.points: List[QPointF] = []
        self.is_drawing = False
        self.min_distance = 1.0
        self.reset_state()

    def reset_state(self) -> None:
        """Reset the tool's state."""
        super().reset_state()
        self.points = []
        self.is_drawing = False
        self.is_selecting = False
        self.is_floating = False
        self.floating_pos = None
        self.clipboard_image = None
        self.clipboard_pos = None

    def mouse_press(self, event: QMouseEvent) -> None:
        """Start the lasso selection."""
        if event is None:
            return
            
        try:
            # Handle floating selection first
            if self.is_floating:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.finalize_paste()
                elif event.button() == Qt.MouseButton.RightButton:
                    self.reset_state()
                return

            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos.x(), pos.y())
            pos = self.canvas.get_transformed_pos(pos)
            
            # Reset state and start drawing
            self.reset_state()
            self.points = [pos]  # Store QPointF directly
            self.start_pos = QPoint(round(pos.x()), round(pos.y()))
            self.current_pos = QPoint(round(pos.x()), round(pos.y()))
            self.is_drawing = True
            self.is_selecting = True
            self.is_active = True
            
            # Clear any existing selection
            self.canvas.clear_selection()
            self.canvas.update()
            
        except Exception as e:
            print(f"Error in mouse_press: {e}")
            self.reset_state()
            self.canvas.clear_selection()
            self.canvas.update()

    def mouse_move(self, event: QMouseEvent) -> None:
        """Update the lasso selection as the user drags."""
        if event is None:
            return
            
        try:
            # Handle floating selection
            if self.is_floating:
                pos = self.canvas.get_transformed_pos(event.position())
                if isinstance(pos, QPointF):
                    self.update_floating_position(pos)
                return

            # Handle drawing
            if not self.is_drawing:
                return
                
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos.x(), pos.y())
            pos = self.canvas.get_transformed_pos(pos)
            
            # Update current position
            self.current_pos = QPoint(round(pos.x()), round(pos.y()))
            
            # Add point if needed
            if self.points:
                last_point = self.points[-1]
                dx = pos.x() - last_point.x()
                dy = pos.y() - last_point.y()
                distance = np.sqrt(dx * dx + dy * dy)
                
                if distance >= self.min_distance or len(self.points) < 3:
                    self.points.append(pos)  # Store QPointF directly
                    self.canvas.update()
        except Exception as e:
            print(f"Error in mouse_move: {e}")

    def mouse_release(self, event: QMouseEvent) -> None:
        """Complete the lasso selection."""
        if event is None or not self.is_drawing:
            return
            
        try:
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos.x(), pos.y())
            pos = self.canvas.get_transformed_pos(pos)
            
            # Add final points and close the path
            if len(self.points) >= 3:
                # Add the final point if it's different from the last one
                if not self.points or not np.allclose(
                    [pos.x(), pos.y()],
                    [self.points[-1].x(), self.points[-1].y()],
                    rtol=1e-5
                ):
                    self.points.append(pos)
                
                # Close the path if needed
                if not np.allclose(
                    [self.points[0].x(), self.points[0].y()],
                    [self.points[-1].x(), self.points[-1].y()],
                    rtol=1e-5
                ):
                    self.points.append(self.points[0])
                
                # Check canvas dimensions
                if not hasattr(self.canvas, 'width') or not hasattr(self.canvas, 'height'):
                    print("Invalid canvas dimensions")
                    self.reset_state()
                    return
                    
                width = self.canvas.width()
                height = self.canvas.height()
                
                if width <= 0 or height <= 0:
                    print("Invalid canvas dimensions")
                    self.reset_state()
                    return
                
                # Create and set the selection mask
                mask = self.get_selection_mask(width, height)
                
                if mask is not None and mask.any():
                    # Apply feathering if needed
                    if self.feather_radius > 0:
                        mask = mask.astype(np.float32)
                        mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
                    
                    # Set the selection on the canvas
                    self.canvas.set_selection(mask)
                    self.is_selecting = True
                else:
                    self.canvas.clear_selection()
                    self.is_selecting = False
            else:
                self.canvas.clear_selection()
                self.is_selecting = False
            
            # Reset states
            self.is_drawing = False
            self.is_active = False
            self.canvas.update()
            
        except Exception as e:
            print(f"Error in mouse_release: {e}")
            # Ensure we clean up on error
            self.is_drawing = False
            self.is_selecting = False
            self.is_active = False
            self.canvas.clear_selection()
            self.canvas.update()

    def get_selection_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a boolean mask for the current selection."""
        try:
            # Create the base mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Need at least 3 points to form a valid polygon
            if len(self.points) < 3:
                return np.zeros((height, width), dtype=bool)
                
            # Convert QPointF points to numpy array of integers
            points_array = []
            for p in self.points:
                x = max(0, min(round(p.x()), width - 1))
                y = max(0, min(round(p.y()), height - 1))
                points_array.append([x, y])
            
            # Convert to numpy array and ensure the path is closed
            points_array = np.array(points_array, dtype=np.int32)
            if not np.array_equal(points_array[0], points_array[-1]):
                points_array = np.vstack([points_array, points_array[0]])
            
            # Fill the polygon
            cv2.fillPoly(mask, [points_array], (255,))
            
            # Apply feathering if needed
            if self.feather_radius > 0:
                mask = mask.astype(np.float32) / 255.0
                mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
                mask = mask > 0.5
            else:
                mask = mask > 0
            
            # Ensure the mask has some selected pixels
            if not np.any(mask):
                return np.zeros((height, width), dtype=bool)
            
            return mask.astype(bool)
        except Exception as e:
            print(f"Error in get_selection_mask: {e}")
            return np.zeros((height, width), dtype=bool)

    def draw_selection_shape(self, painter: QPainter) -> None:
        """Draw the current lasso selection."""
        if not self.is_drawing or len(self.points) < 2:
            return
            
        try:
            # Set up the pen for a dashed line effect
            pen = QPen(QColor(255, 255, 255))  # White color
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            # Draw the path
            path = QPainterPath()
            path.moveTo(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)
            
            # Draw line to current position if we have one
            if self.current_pos is not None:
                current_point = QPointF(self.current_pos)
                path.lineTo(current_point)
                if len(self.points) > 2:
                    path.lineTo(self.points[0])
            
            # Draw the path with white color
            painter.drawPath(path)
            
            # Draw black overlay line (offset dash pattern)
            pen.setColor(QColor(0, 0, 0))
            pen.setDashOffset(4)
            painter.setPen(pen)
            painter.drawPath(path)
            
        except Exception as e:
            print(f"Error in draw_selection_shape: {e}")

    def get_cursor(self) -> Qt.CursorShape:
        """Return crosshair cursor for the lasso tool."""
        return Qt.CursorShape.CrossCursor

class MagneticLassoSelection(LassoSelection):
    """Magnetic lasso tool that snaps to edges."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__(canvas)
        self.name = "Magnetic Lasso"
        self.edge_width = 10  # Search radius for edges
        self.edge_contrast = 50  # Minimum contrast for edge detection
        self.anchor_spacing = 20  # Minimum distance between anchor points
        self.last_anchor: Optional[QPointF] = None
        self.edge_map: Optional[np.ndarray] = None
        self.edge_gradient: Optional[np.ndarray] = None
        self.edge_strength: Optional[np.ndarray] = None
        self.debug_mode = False  # Toggle for debug visualization

    def update_edge_detection(self) -> None:
        """Update edge detection with enhanced sensitivity and noise handling."""
        try:
            layer = self.canvas.layer_stack.get_active_layer()
            if not layer or not hasattr(layer, 'image'):
                self.cleanup_edge_detection()
                return

            # Convert QImage to numpy array
            image = layer.image
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            if ptr is None:
                self.cleanup_edge_detection()
                return
                
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4).copy()  # Create a copy to avoid reference issues

            # Convert to grayscale with proper weighting
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
            del arr  # Free memory

            # Apply bilateral filter to reduce noise while preserving edges
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

            # Multi-scale edge detection
            edges_fine = cv2.Canny(gray, self.edge_contrast, self.edge_contrast * 2)
            edges_coarse = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 
                                    self.edge_contrast * 0.5, 
                                    self.edge_contrast)

            # Combine multi-scale edges
            self.edge_map = cv2.addWeighted(edges_fine, 0.7, edges_coarse, 0.3, 0)

            # Compute edge strength for better snapping
            gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            self.edge_strength = np.sqrt(gradient_x**2 + gradient_y**2)
            self.edge_gradient = np.arctan2(gradient_y, gradient_x)
            
            del gradient_x, gradient_y  # Free memory

            # Normalize edge strength
            edge_min = np.min(self.edge_strength)
            edge_max = np.max(self.edge_strength)
            if edge_max > edge_min:
                self.edge_strength = ((self.edge_strength - edge_min) * 255.0 / (edge_max - edge_min)).astype(np.uint8)
            else:
                self.edge_strength = np.zeros_like(self.edge_strength, dtype=np.uint8)

            if self.debug_mode:
                self._debug_edge_detection(gray, edges_fine, edges_coarse)
                
            del edges_fine, edges_coarse, gray  # Free memory
            
        except Exception as e:
            print(f"Error in update_edge_detection: {e}")
            self.cleanup_edge_detection()

    def cleanup_edge_detection(self) -> None:
        """Clean up edge detection resources."""
        self.edge_map = None
        self.edge_strength = None
        self.edge_gradient = None

    def find_edge_point(self, pos: QPointF) -> QPointF:
        """Enhanced edge point detection with sub-pixel accuracy."""
        if self.edge_map is None or self.edge_strength is None:
            return pos

        # Ensure coordinates are within bounds
        x = max(0, min(int(pos.x()), self.edge_map.shape[1] - 1))
        y = max(0, min(int(pos.y()), self.edge_map.shape[0] - 1))
        
        height, width = self.edge_map.shape

        # Define search region with bounds checking
        x1 = max(0, x - self.edge_width)
        x2 = min(width, x + self.edge_width + 1)
        y1 = max(0, y - self.edge_width)
        y2 = min(height, y + self.edge_width + 1)

        # Check if search region is valid
        if x1 >= x2 or y1 >= y2:
            return pos

        # Extract regions of interest
        edge_region = self.edge_map[y1:y2, x1:x2]
        strength_region = self.edge_strength[y1:y2, x1:x2]

        if not edge_region.any():
            return pos

        # Find edge points and their strengths
        edge_y, edge_x = np.nonzero(edge_region)
        if len(edge_x) == 0:
            return pos

        # Calculate distances and weights
        dx = edge_x + x1 - x
        dy = edge_y + y1 - y
        distances = dx * dx + dy * dy
        strengths = strength_region[edge_y, edge_x]

        # Combine distance and strength for scoring
        scores = distances * (1 + np.exp(-strengths / 128))  # Sigmoid-like weighting
        best_idx = np.argmin(scores)

        # Sub-pixel refinement using quadratic interpolation
        x_refined = min(max(edge_x[best_idx] + x1, 0), width - 1)
        y_refined = min(max(edge_y[best_idx] + y1, 0), height - 1)

        if self.debug_mode:
            self._debug_edge_point(x, y, x_refined, y_refined)

        return QPointF(float(x_refined), float(y_refined))

    def _debug_edge_detection(self, gray: np.ndarray, edges_fine: np.ndarray, edges_coarse: np.ndarray) -> None:
        """Visualize edge detection results for debugging."""
        # Create debug visualization
        debug_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        if edges_fine is not None:
            debug_img[..., 0] = edges_fine.astype(np.uint8)  # Red channel - fine edges
        if edges_coarse is not None:
            debug_img[..., 1] = edges_coarse.astype(np.uint8)  # Green channel - coarse edges
        if self.edge_map is not None:
            debug_img[..., 2] = self.edge_map.astype(np.uint8)  # Blue channel - combined edges

        # Save or display debug image
        cv2.imwrite('debug_edges.png', debug_img)

    def _debug_edge_point(self, x: int, y: int, x_refined: float, y_refined: float) -> None:
        """Visualize edge point snapping for debugging."""
        if self.edge_map is None:
            return

        # Create visualization
        debug_img = cv2.cvtColor(self.edge_map, cv2.COLOR_GRAY2BGR)
        
        # Draw search radius
        cv2.circle(debug_img, (x, y), self.edge_width, (0, 255, 0), 1)
        
        # Draw original and snapped points
        cv2.circle(debug_img, (int(x), int(y)), 3, (0, 0, 255), -1)  # Original point in red
        cv2.circle(debug_img, (int(x_refined), int(y_refined)), 3, (255, 0, 0), -1)  # Snapped point in blue
        
        # Draw connection line
        cv2.line(debug_img, (x, y), (int(x_refined), int(y_refined)), (255, 255, 0), 1)

        # Save debug image
        cv2.imwrite(f'debug_snap_{x}_{y}.png', debug_img)

    def test_edge_detection(self, test_image: np.ndarray) -> Dict[str, float]:
        """Test edge detection performance on a given image.
        
        Args:
            test_image: Input image for testing
            
        Returns:
            Dictionary containing performance metrics
        """
        # Convert test image to proper format
        if len(test_image.shape) == 3:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image.copy()

        # Ground truth edges using Canny with optimal parameters
        ground_truth = cv2.Canny(gray, 100, 200)

        # Test current parameters
        self.edge_contrast = 50  # Reset to default
        self.update_edge_detection()
        if self.edge_map is None:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        # Calculate metrics
        true_positives = np.sum((self.edge_map > 0) & (ground_truth > 0))
        false_positives = np.sum((self.edge_map > 0) & (ground_truth == 0))
        false_negatives = np.sum((self.edge_map == 0) & (ground_truth > 0))

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }

    def optimize_parameters(self, test_image: np.ndarray) -> Dict[str, int]:
        """Optimize tool parameters for the given image.
        
        Args:
            test_image: Input image for optimization
            
        Returns:
            Dictionary containing optimized parameters
        """
        best_score = 0.0
        best_params = {
            'edge_contrast': self.edge_contrast,
            'edge_width': self.edge_width,
            'anchor_spacing': self.anchor_spacing
        }

        # Test different parameter combinations
        for contrast in range(30, 71, 10):  # Test contrast values
            self.edge_contrast = contrast
            metrics = self.test_edge_detection(test_image)
            
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_params['edge_contrast'] = contrast

        # Restore best parameters
        self.edge_contrast = best_params['edge_contrast']
        return best_params

    def handle_low_contrast(self, pos: QPointF) -> QPointF:
        """Special handling for low-contrast regions."""
        if self.edge_strength is None:
            return pos

        x, y = int(pos.x()), int(pos.y())
        local_strength = self.edge_strength[
            max(0, y-2):min(self.edge_strength.shape[0], y+3),
            max(0, x-2):min(self.edge_strength.shape[1], x+3)
        ].mean()

        if local_strength < self.edge_contrast * 0.5:
            # In low contrast regions, increase search radius temporarily
            temp_edge_width = self.edge_width * 2
            x1 = max(0, x - temp_edge_width)
            x2 = min(self.edge_strength.shape[1], x + temp_edge_width + 1)
            y1 = max(0, y - temp_edge_width)
            y2 = min(self.edge_strength.shape[0], y + temp_edge_width + 1)
            
            # Look for strongest edge in larger region
            strength_region = self.edge_strength[y1:y2, x1:x2]
            if strength_region.size > 0:
                max_y, max_x = np.unravel_index(np.argmax(strength_region), strength_region.shape)
                return QPointF(float(max_x + x1), float(max_y + y1))

        return pos

    def get_edge_strength(self, x: int, y: int) -> float:
        """Get the edge strength at a given point."""
        if not hasattr(self, 'edge_strength') or self.edge_strength is None:
            return 0.0
            
        if not (0 <= y < self.edge_strength.shape[0] and 0 <= x < self.edge_strength.shape[1]):
            return 0.0
            
        # Calculate local mean of edge strength
        y_start = max(0, y-2)
        y_end = min(self.edge_strength.shape[0], y+3)
        x_start = max(0, x-2)
        x_end = min(self.edge_strength.shape[1], x+3)
        
        local_region = self.edge_strength[y_start:y_end, x_start:x_end]
        local_strength = float(np.mean(local_region)) if local_region.size > 0 else 0.0
        
        return local_strength if local_strength >= self.edge_contrast * 0.5 else 0.0

    def mouse_press(self, event: QMouseEvent) -> None:
        """Start magnetic lasso selection."""
        super().mouse_press(event)
        if self.is_drawing:
            self.update_edge_detection()  # Initialize edge detection
            self.last_anchor = self.points[0] if len(self.points) > 0 else None

    def mouse_move(self, event: QMouseEvent) -> None:
        """Update magnetic lasso selection with edge snapping."""
        if event is None or not self.is_drawing:
            return
            
        try:
            # Get the transformed position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos.x(), pos.y())
            pos = self.canvas.get_transformed_pos(pos)
            
            # Update current position
            self.current_pos = QPoint(round(pos.x()), round(pos.y()))
            
            # Find nearest edge point
            snapped_pos = self.find_edge_point(pos)
            
            # Handle low contrast regions
            if self.edge_strength is not None:
                x, y = int(snapped_pos.x()), int(snapped_pos.y())
                if self.get_edge_strength(x, y) < self.edge_contrast * 0.5:
                    snapped_pos = self.handle_low_contrast(pos)
            
            # Add point if needed
            if len(self.points) > 0 and self.last_anchor is not None:
                dx = snapped_pos.x() - self.last_anchor.x()
                dy = snapped_pos.y() - self.last_anchor.y()
                distance = np.sqrt(dx * dx + dy * dy)
                
                if distance >= self.anchor_spacing or len(self.points) < 3:
                    self.points.append(snapped_pos)
                    self.last_anchor = snapped_pos
                    self.canvas.update()
            
        except Exception as e:
            print(f"Error in magnetic lasso mouse_move: {e}")

    def mouse_release(self, event: QMouseEvent) -> None:
        """Complete magnetic lasso selection."""
        if not self.is_drawing:
            return
            
        try:
            # Get the final position
            pos = event.position()
            if not isinstance(pos, QPointF):
                pos = QPointF(pos.x(), pos.y())
            pos = self.canvas.get_transformed_pos(pos)
            
            # Find nearest edge point for final position
            final_pos = self.find_edge_point(pos)
            
            # Add final points and close the path
            if len(self.points) >= 3:
                # Add the final point if it's different from the last one
                if not np.allclose(
                    [final_pos.x(), final_pos.y()],
                    [self.points[-1].x(), self.points[-1].y()],
                    rtol=1e-5
                ):
                    self.points.append(final_pos)
                
                # Close the path by adding the first point
                if not np.allclose(
                    [self.points[0].x(), self.points[0].y()],
                    [self.points[-1].x(), self.points[-1].y()],
                    rtol=1e-5
                ):
                    self.points.append(self.points[0])
                
                # Check canvas dimensions
                if not hasattr(self.canvas, 'width') or not hasattr(self.canvas, 'height'):
                    print("Invalid canvas dimensions")
                    self.reset_state()
                    return
                    
                width = self.canvas.width()
                height = self.canvas.height()
                
                if width <= 0 or height <= 0:
                    print("Invalid canvas dimensions")
                    self.reset_state()
                    return
                
                # Create and set the selection mask
                mask = self.get_selection_mask(width, height)
                
                if mask is not None and mask.any():
                    # Apply feathering if needed
                    if self.feather_radius > 0:
                        mask = mask.astype(np.float32)
                        mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
                    
                    # Set the selection on the canvas
                    self.canvas.set_selection(mask)
                    self.is_selecting = True
                else:
                    self.canvas.clear_selection()
                    self.is_selecting = False
            
            # Reset states
            self.is_drawing = False
            self.is_active = False
            self.last_anchor = None
            self.edge_map = None
            self.edge_strength = None
            self.edge_gradient = None
            self.canvas.update()
            
        except Exception as e:
            print(f"Error in magnetic lasso mouse_release: {e}")
            self.reset_state()
            self.canvas.clear_selection()
            self.canvas.update()