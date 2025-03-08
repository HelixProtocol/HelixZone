from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING, cast
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

    def mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        pos = self.canvas.get_transformed_pos(event.pos())
        self.start_pos = pos
        self.current_pos = pos
        self.is_active = True

    def mouse_release(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        self.is_active = False

    def mouse_move(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        pos = self.canvas.get_transformed_pos(event.pos())
        self.current_pos = pos

    def get_cursor(self) -> Qt.CursorShape:
        """Return the cursor to use for this tool."""
        return Qt.CursorShape.ArrowCursor

    def draw_preview(self, painter: QPainter) -> None:
        """Draw a preview of the tool's effect."""
        pass

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
        if not hasattr(self.canvas.layer_stack, 'get_active_layer'):
            return
            
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return

        # Create painter for the current layer
        layer_image = cast(QImage, layer.image)
        painter = QPainter(layer_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set up the pen
        pen = QPen(self.color)
        pen.setWidth(self.size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        # Draw the line
        painter.drawLine(QPointF(start_pos), QPointF(end_pos))
        painter.end()

        # Update the canvas
        self.canvas.update()

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

    def get_selection_mask(self, width: int, height: int) -> NDArray[np.bool_]:
        """Returns a boolean mask of the selected area"""
        mask = np.zeros((height, width), dtype=np.bool_)
        # Implement in subclasses
        return mask

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
        if not hasattr(self.canvas, 'get_selection'):
            return
        
        # Get the current selection mask
        selection = self.canvas.get_selection()
        if selection is None or not isinstance(selection, np.ndarray):
            return
        
        # Get the active layer
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return
        
        # Convert selection to boolean and find bounds
        selection = selection.astype(bool)
        if not selection.any():
            return
            
        y_indices, x_indices = np.where(selection)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return
            
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        
        # Create a new image for the cut content
        cut_image = QImage(max_x - min_x + 1, max_y - min_y + 1, QImage.Format.Format_ARGB32)
        cut_image.fill(Qt.GlobalColor.transparent)
        
        # Draw the original content into the cut image
        painter = QPainter(cut_image)
        painter.drawImage(0, 0, layer.image, min_x, min_y)
        painter.end()
        
        # Store the cut content and its original position
        self.clipboard_image = cut_image
        self.clipboard_pos = QPoint(min_x, min_y)
        
        # Create a new image for the layer with the cut area removed
        new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        new_image.fill(Qt.GlobalColor.transparent)
        
        # Draw the original image
        painter = QPainter(new_image)
        painter.drawImage(0, 0, layer.image)
        
        # Create and apply the inverse mask
        inverse_selection = (~selection).astype(np.uint8) * 255
        selection_bytes = inverse_selection.tobytes()
        selection_img = QImage(selection_bytes, selection.shape[1], selection.shape[0], selection.shape[1], QImage.Format.Format_Grayscale8)
        
        # Apply the inverse mask
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        painter.drawImage(0, 0, selection_img)
        painter.end()
        
        # Update the layer
        layer.set_image(new_image)
        layer.changed.emit()
        self.canvas.update()

    def paste_selection(self, pos: Optional[QPoint] = None) -> None:
        """Paste the previously cut/copied content."""
        if self.clipboard_image is None:
            return
            
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return
            
        # Create a new image with the same size
        new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        new_image.fill(Qt.GlobalColor.transparent)
        
        # Draw the original image
        painter = QPainter(new_image)
        painter.drawImage(0, 0, layer.image)
        
        # Calculate paste position
        paste_pos = pos if pos is not None else self.clipboard_pos
        if paste_pos is None:
            # Default to center if no position is specified
            paste_pos = QPoint(
                (layer.image.width() - self.clipboard_image.width()) // 2,
                (layer.image.height() - self.clipboard_image.height()) // 2
            )
        
        # Draw the clipboard content
        painter.drawImage(paste_pos, self.clipboard_image)
        painter.end()
        
        # Update the layer
        layer.set_image(new_image)
        layer.changed.emit()
        self.canvas.update()

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
        
        # Create a new image with the same size
        new_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        new_image.fill(Qt.GlobalColor.transparent)
        
        # Draw the original image
        painter = QPainter(new_image)
        painter.drawImage(0, 0, layer.image)
        
        # Create a mask image for the selection
        mask_image = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        mask_image.fill(Qt.GlobalColor.transparent)
        
        # Draw the selection path with the new color
        mask_painter = QPainter(mask_image)
        mask_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        color.setAlpha(int(255 * self.opacity))  # Use tool opacity
        mask_painter.fillPath(self.selection_path, color)
        mask_painter.end()
        
        # Convert selection to QImage for masking
        selection_bytes = (selection.astype(np.uint8) * 255).tobytes()
        selection_img = QImage(selection_bytes, selection.shape[1], selection.shape[0], selection.shape[1], QImage.Format.Format_Grayscale8)
        
        # Apply selection mask to the color mask
        mask_painter = QPainter(mask_image)
        mask_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        mask_painter.drawImage(0, 0, selection_img)
        mask_painter.end()
        
        # Blend the mask with the original image
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(0, 0, mask_image)
        painter.end()
        
        # Update the layer
        layer.set_image(new_image)
        layer.changed.emit()
        self.canvas.update()

    def copy_selection(self) -> None:
        """Copy the selected area without removing it."""
        if not hasattr(self.canvas, 'get_selection'):
            return
        
        # Get the current selection mask
        selection = self.canvas.get_selection()
        if selection is None or not isinstance(selection, np.ndarray):
            return
        
        # Get the active layer
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer or not hasattr(layer, 'image'):
            return
        
        # Convert selection to boolean and find bounds
        selection = selection.astype(bool)
        if not selection.any():
            return
            
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
        painter.end()
        
        # Store the copied content and its original position
        self.clipboard_image = copy_image
        self.clipboard_pos = QPoint(min_x, min_y)

    def start_floating_paste(self, pos: QPoint) -> None:
        """Start floating paste mode at the given position."""
        if self.clipboard_image is None:
            return
        
        self.is_floating = True
        self.floating_pos = pos
        self.canvas.update()

    def update_floating_position(self, pos: QPoint) -> None:
        """Update the position of the floating selection."""
        if self.is_floating:
            self.floating_pos = pos
            self.canvas.update()

    def finalize_paste(self) -> None:
        """Finalize the paste operation at the current floating position."""
        if self.is_floating and self.floating_pos is not None:
            self.paste_selection(self.floating_pos)
            self.is_floating = False
            self.floating_pos = None
            self.canvas.update()

    def mouse_move(self, event: QMouseEvent) -> None:
        pos = self.canvas.get_transformed_pos(event.pos())
        if self.is_floating:
            # Update floating selection position
            offset_x = self.clipboard_image.width() // 2 if self.clipboard_image else 0
            offset_y = self.clipboard_image.height() // 2 if self.clipboard_image else 0
            self.floating_pos = QPoint(pos.x() - offset_x, pos.y() - offset_y)
            self.canvas.update()
            return
        
        super().mouse_move(event)

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
        self.points: List[QPoint] = []

    def mouse_press(self, event: QMouseEvent) -> None:
        super().mouse_press(event)
        pos = self.canvas.get_transformed_pos(event.pos())
        self.points = [pos]
        self.selection_path = QPainterPath()
        self.selection_path.moveTo(QPointF(pos))

    def mouse_move(self, event: QMouseEvent) -> None:
        super().mouse_move(event)
        if self.is_active and self.current_pos is not None:
            self.points.append(self.current_pos)
            self.selection_path.lineTo(QPointF(self.current_pos))

    def mouse_release(self, event: QMouseEvent) -> None:
        if self.is_active and len(self.points) > 0:
            self.points.append(self.points[0])  # Close the path
            self.selection_path.lineTo(QPointF(self.points[0]))
        super().mouse_release(event)

    def draw_selection_shape(self, painter: QPainter) -> None:
        if self.is_active and len(self.points) > 1:
            painter.drawPath(self.selection_path)

    def get_selection_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a boolean mask for the current selection.

        Args:
            width (int): Width of the mask
            height (int): Height of the mask

        Returns:
            np.ndarray: Boolean mask where True indicates selected pixels
        """
        if not self.points or len(self.points) < 3:
            return np.zeros((height, width), dtype=bool)

        # Convert points to numpy array for faster processing
        points = np.array([(p.x(), p.y()) for p in self.points])

        # Create bounding box for optimization
        min_x = max(0, int(np.min(points[:, 0])))
        max_x = min(width - 1, int(np.ceil(np.max(points[:, 0]))))
        min_y = max(0, int(np.min(points[:, 1])))
        max_y = min(height - 1, int(np.ceil(np.max(points[:, 1]))))

        # Initialize mask
        mask = np.zeros((height, width), dtype=bool)

        # Process each row within bounding box
        for y in range(min_y, max_y + 1):
            crossings = []
            
            # Check each edge
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                # Skip if edge is outside current y
                if y < min(p1[1], p2[1]) or y >= max(p1[1], p2[1]):
                    continue
                
                # Handle horizontal edges
                if p1[1] == p2[1] and y == p1[1]:
                    x_min = min(p1[0], p2[0])
                    x_max = max(p1[0], p2[0])
                    # Get adjacent points
                    prev = points[(i - 1) % len(points)]
                    next = points[(i + 2) % len(points)]
                    # Add crossings if adjacent edges cross in opposite directions
                    if prev[1] < y and next[1] > y:  # Up to down
                        crossings.extend([x_min, x_max])
                    elif prev[1] > y and next[1] < y:  # Down to up
                        crossings.extend([x_min, x_max])
                    continue
                
                # Calculate x-intersection for non-horizontal edges
                if p1[1] != p2[1]:
                    t = (y - p1[1]) / (p2[1] - p1[1])
                    if 0 <= t <= 1:  # Intersection is on the edge
                        x = p1[0] + t * (p2[0] - p1[0])
                        
                        # Handle vertices
                        if y == p1[1]:  # Starting vertex
                            prev = points[(i - 1) % len(points)]
                            if prev[1] != y:  # Not part of horizontal edge
                                if (prev[1] < y and p2[1] > y) or (prev[1] > y and p2[1] < y):
                                    crossings.append(x)
                        elif y == p2[1]:  # Ending vertex
                            next = points[(i + 2) % len(points)]
                            if next[1] != y:  # Not part of horizontal edge
                                if (p1[1] < y and next[1] > y) or (p1[1] > y and next[1] < y):
                                    crossings.append(x)
                        else:  # Regular crossing
                            crossings.append(x)
            
            if crossings:
                # Sort crossings
                crossings.sort()
                
                # Fill between pairs of crossings
                for i in range(0, len(crossings) - 1, 2):
                    start_x = max(min_x, int(np.ceil(crossings[i])))
                    end_x = min(max_x, int(crossings[i + 1]))
                    if start_x <= end_x:
                        mask[y, start_x:end_x + 1] = True

        # Apply Gaussian blur for feathering if radius > 0
        if self.feather_radius > 0:
            mask = mask.astype(np.float32)
            mask = cv2.GaussianBlur(mask, (0, 0), self.feather_radius)
            mask = mask > 0.5

        return mask