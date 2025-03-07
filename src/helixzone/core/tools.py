from PyQt6.QtCore import Qt, QPoint, QRect, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
import numpy as np

class Tool:
    """Base class for all tools."""
    def __init__(self, canvas):
        self.canvas = canvas
        self.active = False
        self.last_pos = None
        
    def mouse_press(self, event):
        """Handle mouse press events."""
        pass
        
    def mouse_release(self, event):
        """Handle mouse release events."""
        pass
        
    def mouse_move(self, event):
        """Handle mouse move events."""
        pass
        
    def get_cursor(self):
        """Return the cursor to use for this tool."""
        return Qt.CursorShape.ArrowCursor

class BrushTool(Tool):
    """Brush tool for painting."""
    def __init__(self, canvas):
        super().__init__(canvas)
        self.size = 10
        self.hardness = 0.8  # 0.0 to 1.0
        self.opacity = 1.0   # 0.0 to 1.0
        self.color = QColor(0, 0, 0)  # Default to black
        
    def mouse_press(self, event):
        self.active = True
        pos = self.canvas.get_transformed_pos(event.pos())
        self.last_pos = pos
        self.draw_stroke(pos, pos)
        
    def mouse_release(self, event):
        self.active = False
        self.last_pos = None
        
    def mouse_move(self, event):
        if self.active and self.last_pos:
            pos = self.canvas.get_transformed_pos(event.pos())
            self.draw_stroke(self.last_pos, pos)
            self.last_pos = pos
            
    def get_cursor(self):
        return Qt.CursorShape.CrossCursor
        
    def draw_stroke(self, start_pos, end_pos):
        """Draw a stroke between two points."""
        # Get the active layer
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer:
            return
            
        # Create a painter for the layer
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up the pen
        pen = QPen()
        pen.setColor(self.color)
        pen.setWidth(self.size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        # Draw the line
        path = QPainterPath()
        path.moveTo(QPointF(start_pos))
        path.lineTo(QPointF(end_pos))
        painter.strokePath(path, pen)
        
        painter.end()
        
        # Update the canvas
        layer.changed.emit()
        self.canvas.update()

class EraserTool(BrushTool):
    """Eraser tool - similar to brush but clears pixels."""
    def __init__(self, canvas):
        super().__init__(canvas)
        self.color = QColor(0, 0, 0, 0)  # Transparent
        
    def draw_stroke(self, start_pos, end_pos):
        """Draw a transparent stroke between two points."""
        layer = self.canvas.layer_stack.get_active_layer()
        if not layer:
            return
            
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        
        pen = QPen()
        pen.setColor(self.color)
        pen.setWidth(self.size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        path = QPainterPath()
        path.moveTo(QPointF(start_pos))
        path.lineTo(QPointF(end_pos))
        painter.strokePath(path, pen)
        
        painter.end()
        
        layer.changed.emit()
        self.canvas.update()

class ToolManager:
    """Manages the current tool and tool switching."""
    def __init__(self, canvas):
        self.canvas = canvas
        self.tools = {
            'brush': BrushTool(canvas),
            'eraser': EraserTool(canvas)
        }
        self.current_tool = self.tools['brush']
        
    def set_tool(self, tool_name):
        """Switch to a different tool."""
        if tool_name in self.tools:
            self.current_tool = self.tools[tool_name]
            # Update cursor
            self.canvas.setCursor(self.current_tool.get_cursor())
            
    def get_current_tool(self):
        """Get the current tool."""
        return self.current_tool 