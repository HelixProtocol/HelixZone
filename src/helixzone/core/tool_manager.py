from typing import Dict, TYPE_CHECKING
from PyQt6.QtCore import QObject

if TYPE_CHECKING:
    from ..gui.canvas import Canvas
    from .tools import Tool, BrushTool, EraserTool, RectangleSelection, EllipseSelection, LassoSelection

class ToolManager(QObject):
    """Manages the current tool and tool switching."""
    def __init__(self, canvas: 'Canvas'):
        super().__init__()
        self.canvas = canvas
        
        # Import tools here to avoid circular imports
        from .tools import BrushTool, EraserTool, RectangleSelection, EllipseSelection, LassoSelection
        
        self.tools: Dict[str, 'Tool'] = {
            'brush': BrushTool(canvas),
            'eraser': EraserTool(canvas),
            'rectangle_selection': RectangleSelection(canvas),
            'ellipse_selection': EllipseSelection(canvas),
            'lasso_selection': LassoSelection(canvas)
        }
        self.current_tool = self.tools['brush']
        
        # Update cursor
        self.canvas.setCursor(self.current_tool.get_cursor())
    
    def set_tool(self, tool_name: str) -> None:
        """Switch to a different tool."""
        if tool_name in self.tools:
            self.current_tool = self.tools[tool_name]
            # Update cursor
            self.canvas.setCursor(self.current_tool.get_cursor())
    
    def get_current_tool(self) -> 'Tool':
        """Get the current tool."""
        return self.current_tool 