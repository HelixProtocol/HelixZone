import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor
from helixzone.gui.main_window import MainWindow
from helixzone.gui.canvas import CanvasView

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication([])

@pytest.fixture
def main_window(app):
    """Create the main window instance."""
    window = MainWindow()
    return window

def test_tool_selection(main_window):
    """Test tool selection and state."""
    # Select brush tool
    main_window.select_tool('brush')
    assert main_window.canvas_view.canvas.tool_manager.current_tool == 'brush'
    
    # Select eraser tool
    main_window.select_tool('eraser')
    assert main_window.canvas_view.canvas.tool_manager.current_tool == 'eraser'

def test_layer_operations(main_window):
    """Test layer management operations."""
    initial_count = len(main_window.canvas_view.canvas.layer_stack.layers)
    
    # Add new layer
    main_window.add_layer()
    assert len(main_window.canvas_view.canvas.layer_stack.layers) == initial_count + 1
    
    # Test layer visibility
    layer = main_window.canvas_view.canvas.layer_stack.get_active_layer()
    if layer:
        layer.set_visible(False)
        assert not layer.visible
        layer.set_visible(True)
        assert layer.visible

@pytest.mark.integration
def test_drawing_operations(main_window):
    """Test basic drawing operations."""
    canvas = main_window.canvas_view.canvas
    
    # Select brush tool
    main_window.select_tool('brush')
    
    # Simulate drawing
    start_pos = QPoint(100, 100)
    end_pos = QPoint(200, 200)
    
    # Get color before
    color_before = canvas.get_image().pixel(100, 100)
    
    # Simulate mouse events
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: start_pos, 'button': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: end_pos, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: end_pos, 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Get color after
    color_after = canvas.get_image().pixel(100, 100)
    
    # Should have changed the pixel color
    assert color_before != color_after

@pytest.mark.integration
def test_undo_redo(main_window):
    """Test undo/redo functionality."""
    canvas = main_window.canvas_view.canvas
    
    # Make a change
    layer = canvas.layer_stack.get_active_layer()
    if layer:
        original_color = layer.image.pixel(0, 0)
        layer.image.setPixel(0, 0, QColor(255, 0, 0).rgb())
        canvas.command_stack.push_command("test")
        
        # Test undo
        canvas.undo()
        assert layer.image.pixel(0, 0) == original_color
        
        # Test redo
        canvas.redo()
        assert layer.image.pixel(0, 0) == QColor(255, 0, 0).rgb()

@pytest.mark.performance
def test_ui_responsiveness(main_window):
    """Test UI responsiveness with heavy operations."""
    canvas = main_window.canvas_view.canvas
    
    # Create large image
    for _ in range(10):
        main_window.add_layer()
    
    # Perform multiple operations
    for i in range(100):
        layer = canvas.layer_stack.get_active_layer()
        if layer:
            layer.image.setPixel(i, i, QColor(255, 0, 0).rgb())
            canvas.update()  # Should not freeze UI 