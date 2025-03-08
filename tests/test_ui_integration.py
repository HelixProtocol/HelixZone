import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QPainterPath
from helixzone.gui.main_window import MainWindow
from helixzone.gui.canvas import CanvasView
from helixzone.core.tools import BrushTool, EraserTool
from helixzone.core.commands import DrawCommand
import numpy as np

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
    assert isinstance(main_window.canvas_view.canvas.tool_manager.current_tool, BrushTool)
    
    # Select eraser tool
    main_window.select_tool('eraser')
    assert isinstance(main_window.canvas_view.canvas.tool_manager.current_tool, EraserTool)

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
        command = DrawCommand(layer, layer.image)
        layer.image.setPixel(0, 0, QColor(255, 0, 0).rgb())
        command.execute()
        canvas.command_stack.push(command)
        
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

@pytest.mark.integration
def test_lasso_selection(main_window):
    """Test lasso selection tool functionality."""
    canvas = main_window.canvas_view.canvas
    
    # Select lasso tool
    main_window.select_tool('lasso_selection')
    
    # Create a triangular selection
    points = [
        QPoint(100, 100),  # Top
        QPoint(150, 200),  # Bottom right
        QPoint(50, 200),   # Bottom left
    ]
    
    # Simulate mouse events for drawing the selection
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Move to each point
    for point in points[1:]:
        canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    
    # Release to complete the selection
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: points[-1], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Get the selection mask
    tool = canvas.tool_manager.get_current_tool()
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    
    # Verify the mask is a boolean numpy array
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    
    # Check that some points inside the triangle are selected
    assert mask[150, 100]  # Point inside triangle
    assert not mask[50, 50]  # Point outside triangle
    
    # Verify the selection path was created
    assert len(tool.points) >= 3  # Should have at least 3 points for triangle
    assert isinstance(tool.selection_path, QPainterPath)
    assert not tool.selection_path.isEmpty()

@pytest.mark.integration
def test_lasso_selection_edge_cases(main_window):
    """Test lasso selection tool edge cases."""
    canvas = main_window.canvas_view.canvas
    main_window.select_tool('lasso_selection')
    tool = canvas.tool_manager.get_current_tool()
    
    # Test 1: Single point selection (should result in empty selection)
    point = QPoint(100, 100)
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: point, 'button': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: point, 'button': lambda: Qt.MouseButton.LeftButton}))
    
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    assert not mask.any()  # Mask should be all False
    assert len(tool.points) <= 2  # Should have at most 2 points (start and end)
    
    # Test 2: Two-point selection (should result in empty selection)
    points = [QPoint(100, 100), QPoint(200, 200)]
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: points[1], 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: points[1], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    assert not mask.any()  # Mask should be all False
    
    # Test 3: Complex shape (star-like pattern)
    star_points = [
        QPoint(150, 100),  # Top
        QPoint(175, 140),  # Right upper
        QPoint(200, 120),  # Right point
        QPoint(175, 160),  # Right lower
        QPoint(150, 200),  # Bottom
        QPoint(125, 160),  # Left lower
        QPoint(100, 120),  # Left point
        QPoint(125, 140),  # Left upper
    ]
    
    # Draw the star
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: star_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    for point in star_points[1:]:
        canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: star_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    # Check points that should be inside and outside the star
    assert mask[150, 150]  # Center should be selected
    assert not mask[50, 50]  # Far corner should not be selected
    assert len(tool.points) > 8  # Should have at least the number of star points

@pytest.mark.integration
def test_lasso_selection_feathering(main_window):
    """Test lasso selection tool feathering functionality."""
    canvas = main_window.canvas_view.canvas
    main_window.select_tool('lasso_selection')
    tool = canvas.tool_manager.get_current_tool()
    
    # Create a simple triangular selection
    points = [
        QPoint(100, 100),  # Top
        QPoint(150, 200),  # Bottom right
        QPoint(50, 200),   # Bottom left
    ]
    
    # Test different feather radii
    feather_radii = [0, 5, 10]
    for radius in feather_radii:
        # Set feather radius
        tool.feather_radius = radius
        
        # Draw the selection
        canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
        for point in points[1:]:
            canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
        canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
        
        # Get the selection mask
        mask = tool.get_selection_mask(canvas.width(), canvas.height())
        
        # Basic assertions
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        
        if radius == 0:
            # For no feathering, check exact boundaries
            assert mask[150, 100]  # Point inside triangle
            assert not mask[50, 50]  # Point outside triangle
        else:
            # For feathered selections, check that the mask has some true values
            assert mask.any()
            # The feathered region should be larger than the non-feathered region
            non_feathered = tool.get_selection_mask(canvas.width(), canvas.height())
            assert mask.sum() >= non_feathered.sum() 

@pytest.mark.performance
def test_lasso_selection_performance(main_window):
    """Test lasso selection performance with large numbers of points."""
    import time
    canvas = main_window.canvas_view.canvas
    main_window.select_tool('lasso_selection')
    tool = canvas.tool_manager.get_current_tool()
    
    # Create a spiral pattern with many points
    center_x, center_y = 400, 300
    points = []
    for i in range(720):  # Two full rotations
        angle = np.radians(i)
        radius = 100 + i / 10
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append(QPoint(int(x), int(y)))
    
    # Measure time for drawing and mask creation
    start_time = time.time()
    
    # Draw the spiral
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    for point in points[1:]:
        canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Get the selection mask
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 2.0  # Should complete within 2 seconds
    assert len(tool.points) >= 720  # Should have captured all points
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    
    # Verify the spiral selection
    assert mask[center_y, center_x]  # Center should be selected
    assert not mask[0, 0]  # Corner should not be selected

@pytest.mark.integration
def test_lasso_selection_complex_feathering(main_window):
    """Test feathering with complex shapes and edge cases."""
    canvas = main_window.canvas_view.canvas
    main_window.select_tool('lasso_selection')
    tool = canvas.tool_manager.get_current_tool()
    
    # Test 1: Self-intersecting shape (figure-8)
    figure8_points = [
        QPoint(150, 100),  # Top of upper loop
        QPoint(200, 150),  # Right of upper loop
        QPoint(150, 200),  # Bottom of upper loop
        QPoint(100, 150),  # Left of upper loop
        QPoint(150, 100),  # Back to top (complete upper loop)
        QPoint(150, 200),  # To bottom of upper loop
        QPoint(200, 250),  # Right of lower loop
        QPoint(150, 300),  # Bottom of lower loop
        QPoint(100, 250),  # Left of lower loop
        QPoint(150, 200),  # Back to intersection
    ]
    
    # Draw the figure-8
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: figure8_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    for point in figure8_points[1:]:
        canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: figure8_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Test with different feather radii
    for radius in [0, 5, 10]:
        tool.feather_radius = radius
        mask = tool.get_selection_mask(canvas.width(), canvas.height())
        
        # Verify basic properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        
        if radius == 0:
            # Check specific points in unfeathered selection
            assert mask[150, 150]  # Center of upper loop
            assert mask[150, 250]  # Center of lower loop
            assert not mask[50, 50]  # Outside point
        else:
            # For feathered selection, verify the feathering effect
            non_feathered = tool.get_selection_mask(canvas.width(), canvas.height())
            assert mask.sum() > non_feathered.sum()
            
            # Check that feathering preserves the general shape
            assert mask[150, 150]  # Center of upper loop should still be selected
            assert mask[150, 250]  # Center of lower loop should still be selected
    
    # Test 2: Sharp corners and thin regions
    zigzag_points = [
        QPoint(100, 100),
        QPoint(200, 110),
        QPoint(100, 120),
        QPoint(200, 130),
        QPoint(100, 140),
        QPoint(200, 150),
    ]
    
    # Draw the zigzag
    canvas.mousePressEvent(type('MockEvent', (), {'pos': lambda: zigzag_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    for point in zigzag_points[1:]:
        canvas.mouseMoveEvent(type('MockEvent', (), {'pos': lambda: point, 'buttons': lambda: Qt.MouseButton.LeftButton}))
    canvas.mouseReleaseEvent(type('MockEvent', (), {'pos': lambda: zigzag_points[0], 'button': lambda: Qt.MouseButton.LeftButton}))
    
    # Test with large feather radius
    tool.feather_radius = 15
    mask = tool.get_selection_mask(canvas.width(), canvas.height())
    
    # Verify that thin regions are still preserved
    assert mask[115, 150]  # Point in the middle of zigzag
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_ 