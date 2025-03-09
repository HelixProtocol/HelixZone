import pytest
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QMouseEvent, QPainterPath
from PyQt6.QtTest import QTest
from typing import Optional, List, Dict, Any, cast, Union
from helixzone.core.tools import (
    LassoSelection,
    MagneticLassoSelection,
    RectangleSelection,
    EllipseSelection,
    Tool
)
from helixzone.gui.canvas import Canvas
from helixzone.core.layer import Layer

class MockCanvas(Canvas):
    """Mock canvas for testing selection tools."""
    def __init__(self):
        super().__init__(None)  # Pass None as parent
        self._width = 500
        self._height = 500
        self.layer_stack = MockLayerStack()
        self.tool_manager = MockToolManager()
        self.command_stack = MockCommandStack()
        self.update_requested = False
        self._selection_mask = None
    
    def width(self) -> int:
        return self._width
    
    def height(self) -> int:
        return self._height
    
    def update(self):
        self.update_requested = True
    
    def get_image(self):
        return self.layer_stack.get_active_layer().image
    
    def get_transformed_pos(self, pos: Union[QPoint, QPointF]) -> QPointF:
        """Convert screen coordinates to image coordinates."""
        try:
            # Convert to QPointF if needed
            if isinstance(pos, QPoint):
                pos = QPointF(pos)
            elif not isinstance(pos, QPointF):
                return QPointF(0, 0)  # Return safe default if pos is invalid
            
            # Ensure coordinates are within bounds
            x = max(0.0, min(pos.x(), float(self._width - 1)))
            y = max(0.0, min(pos.y(), float(self._height - 1)))
            
            return QPointF(x, y)
        except Exception as e:
            print(f"Error in get_transformed_pos: {e}")
            return QPointF(0, 0)
    
    def get_selection(self) -> Optional[np.ndarray]:
        """Get the current selection mask."""
        return self._selection_mask
    
    def set_selection(self, mask: np.ndarray) -> None:
        """Set the selection mask."""
        if mask is not None:
            self._selection_mask = mask.astype(np.float32)
            self.update()
    
    def clear_selection(self) -> None:
        """Clear the current selection."""
        self._selection_mask = None
        self.update()

class MockLayerStack:
    """Mock layer stack for testing."""
    def __init__(self):
        # Create a blank QImage for testing
        test_image = QImage(500, 500, QImage.Format.Format_ARGB32)
        test_image.fill(Qt.GlobalColor.white)  # Fill with white background
        
        # Create the layer with the test image
        self.active_layer = Layer("Layer 1", (500, 500))
        self.active_layer.image = test_image  # Set the image directly
        self.active_layer.changed.emit()  # Emit signal to notify of change
        self.layers = [self.active_layer]
    
    def get_active_layer(self):
        return self.active_layer

class MockToolManager:
    """Mock tool manager for testing."""
    def __init__(self):
        self.current_tool = None
        self.tools = {}
    
    def get_current_tool(self):
        return self.current_tool
    
    def set_current_tool(self, tool):
        self.current_tool = tool
    
    def set_tool(self, tool_name: str) -> None:
        """Switch to a different tool."""
        from helixzone.core.tools import BrushTool, EraserTool, RectangleSelection, EllipseSelection, LassoSelection, MagneticLassoSelection
        
        if not self.tools:
            # Initialize tools if not already done
            canvas = self.current_tool.canvas if self.current_tool else None
            if canvas:
                self.tools = {
                    'brush': BrushTool(canvas),
                    'eraser': EraserTool(canvas),
                    'rectangle_selection': RectangleSelection(canvas),
                    'ellipse_selection': EllipseSelection(canvas),
                    'lasso_selection': LassoSelection(canvas),
                    'magnetic_lasso': MagneticLassoSelection(canvas)
                }
        
        if tool_name in self.tools:
            self.current_tool = self.tools[tool_name]

class MockCommandStack:
    """Mock command stack for testing."""
    def __init__(self):
        self.commands = []
    
    def push(self, command):
        self.commands.append(command)
    
    def undo(self):
        if self.commands:
            self.commands.pop().undo()
    
    def redo(self):
        if self.commands:
            self.commands[-1].redo()

class TestLassoSelection:
    """Test suite for Lasso Selection tool."""
    
    @pytest.fixture
    def canvas(self):
        return MockCanvas()
        
    @pytest.fixture
    def lasso(self, canvas):
        return LassoSelection(canvas)
        
    @staticmethod
    def _create_mouse_event(pos: QPointF, event_type: QEvent.Type = QEvent.Type.MouseButtonPress) -> QMouseEvent:
        """Create a proper QMouseEvent for testing."""
        try:
            # Ensure pos is QPointF
            if not isinstance(pos, QPointF):
                pos = QPointF(pos)
            
            # Set up buttons based on event type
            if event_type == QEvent.Type.MouseMove:
                button = Qt.MouseButton.NoButton
                buttons = Qt.MouseButton.LeftButton
            else:
                button = Qt.MouseButton.LeftButton
                buttons = Qt.MouseButton.LeftButton
            
            # Create event with proper position handling
            return QMouseEvent(
                event_type,
                pos,  # localPos
                pos,  # globalPos (same as local for test)
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier
            )
        except Exception as e:
            print(f"Error creating mouse event: {e}")
            # Return a safe default event
            return QMouseEvent(
                event_type,
                QPointF(0, 0),
                QPointF(0, 0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier
            )
        
    def test_basic_selection(self, lasso):
        """Test basic lasso selection creation."""
        try:
            # Reset tool state and verify
            lasso.reset_state()
            assert not lasso.is_active, "Tool should not be active after reset"
            assert len(lasso.points) == 0, "Points should be empty after reset"
            
            # Simulate drawing a triangle with QPointF points
            points = [
                QPointF(100.0, 100.0),
                QPointF(200.0, 100.0),
                QPointF(150.0, 200.0),
                QPointF(100.0, 100.0)  # Close the path
            ]
            
            # Start selection
            print("Starting selection...")
            start_event = self._create_mouse_event(points[0], QEvent.Type.MouseButtonPress)
            lasso.mouse_press(start_event)
            assert lasso.is_active, "Tool should be active after mouse press"
            assert len(lasso.points) > 0, "Should have initial point after mouse press"
            
            # Draw points
            print("Drawing selection points...")
            for i, point in enumerate(points[1:-1]):
                move_event = self._create_mouse_event(point, QEvent.Type.MouseMove)
                lasso.mouse_move(move_event)
                assert len(lasso.points) >= i + 2, f"Expected at least {i + 2} points, got {len(lasso.points)}"
            
            # Complete selection with the closing point
            print("Completing selection...")
            end_event = self._create_mouse_event(points[-1], QEvent.Type.MouseButtonRelease)
            lasso.mouse_release(end_event)
            assert not lasso.is_active, "Tool should not be active after mouse release"
            
            # Get and verify selection mask
            print("Verifying selection mask...")
            mask = lasso.get_selection_mask(500, 500)
            assert mask is not None, "Selection mask should not be None"
            assert mask.shape == (500, 500), f"Expected shape (500, 500), got {mask.shape}"
            assert mask.any(), "Selection mask should have some selected pixels"
            assert mask.dtype == bool, f"Expected boolean dtype, got {mask.dtype}"
            
            # Verify the selection is properly closed
            print("Verifying selection closure...")
            if len(lasso.points) >= 2:
                first_point = lasso.points[0]
                last_point = lasso.points[-1]
                assert np.allclose(
                    [first_point.x(), first_point.y()],
                    [last_point.x(), last_point.y()],
                    rtol=1e-5
                ), "Selection path should be closed"
            
            # Verify canvas state
            print("Verifying canvas state...")
            canvas_selection = lasso.canvas.get_selection()
            assert canvas_selection is not None, "Canvas selection should not be None"
            assert canvas_selection.shape == (500, 500), "Canvas selection should have correct shape"
            assert canvas_selection.any(), "Canvas selection should have selected pixels"
            
            # Clear selection for next test
            print("Cleaning up...")
            lasso.canvas.clear_selection()
            lasso.reset_state()
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def test_feathering(self, lasso):
        """Test selection feathering."""
        lasso.feather_radius = 5
        
        # Create simple selection
        points = [
            QPointF(100.0, 100.0),
            QPointF(200.0, 100.0),
            QPointF(150.0, 200.0),
            QPointF(100.0, 100.0)
        ]
        
        # Draw selection
        start_event = self._create_mouse_event(points[0], QEvent.Type.MouseButtonPress)
        lasso.mouse_press(start_event)
        
        for point in points[1:]:
            move_event = self._create_mouse_event(point, QEvent.Type.MouseMove)
            lasso.mouse_move(move_event)
        
        end_event = self._create_mouse_event(points[-1], QEvent.Type.MouseButtonRelease)
        lasso.mouse_release(end_event)
        
        # Get feathered mask
        mask = lasso.get_selection_mask(500, 500)
        
        # Verify feathering created smooth edges
        assert mask.dtype == bool
        assert np.sum(mask) > 0  # Should have selected pixels
        
    def test_cut_copy_paste(self, lasso):
        """Test cut, copy, and paste operations."""
        # Create selection
        points = [
            QPointF(100.0, 100.0),
            QPointF(200.0, 100.0),
            QPointF(150.0, 200.0),
            QPointF(100.0, 100.0)
        ]
        
        start_event = self._create_mouse_event(points[0], QEvent.Type.MouseButtonPress)
        lasso.mouse_press(start_event)
        
        for point in points[1:]:
            move_event = self._create_mouse_event(point, QEvent.Type.MouseMove)
            lasso.mouse_move(move_event)
        
        end_event = self._create_mouse_event(points[-1], QEvent.Type.MouseButtonRelease)
        lasso.mouse_release(end_event)
        
        # Test copy
        lasso.copy_selection()
        assert lasso.clipboard_image is not None
        
        # Test cut
        original_image = QImage(lasso.canvas.layer_stack.get_active_layer().image)
        lasso.cut_selection()
        assert lasso.clipboard_image is not None
        
        # Test paste
        paste_pos = QPointF(300.0, 300.0)
        lasso.paste_selection(paste_pos)
        
        # Verify changes
        current_image = lasso.canvas.layer_stack.get_active_layer().image
        assert not self._images_identical(original_image, current_image)
        
    def test_recolor(self, lasso):
        """Test recoloring selection."""
        # Create selection
        points = [
            QPointF(100.0, 100.0),
            QPointF(200.0, 100.0),
            QPointF(150.0, 200.0),
            QPointF(100.0, 100.0)
        ]
        
        start_event = self._create_mouse_event(points[0], QEvent.Type.MouseButtonPress)
        lasso.mouse_press(start_event)
        
        for point in points[1:]:
            move_event = self._create_mouse_event(point, QEvent.Type.MouseMove)
            lasso.mouse_move(move_event)
        
        end_event = self._create_mouse_event(points[-1], QEvent.Type.MouseButtonRelease)
        lasso.mouse_release(end_event)
        
        # Test recolor
        color = QColor(255, 0, 0)  # Red
        lasso.recolor_selection(color)
        
        # Verify changes
        current_image = lasso.canvas.layer_stack.get_active_layer().image
        assert current_image is not None
        
    @staticmethod
    def _images_identical(img1: QImage, img2: QImage) -> bool:
        """Compare two QImages for equality."""
        if img1.size() != img2.size():
            return False
            
        # Convert images to numpy arrays for comparison
        def qimage_to_array(img: QImage) -> np.ndarray:
            # Convert QImage to format that ensures proper comparison
            if img.format() != QImage.Format.Format_ARGB32:
                img = img.convertToFormat(QImage.Format.Format_ARGB32)
                
            width = img.width()
            height = img.height()
            ptr = img.constBits()
            if ptr is None:
                return np.array([])
            ptr.setsize(height * width * 4)
            return np.array(ptr).reshape(height, width, 4)
            
        arr1 = qimage_to_array(img1)
        arr2 = qimage_to_array(img2)
        
        if arr1.size == 0 or arr2.size == 0:
            return False
            
        # Compare all channels including alpha
        return np.array_equal(arr1, arr2)

class TestMagneticLassoSelection:
    """Test suite for Magnetic Lasso Selection tool."""
    
    @pytest.fixture
    def canvas(self):
        return MockCanvas()
    
    @pytest.fixture
    def magnetic_lasso(self, canvas):
        return MagneticLassoSelection(canvas)
    
    @staticmethod
    def _create_mouse_event(pos: QPointF, event_type: QEvent.Type = QEvent.Type.MouseButtonPress) -> QMouseEvent:
        """Create a proper QMouseEvent for testing."""
        try:
            # Ensure pos is QPointF
            if not isinstance(pos, QPointF):
                pos = QPointF(pos)
            
            # Set up buttons based on event type
            if event_type == QEvent.Type.MouseMove:
                button = Qt.MouseButton.NoButton
                buttons = Qt.MouseButton.LeftButton
            else:
                button = Qt.MouseButton.LeftButton
                buttons = Qt.MouseButton.LeftButton
            
            # Create event with proper position handling
            return QMouseEvent(
                event_type,
                pos,  # localPos
                pos,  # globalPos (same as local for test)
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier
            )
        except Exception as e:
            print(f"Error creating mouse event: {e}")
            # Return a safe default event
            return QMouseEvent(
                event_type,
                QPointF(0, 0),
                QPointF(0, 0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier
            )
    
    def test_edge_detection(self, magnetic_lasso):
        """Test edge detection functionality."""
        # Create test image with clear edges
        layer = magnetic_lasso.canvas.layer_stack.get_active_layer()
        image = layer.image
        
        # Draw a rectangle
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(100, 100, 200, 200)
        painter.end()
        
        # Update edge detection
        magnetic_lasso.update_edge_detection()
        
        # Verify edge detection results
        assert magnetic_lasso.edge_map is not None
        assert magnetic_lasso.edge_gradient is not None
        assert magnetic_lasso.edge_strength is not None
        
    def test_edge_snapping(self, magnetic_lasso):
        """Test edge snapping behavior."""
        # Create test image with clear edges
        layer = magnetic_lasso.canvas.layer_stack.get_active_layer()
        image = layer.image
        
        # Draw a rectangle
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(100, 100, 200, 200)
        painter.end()
        
        # Update edge detection
        magnetic_lasso.update_edge_detection()
        
        # Test snapping near edge
        test_point = QPointF(102.0, 150.0)  # Near vertical edge
        snapped_point = magnetic_lasso.find_edge_point(test_point)
        
        # Verify snapping
        assert abs(snapped_point.x() - 100) < magnetic_lasso.edge_width
        
    def test_low_contrast_handling(self, magnetic_lasso):
        """Test handling of low contrast regions."""
        # Create test image with gradual gradient
        layer = magnetic_lasso.canvas.layer_stack.get_active_layer()
        image = layer.image
        
        # Create gradient
        for x in range(image.width()):
            color = QColor(x // 2, x // 2, x // 2)
            for y in range(image.height()):
                image.setPixelColor(x, y, color)
        
        # Update edge detection
        magnetic_lasso.update_edge_detection()
        
        # Test point in low contrast region
        test_point = QPointF(250.0, 250.0)
        result_point = magnetic_lasso.handle_low_contrast(test_point)
        
        # Verify handling
        assert isinstance(result_point, QPointF)
        
    def test_complete_selection(self, magnetic_lasso):
        """Test complete selection process."""
        # Create test image
        layer = magnetic_lasso.canvas.layer_stack.get_active_layer()
        image = layer.image
        
        # Draw rectangle
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(100, 100, 200, 200)
        painter.end()
        
        # Update edge detection
        magnetic_lasso.update_edge_detection()
        
        # Simulate selection process
        points = [
            QPointF(100.0, 100.0),
            QPointF(300.0, 100.0),
            QPointF(300.0, 300.0),
            QPointF(100.0, 300.0),
            QPointF(100.0, 100.0)
        ]
        
        # Start selection
        magnetic_lasso.mouse_press(self._create_mouse_event(points[0], QEvent.Type.MouseButtonPress))
        
        # Draw selection
        for point in points[1:]:
            magnetic_lasso.current_pos = QPoint(round(point.x()), round(point.y()))
            magnetic_lasso.mouse_move(self._create_mouse_event(point, QEvent.Type.MouseMove))
        
        # Complete selection
        magnetic_lasso.mouse_release(self._create_mouse_event(points[-1], QEvent.Type.MouseButtonRelease))
        
        # Verify selection
        mask = magnetic_lasso.get_selection_mask(500, 500)
        assert mask.any()
        assert mask.shape == (500, 500)
        
def test_selection_tools_integration():
    """Integration test for all selection tools."""
    canvas = MockCanvas()
    tools = [
        RectangleSelection(canvas),
        EllipseSelection(canvas),
        LassoSelection(canvas),
        MagneticLassoSelection(canvas)
    ]
    
    # Test each tool
    for tool in tools:
        # Reset tool state
        tool.reset_state()
        
        # Create selection with proper size
        start_pos = QPointF(100.0, 100.0)
        end_pos = QPointF(300.0, 300.0)  # Larger selection area
        
        # Create proper mouse events with QPointF positions
        press_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start_pos,  # localPos
            start_pos,  # globalPos (same as local for test)
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            end_pos,  # localPos
            end_pos,  # globalPos (same as local for test)
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        release_event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            end_pos,  # localPos
            end_pos,  # globalPos (same as local for test)
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        # Start selection
        tool.mouse_press(press_event)
        
        # For lasso tools, create more points
        if isinstance(tool, (LassoSelection, MagneticLassoSelection)):
            # Create a square path with QPointF points
            points = [
                QPointF(100.0, 100.0),
                QPointF(300.0, 100.0),
                QPointF(300.0, 300.0),
                QPointF(100.0, 300.0),
                QPointF(100.0, 100.0)  # Close the path
            ]
            
            for point in points[1:]:  # Skip first point as it's already handled
                point_event = QMouseEvent(
                    QEvent.Type.MouseMove,
                    point,  # localPos
                    point,  # globalPos (same as local for test)
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier
                )
                tool.mouse_move(point_event)
        else:
            # For rectangle and ellipse, just move to end position
            tool.mouse_move(move_event)
        
        # Complete selection
        tool.mouse_release(release_event)
        
        # Test selection mask
        mask = tool.get_selection_mask(500, 500)
        assert mask is not None, f"Selection mask is None for {tool.__class__.__name__}"
        assert mask.any(), f"Selection mask is empty for {tool.__class__.__name__}"
        assert mask.shape == (500, 500), f"Incorrect mask shape for {tool.__class__.__name__}"
        
        # Test copy operation
        tool.reset_state()  # Reset before copy
        tool.copy_selection()
        assert tool.clipboard_image is not None, f"Copy failed for {tool.__class__.__name__}"
        
        # Test cut operation
        tool.reset_state()  # Reset before cut
        original_image = QImage(canvas.layer_stack.get_active_layer().image)
        tool.cut_selection()
        assert tool.clipboard_image is not None, f"Cut failed for {tool.__class__.__name__}"
        current_image = canvas.layer_stack.get_active_layer().image
        assert not TestLassoSelection._images_identical(original_image, current_image), f"Cut did not modify image for {tool.__class__.__name__}"
        
        # Test paste operation
        tool.reset_state()  # Reset before paste
        paste_pos = QPointF(400.0, 400.0)
        tool.paste_selection(paste_pos)
        
        # Test recolor operation
        tool.reset_state()  # Reset before recolor
        new_color = QColor(255, 0, 0)  # Red
        tool.recolor_selection(new_color)
        
        # Clear selection for next tool
        canvas.clear_selection()
        tool.reset_state() 