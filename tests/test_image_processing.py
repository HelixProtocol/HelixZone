import pytest
import numpy as np
from PyQt6.QtGui import QImage, QColor
from helixzone.core.layer import Layer

def test_layer_creation():
    """Test basic layer creation and properties."""
    layer = Layer(name="Test Layer", size=(100, 100))
    assert layer.name == "Test Layer"
    assert layer.image.width() == 100
    assert layer.image.height() == 100
    assert layer.visible == True
    assert layer.opacity == 1.0

def test_layer_opacity():
    """Test layer opacity settings."""
    layer = Layer()
    layer.set_opacity(0.5)
    assert layer.opacity == 0.5
    # Test bounds
    layer.set_opacity(-0.1)
    assert layer.opacity == 0.0
    layer.set_opacity(1.5)
    assert layer.opacity == 1.0

def test_image_clear():
    """Test clearing layer content."""
    layer = Layer(size=(50, 50))
    # Fill with red
    layer.image.fill(QColor(255, 0, 0).rgb())
    # Clear
    layer.clear()
    # Check transparency
    assert layer.image.pixel(0, 0) == 0

@pytest.mark.performance
def test_large_image_handling():
    """Test performance with large images."""
    size = (4000, 4000)  # 16MP image
    layer = Layer(size=size)
    # Fill with pattern
    for x in range(0, size[0], 100):
        for y in range(0, size[1], 100):
            layer.image.setPixel(x, y, QColor(255, 0, 0).rgb())
    # Should handle large images without memory issues
    layer.resize(2000, 2000)  # Test downscaling
    assert layer.image.width() == 2000
    assert layer.image.height() == 2000

@pytest.mark.memory
def test_memory_management():
    """Test for memory leaks in image operations."""
    initial_layers = []
    for _ in range(100):
        layer = Layer(size=(1000, 1000))
        layer.image.fill(QColor(255, 0, 0).rgb())
        initial_layers.append(layer)
    # Force cleanup
    initial_layers.clear()
    # If no memory leak, this should complete without issues
    final_layer = Layer(size=(1000, 1000))
    assert final_layer.image.width() == 1000

def test_numpy_conversion():
    """Test conversion between QImage and numpy array."""
    # Create test pattern
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[25:75, 25:75] = [255, 0, 0]  # Red square
    
    layer = Layer(size=(100, 100))
    layer.set_image(arr)
    
    # Check if pattern is preserved
    pixel = layer.image.pixel(50, 50)
    color = QColor(pixel)
    assert color.red() == 255
    assert color.green() == 0
    assert color.blue() == 0 