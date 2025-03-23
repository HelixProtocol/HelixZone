"""Tests for the type checker module."""

import os
import numpy as np
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QSize
from pathlib import Path
from helixzone.core.type_checker import (
    validate_image,
    ensure_numpy_array,
    qimage_to_numpy,
    ProcessingError,
    ProcessingParams
)

@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image file."""
    # Create a simple test image
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[25:75, 25:75] = 255  # White square in the middle
    
    # Save as PNG
    img_path = tmp_path / "test.png"
    qimg = QImage(
        img_data.data,
        img_data.shape[1],
        img_data.shape[0],
        img_data.shape[1] * 3,
        QImage.Format.Format_RGB888
    )
    qimg.save(str(img_path))
    
    return img_path

@pytest.fixture
def test_qimage():
    """Create a test QImage."""
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[25:75, 25:75] = 255
    
    return QImage(
        img_data.data,
        img_data.shape[1],
        img_data.shape[0],
        img_data.shape[1] * 3,
        QImage.Format.Format_RGB888
    )

@pytest.fixture
def test_array():
    """Create a test numpy array."""
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[25:75, 25:75] = 255
    return img_data

def test_validate_image_path(test_image_path):
    """Test image validation with file path."""
    assert validate_image(test_image_path)
    assert validate_image(str(test_image_path))
    assert not validate_image("nonexistent.png")
    assert not validate_image("invalid.txt")

def test_validate_image_qimage(test_qimage):
    """Test image validation with QImage."""
    assert validate_image(test_qimage)
    
    # Test null QImage
    null_image = QImage()
    assert not validate_image(null_image)

def test_validate_image_array(test_array):
    """Test image validation with numpy array."""
    assert validate_image(test_array)
    
    # Test invalid arrays
    invalid_dtype = np.zeros((100, 100), dtype=np.float32)
    assert not validate_image(invalid_dtype)
    
    invalid_shape = np.zeros((100,), dtype=np.uint8)
    assert not validate_image(invalid_shape)
    
    invalid_channels = np.zeros((100, 100, 5), dtype=np.uint8)
    assert not validate_image(invalid_channels)

def test_ensure_numpy_array_path(test_image_path):
    """Test array conversion from file path."""
    arr = ensure_numpy_array(test_image_path)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (100, 100, 3)
    
    with pytest.raises(ProcessingError):
        ensure_numpy_array("nonexistent.png")

def test_ensure_numpy_array_qimage(test_qimage):
    """Test array conversion from QImage."""
    arr = ensure_numpy_array(test_qimage)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (100, 100, 3)
    
    with pytest.raises(ProcessingError):
        ensure_numpy_array(QImage())

def test_ensure_numpy_array_array(test_array):
    """Test array conversion from numpy array."""
    arr = ensure_numpy_array(test_array)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (100, 100, 3)
    assert np.array_equal(arr, test_array)
    
    with pytest.raises(ProcessingError):
        ensure_numpy_array(np.zeros((100,), dtype=np.uint8))

def test_qimage_to_numpy(test_qimage):
    """Test QImage to numpy array conversion."""
    arr = qimage_to_numpy(test_qimage)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (100, 100, 3)
    
    # Test with alpha channel
    rgba_image = test_qimage.convertToFormat(QImage.Format.Format_RGBA8888)
    arr = qimage_to_numpy(rgba_image)
    assert arr.shape[-1] == 3  # Should strip alpha if all 255
    
    with pytest.raises(ProcessingError):
        qimage_to_numpy(QImage())

def test_processing_params():
    """Test ProcessingParams dataclass."""
    # Test default values
    params = ProcessingParams(
        sigma=1.0,
        threshold=0.5,
        kernel_size=3,
        iterations=1,
        normalize=True
    )
    assert params.sigma == 1.0
    assert params.threshold == 0.5
    assert params.kernel_size == 3
    assert params.iterations == 1
    assert params.normalize is True
    
    # Test custom values
    custom_params = ProcessingParams(
        sigma=2.0,
        threshold=0.7,
        kernel_size=5,
        iterations=2,
        normalize=False
    )
    assert custom_params.sigma == 2.0
    assert custom_params.threshold == 0.7
    assert custom_params.kernel_size == 5
    assert custom_params.iterations == 2
    assert custom_params.normalize is False 