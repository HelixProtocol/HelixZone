"""Unit tests for image processing functions with focus on edge cases."""

import unittest
import numpy as np
import cv2
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.helixzone.core.utils import (
    safe_gaussian_blur,
    safe_canny,
    safe_resize,
    safe_morphology,
    safe_threshold
)
from src.helixzone.core.type_defs import (
    Mat, ImageFloat, ImageUInt8, ImageBool,
    ensure_float32, ensure_uint8
)

class TestSafeImageOperations(unittest.TestCase):
    """Test safe wrapper functions for OpenCV operations."""
    
    def setUp(self):
        """Set up test case."""
        # Create test images of different types
        self.img_uint8 = np.ones((100, 100), dtype=np.uint8) * 128
        self.img_float32 = np.ones((100, 100), dtype=np.float32) * 0.5
        self.img_empty = np.array([], dtype=np.uint8)
        self.img_tiny = np.ones((1, 1), dtype=np.uint8)
        self.img_color = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
    def test_safe_gaussian_blur_valid(self):
        """Test safe_gaussian_blur with valid inputs."""
        result = safe_gaussian_blur(self.img_uint8, (5, 5), 1.0)
        self.assertEqual(result.shape, self.img_uint8.shape)
        self.assertEqual(result.dtype, self.img_uint8.dtype)
        
        result_float = safe_gaussian_blur(self.img_float32, (5, 5), 1.0)
        self.assertEqual(result_float.shape, self.img_float32.shape)
        self.assertEqual(result_float.dtype, self.img_float32.dtype)
        
    def test_safe_gaussian_blur_edge_cases(self):
        """Test safe_gaussian_blur with edge cases."""
        # Test with empty array
        result = safe_gaussian_blur(self.img_empty, (5, 5), 1.0)
        self.assertEqual(result.size, 0)
        
        # Test with tiny image
        result = safe_gaussian_blur(self.img_tiny, (5, 5), 1.0)
        self.assertEqual(result.shape, (1, 1))
        
        # Test with even kernel size (should be converted to odd)
        result = safe_gaussian_blur(self.img_uint8, (4, 4), 1.0)
        self.assertEqual(result.shape, self.img_uint8.shape)
        
        # Test with zero sigma
        result = safe_gaussian_blur(self.img_uint8, (5, 5), 0.0)
        self.assertEqual(result.shape, self.img_uint8.shape)
        
    def test_safe_canny_valid(self):
        """Test safe_canny with valid inputs."""
        result = safe_canny(self.img_uint8, 100, 200)
        self.assertEqual(result.shape, self.img_uint8.shape)
        self.assertEqual(result.dtype, np.uint8)
        
    def test_safe_canny_edge_cases(self):
        """Test safe_canny with edge cases."""
        # Test with empty array
        result = safe_canny(self.img_empty, 100, 200)
        self.assertEqual(result.shape, (1, 1))
        
        # Test with tiny image
        result = safe_canny(self.img_tiny, 100, 200)
        self.assertEqual(result.shape, (1, 1))
        
        # Test with color image
        result = safe_canny(self.img_color, 100, 200)
        self.assertEqual(result.shape, (100, 100))
        
        # Test with float image
        result = safe_canny(self.img_float32, 0.4, 0.8)
        self.assertEqual(result.shape, self.img_float32.shape)
        
    def test_safe_resize_valid(self):
        """Test safe_resize with valid inputs."""
        result = safe_resize(self.img_uint8, (50, 50))
        self.assertEqual(result.shape, (50, 50))
        self.assertEqual(result.dtype, self.img_uint8.dtype)
        
    def test_safe_resize_edge_cases(self):
        """Test safe_resize with edge cases."""
        # Test with empty array
        result = safe_resize(self.img_empty, (50, 50))
        self.assertEqual(result.size, 0)
        
        # Test with invalid target size
        result = safe_resize(self.img_uint8, (0, 50))
        self.assertEqual(result.shape, self.img_uint8.shape)  # Should return original
        
    def test_safe_threshold_valid(self):
        """Test safe_threshold with valid inputs."""
        ret, result = safe_threshold(self.img_uint8, 100, 255)
        self.assertEqual(result.shape, self.img_uint8.shape)
        self.assertEqual(result.dtype, self.img_uint8.dtype)
        
    def test_safe_threshold_edge_cases(self):
        """Test safe_threshold with edge cases."""
        # Test with empty array
        ret, result = safe_threshold(self.img_empty, 100, 255)
        self.assertEqual(result.size, 0)
        
        # Test with float image
        ret, result = safe_threshold(self.img_float32, 0.5, 1.0)
        self.assertEqual(result.shape, self.img_float32.shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

class TestTypeConversions(unittest.TestCase):
    """Test image type conversion functions."""
    
    def setUp(self):
        """Set up test case."""
        self.img_uint8 = np.ones((100, 100), dtype=np.uint8) * 128
        self.img_float32 = np.ones((100, 100), dtype=np.float32) * 0.5
        
    def test_ensure_float32(self):
        """Test ensure_float32 function."""
        # Convert uint8 to float32
        float_img = ensure_float32(self.img_uint8)
        self.assertEqual(float_img.dtype, np.float32)
        self.assertEqual(float_img.shape, self.img_uint8.shape)
        
        # Already float32
        float_img2 = ensure_float32(self.img_float32)
        self.assertEqual(float_img2.dtype, np.float32)
        self.assertEqual(float_img2.shape, self.img_float32.shape)
        
    def test_ensure_uint8(self):
        """Test ensure_uint8 function."""
        # Convert float32 to uint8
        uint8_img = ensure_uint8(self.img_float32)
        self.assertEqual(uint8_img.dtype, np.uint8)
        self.assertEqual(uint8_img.shape, self.img_float32.shape)
        
        # Already uint8
        uint8_img2 = ensure_uint8(self.img_uint8)
        self.assertEqual(uint8_img2.dtype, np.uint8)
        self.assertEqual(uint8_img2.shape, self.img_uint8.shape)
        
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        empty_uint8 = np.array([], dtype=np.uint8)
        empty_float32 = np.array([], dtype=np.float32)
        
        # These should not raise exceptions
        result1 = ensure_float32(empty_uint8)
        result2 = ensure_uint8(empty_float32)
        
        self.assertEqual(result1.size, 0)
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(result2.size, 0)
        self.assertEqual(result2.dtype, np.uint8)
        
if __name__ == '__main__':
    unittest.main()
