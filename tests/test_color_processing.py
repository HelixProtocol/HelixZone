"""
Tests for color processing edge cases and error handling.
"""

import pytest
import numpy as np
import cv2
from helixzone.core.ml_utils import EnhancedLassoFeathering

class TestColorProcessingEdgeCases:
    @pytest.fixture
    def feathering(self):
        return EnhancedLassoFeathering()
        
    @pytest.fixture
    def test_images(self):
        """Generate test images with challenging patterns."""
        images = {}
        
        # High contrast edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        images["high_contrast"] = img
        
        # Color gradients
        x = np.linspace(0, 255, 100)
        y = np.linspace(0, 255, 100)
        X, Y = np.meshgrid(x, y)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[..., 0] = X  # Blue gradient
        img[..., 1] = Y  # Green gradient
        img[..., 2] = (X + Y) / 2  # Red gradient
        images["gradients"] = img
        
        # Noise pattern
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        images["noise"] = img
        
        # Single color
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        images["solid"] = img
        
        # Checkerboard
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        images["checkerboard"] = img
        
        return images
        
    @pytest.fixture
    def test_masks(self):
        """Generate test masks with various patterns."""
        masks = {}
        
        # Center circle
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, (255,), -1)
        masks["circle"] = mask
        
        # Diagonal line
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (0, 0), (99, 99), (255,), 5)
        masks["line"] = mask
        
        # Complex shape
        mask = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array([[25, 25], [75, 25], [75, 75], [25, 75]], np.int32)
        cv2.fillPoly(mask, [pts], (255,))
        masks["polygon"] = mask
        
        return masks
        
    def test_high_contrast_edges(self, feathering, test_images, test_masks):
        """Test color processing with high contrast edges."""
        result = feathering.apply_color_aware_feathering(
            test_images["high_contrast"],
            test_masks["line"]
        )
        
        # Check smoothness across edge
        edge_region = result[45:55, 45:55]
        gradient = np.abs(np.diff(edge_region, axis=1))
        assert np.max(gradient) < 0.5, "Color transition should be smooth"
        
    def test_color_gradients(self, feathering, test_images, test_masks):
        """Test color processing with smooth gradients."""
        result = feathering.apply_color_aware_feathering(
            test_images["gradients"],
            test_masks["circle"]
        )
        
        # Check gradient preservation
        center = result[40:60, 40:60]
        grad_x = np.abs(np.diff(center, axis=1))
        grad_y = np.abs(np.diff(center, axis=0))
        assert np.mean(grad_x) > 0.01, "Gradients should be preserved"
        assert np.mean(grad_y) > 0.01, "Gradients should be preserved"
        
    def test_noise_patterns(self, feathering, test_images, test_masks):
        """Test color processing with noise patterns."""
        result = feathering.apply_color_aware_feathering(
            test_images["noise"],
            test_masks["polygon"]
        )
        
        # Check noise reduction
        mask_region = test_masks["polygon"] > 0
        original_std = np.std(test_images["noise"][mask_region])
        result_std = np.std(result[mask_region])
        assert result_std < original_std, "Noise should be reduced"
        
    def test_solid_colors(self, feathering, test_images, test_masks):
        """Test color processing with solid colors."""
        result = feathering.apply_color_aware_feathering(
            test_images["solid"],
            test_masks["circle"]
        )
        
        # Check color preservation
        mask_region = test_masks["circle"] > 0
        mean_value = np.mean(result[mask_region])
        assert abs(mean_value - 128.0) < 5.0, "Solid color should be preserved"
        
    def test_checkerboard_pattern(self, feathering, test_images, test_masks):
        """Test color processing with checkerboard pattern."""
        result = feathering.apply_color_aware_feathering(
            test_images["checkerboard"],
            test_masks["line"]
        )
        
        # Check pattern preservation
        line_region = test_masks["line"] > 0
        pattern_diff = np.abs(
            test_images["checkerboard"][line_region].astype(float) / 255 -
            result[line_region]
        )
        assert np.mean(pattern_diff) < 0.3, "Pattern structure should be preserved"
        
    def test_invalid_inputs(self, feathering):
        """Test error handling for invalid inputs."""
        # Wrong number of channels
        with pytest.raises(ValueError):
            img = np.zeros((100, 100, 4), dtype=np.uint8)  # 4 channels
            mask = np.zeros((100, 100), dtype=np.uint8)
            feathering.apply_color_aware_feathering(img, mask)
            
        # Incompatible shapes
        with pytest.raises(ValueError):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            mask = np.zeros((50, 50), dtype=np.uint8)  # Wrong size
            feathering.apply_color_aware_feathering(img, mask)
            
        # Invalid mask values
        with pytest.raises(ValueError):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            mask = np.ones((100, 100), dtype=np.uint8) * 128  # Invalid values
            feathering.apply_color_aware_feathering(img, mask)
            
    def test_extreme_parameters(self, feathering, test_images, test_masks):
        """Test color processing with extreme parameter values."""
        # Very small alpha
        result_small = feathering.apply_color_aware_feathering(
            test_images["gradients"],
            test_masks["circle"],
            alpha=0.0001
        )
        
        # Very large alpha
        result_large = feathering.apply_color_aware_feathering(
            test_images["gradients"],
            test_masks["circle"],
            alpha=10.0
        )
        
        # Check that results are different
        assert not np.allclose(result_small, result_large)
        
        # Extreme channel weights
        result_l = feathering.apply_color_aware_feathering(
            test_images["gradients"],
            test_masks["circle"],
            l_weight=10.0,
            ab_weight=0.1
        )
        
        result_ab = feathering.apply_color_aware_feathering(
            test_images["gradients"],
            test_masks["circle"],
            l_weight=0.1,
            ab_weight=10.0
        )
        
        # Check that results are different
        assert not np.allclose(result_l, result_ab)
        
    def test_memory_efficiency(self, feathering):
        """Test memory usage with large images."""
        import psutil
        process = psutil.Process()
        
        # Create large image
        img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        cv2.circle(mask, (1024, 1024), 512, (255,), -1)
        
        # Measure memory before
        mem_before = process.memory_info().rss
        
        # Process image
        _ = feathering.apply_color_aware_feathering(img, mask)
        
        # Measure memory after
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / (1024 * 1024)  # MB
        
        # Should use less than 10x image size
        img_size = img.nbytes / (1024 * 1024)  # MB
        assert mem_used < img_size * 10, "Memory usage should be reasonable"
        
    def test_gpu_fallback(self, feathering, test_images, test_masks):
        """Test GPU acceleration fallback."""
        # Try GPU processing
        try:
            feathering.enable_gpu()
            result_gpu = feathering.apply_color_aware_feathering(
                test_images["gradients"],
                test_masks["circle"]
            )
        except (AttributeError, RuntimeError):
            # Should fall back to CPU gracefully
            result_cpu = feathering.apply_color_aware_feathering(
                test_images["gradients"],
                test_masks["circle"]
            )
            assert result_cpu is not None, "CPU fallback should work" 