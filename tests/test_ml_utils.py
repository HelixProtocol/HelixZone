"""Tests for machine learning utilities."""

import numpy as np
import pytest
import cv2
from helixzone.core.ml_utils import (
    lasso_selection_performance,
    EnhancedLassoFeathering,
    compute_lbp,
    compute_gabor_features
)

def test_lasso_selection_performance():
    """Test the Lasso regression performance evaluation function."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # Create sparse coefficients
    true_coef = np.zeros(n_features)
    true_coef[0:5] = [1.0, -2.0, 3.0, -4.0, 5.0]
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with noise
    y = np.dot(X, true_coef) + np.random.randn(n_samples) * 0.1
    
    # Test with default alpha range
    results = lasso_selection_performance(X, y)
    
    # Check results structure
    assert isinstance(results, dict)
    assert all(key in results for key in ['alpha', 'mse', 'r2', 'n_features'])
    assert all(len(results[key]) == 50 for key in results)  # Default 50 alpha values
    
    # Check data types
    assert all(isinstance(alpha, float) for alpha in results['alpha'])
    assert all(isinstance(mse, float) for mse in results['mse'])
    assert all(isinstance(r2, float) for r2 in results['r2'])
    assert all(isinstance(n, int) for n in results['n_features'])
    
    # Check value ranges
    assert all(alpha > 0 for alpha in results['alpha'])  # Alpha should be positive
    assert all(mse >= 0 for mse in results['mse'])  # MSE should be non-negative
    assert all(-1 <= r2 <= 1 for r2 in results['r2'])  # R² should be between -1 and 1
    assert all(0 <= n <= n_features for n in results['n_features'])  # Number of features should be in valid range

def test_lasso_selection_performance_custom_alpha():
    """Test Lasso performance evaluation with custom alpha range."""
    # Generate simple data
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    
    # Custom alpha range
    alpha_range = np.array([0.1, 1.0, 10.0])
    
    # Test with custom alpha range
    results = lasso_selection_performance(X, y, alpha_range)
    
    # Check number of results matches custom alpha range
    assert len(results['alpha']) == len(alpha_range)
    assert np.allclose(results['alpha'], alpha_range)

def test_lasso_selection_performance_edge_cases():
    """Test Lasso performance evaluation with edge cases."""
    # Test with minimal data
    X_min = np.random.randn(10, 2)
    y_min = np.random.randn(10)
    results_min = lasso_selection_performance(X_min, y_min)
    assert all(len(results_min[key]) == 50 for key in results_min)
    
    # Test with single feature
    X_single = np.random.randn(100, 1)
    y_single = np.random.randn(100)
    results_single = lasso_selection_performance(X_single, y_single)
    assert all(n <= 1 for n in results_single['n_features'])

def test_lasso_selection_performance_input_validation():
    """Test input validation for Lasso performance evaluation."""
    # Test with invalid shapes
    with pytest.raises(ValueError):
        X_invalid = np.random.randn(10, 5)
        y_invalid = np.random.randn(15)  # Mismatched length
        lasso_selection_performance(X_invalid, y_invalid)
    
    # Test with invalid alpha range
    with pytest.raises(ValueError):
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        alpha_range = np.array([-1, 0, 1])  # Invalid negative alpha
        lasso_selection_performance(X, y, alpha_range)

# New tests for EnhancedLassoFeathering
class TestEnhancedLassoFeathering:
    """Test suite for EnhancedLassoFeathering class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.feathering = EnhancedLassoFeathering()
        
        # Create a test image with clear smooth and complex regions
        self.complex_image = np.zeros((100, 100), dtype=np.float32)
        # Add smooth gradient in top-left
        x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        self.complex_image[:50, :50] = x * y
        # Add high-frequency pattern in bottom-right
        x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
        self.complex_image[50:, 50:] = np.sin(x) * np.cos(y)
        
        # Create a simple test mask
        self.test_mask = np.zeros((100, 100), dtype=np.float32)
        self.test_mask[25:75, 25:75] = 1.0
        
        # Create other test images
        self.test_image = np.random.rand(100, 100).astype(np.float32)
        self.edge_image = np.zeros((100, 100), dtype=np.float32)
        self.edge_image[40:60, 40:60] = 1.0
        self.color_image = np.random.rand(100, 100, 3).astype(np.float32)

    def test_apply_lasso_feathering_parameters(self):
        """Test the effect of different alpha values."""
        # Test with high and low alpha values
        result_high_alpha = self.feathering.apply_lasso_feathering(
            self.test_image,
            self.test_mask,
            alpha=1.0,
            content_aware=False  # Disable content-aware for clearer alpha effect
        )
        
        result_low_alpha = self.feathering.apply_lasso_feathering(
            self.test_image,
            self.test_mask,
            alpha=0.01,
            content_aware=False
        )
        
        # Higher alpha should create smoother transitions
        grad_high = np.gradient(result_high_alpha)
        grad_low = np.gradient(result_low_alpha)
        
        grad_high_mag = np.sqrt(grad_high[0]**2 + grad_high[1]**2)
        grad_low_mag = np.sqrt(grad_low[0]**2 + grad_low[1]**2)
        
        # Higher alpha should result in lower gradients
        assert np.mean(grad_high_mag) < np.mean(grad_low_mag)

    def test_apply_lasso_feathering_content_aware(self):
        """Test edge preservation in content-aware mode."""
        # Create a test image with a strong edge
        edge_image = np.zeros((100, 100), dtype=np.float32)
        edge_image[40:60, :] = 1.0  # Horizontal edge
        
        # Create a test mask that crosses the edge
        test_mask = np.zeros((100, 100), dtype=np.float32)
        test_mask[30:70, 30:70] = 1.0
        
        # Test with and without content-aware mode
        result_content_aware = self.feathering.apply_lasso_feathering(
            edge_image,
            test_mask,
            content_aware=True,
            adaptive_width=False  # Disable adaptive width to focus on content awareness
        )

        result_basic = self.feathering.apply_lasso_feathering(
            edge_image,
            test_mask,
            content_aware=False,
            adaptive_width=False
        )

        # Create transition mask
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(test_mask.astype(np.uint8), kernel, iterations=2)
        eroded = cv2.erode(test_mask.astype(np.uint8), kernel, iterations=2)
        transition_mask = (dilated - eroded).astype(bool)

        # Calculate gradients in the transition region
        grad_content = np.gradient(result_content_aware)
        grad_basic = np.gradient(result_basic)

        grad_content_mag = np.sqrt(grad_content[0]**2 + grad_content[1]**2)
        grad_basic_mag = np.sqrt(grad_basic[0]**2 + grad_basic[1]**2)

        # Compare maximum gradients in transition region
        max_grad_content = np.max(grad_content_mag[transition_mask])
        max_grad_basic = np.max(grad_basic_mag[transition_mask])

        # Content-aware version should preserve edges better
        assert max_grad_content > max_grad_basic

    def test_apply_lasso_feathering_adaptive_width(self):
        """Test adaptive width behavior."""
        # Create a test image with clear smooth and complex regions
        test_image = np.zeros((100, 100), dtype=np.float32)
        
        # Add very smooth gradient in top-left
        x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        test_image[:50, :50] = 0.5 * (x + y)  # Linear gradient
        
        # Add very high-frequency pattern in bottom-right
        x, y = np.meshgrid(np.linspace(0, 20, 50), np.linspace(0, 20, 50))
        test_image[50:, 50:] = 0.5 + 0.5 * np.sin(x) * np.cos(y)  # High-frequency pattern
        
        # Create a test mask that crosses both regions
        test_mask = np.zeros((100, 100), dtype=np.float32)
        test_mask[20:80, 20:80] = 1.0  # Larger mask to ensure it crosses both regions
        
        result = self.feathering.apply_lasso_feathering(
            test_image,
            test_mask,
            adaptive_width=True,
            content_aware=False  # Disable content-aware to focus on adaptive width
        )

        # Define transition regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(test_mask.astype(np.uint8), kernel, iterations=2)
        eroded = cv2.erode(test_mask.astype(np.uint8), kernel, iterations=2)
        transition_mask = (dilated - eroded).astype(bool)

        # Create edge strength map
        edge_strength = cv2.Canny((test_image * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        edge_strength = cv2.dilate(edge_strength, np.ones((3, 3), np.uint8))
        complex_mask = edge_strength > 0.5

        # Measure transition widths in different regions
        smooth_region = result[~complex_mask & transition_mask]  # Smooth transitions
        complex_region = result[complex_mask & transition_mask]  # Complex transitions

        # Compute gradient magnitude of transitions
        def measure_transition_width(region):
            if len(region) == 0:
                return 0.0
            # Compute gradient magnitude
            dx = np.gradient(region)
            # Return inverse of mean gradient magnitude (larger value = wider transition)
            mean_gradient = np.mean(np.abs(dx))
            return 1.0 / (mean_gradient + 1e-6)  # Add small epsilon to avoid division by zero

        smooth_width = measure_transition_width(smooth_region)
        complex_width = measure_transition_width(complex_region)

        # Ensure we have enough samples in both regions
        assert len(smooth_region) > 0 and len(complex_region) > 0
        # Smooth region should have relatively wider transitions (smaller gradients)
        assert smooth_width > complex_width * 1.5

    def test_apply_lasso_feathering_color_handling(self):
        """Test color image handling."""
        # Test color-aware feathering
        result_color = self.feathering.apply_color_aware_feathering(
            self.color_image,
            self.test_mask
        )
        
        # Test grayscale feathering
        gray_image = cv2.cvtColor(
            (self.color_image * 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        
        result_gray = self.feathering.apply_lasso_feathering(
            gray_image,
            self.test_mask
        )
        
        # Color version should preserve more detail
        assert result_color.ndim == 3  # Should be a color image
        assert result_gray.ndim == 2  # Should be grayscale
        
        # Convert color result to grayscale for comparison
        result_color_gray = cv2.cvtColor(
            (result_color * 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        
        # Color version should have different transitions than grayscale
        assert not np.allclose(result_color_gray, result_gray, rtol=0.1, atol=0.1)

    def test_edge_strength_computation(self):
        """Test edge strength computation."""
        edge_map = self.feathering.compute_edge_strength(self.edge_image)
        
        # Edge map should be normalized
        assert np.all(edge_map >= 0) and np.all(edge_map <= 1)
        
        # Should detect the horizontal edge
        edge_region = edge_map[44:56, :]  # Around the edge
        non_edge_region = edge_map[0:40, :]  # Away from the edge
        
        assert np.mean(edge_region) > np.mean(non_edge_region)

    def test_feature_creation(self):
        """Test advanced feature creation."""
        coords = [(50, 50), (45, 45)]  # Test coordinates
        features = self.feathering.create_advanced_features(self.test_image, coords)
        
        # Check feature matrix shape and properties
        assert features.shape[0] == len(coords)  # One row per coordinate
        assert features.shape[1] == 11  # Number of features per point
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))  # No NaN or inf values

    def test_input_validation(self):
        """Test input validation."""
        # Test invalid image dimensions
        invalid_image = np.random.rand(100, 100, 4)  # 4 channels
        with pytest.raises(ValueError):
            self.feathering.apply_lasso_feathering(invalid_image, self.test_mask)
        
        # Test mismatched shapes
        invalid_mask = np.zeros((50, 50))
        with pytest.raises(ValueError):
            self.feathering.apply_lasso_feathering(self.test_image, invalid_mask)
        
        # Test invalid image type for color-aware feathering
        with pytest.raises(ValueError):
            self.feathering.apply_color_aware_feathering(self.test_image, self.test_mask)

def test_compute_lbp():
    """Test Local Binary Pattern computation."""
    # Create test patch
    patch = np.random.rand(10, 10)
    lbp = compute_lbp(patch)
    
    assert lbp.shape == patch.shape
    assert np.all(lbp >= 0)
    assert np.all(lbp < 256)  # 8-bit LBP

def test_compute_gabor_features():
    """Test Gabor feature computation."""
    # Create test patch
    patch = np.random.rand(10, 10)
    features = compute_gabor_features(patch)
    
    assert features.ndim == 1
    assert len(features) == 16  # 4 features * 4 orientations
    assert np.all(np.isfinite(features)) 