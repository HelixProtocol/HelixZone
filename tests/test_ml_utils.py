"""Tests for machine learning utilities."""

import numpy as np
import pytest
import cv2
from helixzone.core.ml_utils import (
    lasso_selection_performance,
    EnhancedLassoFeathering,
    compute_lbp,
    compute_gabor_features,
    create_feature_matrix
)
from PIL import Image
import re
import scipy.sparse
from sklearn.preprocessing import RobustScaler

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
    
    assert isinstance(features, list)
    assert len(features) == 16  # 4 features * 4 orientations
    assert all(isinstance(f, float) for f in features)
    assert all(np.isfinite(f) for f in features)

def test_feature_matrix_color_image():
    """Test feature matrix creation with color images."""
    # Create a color test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = [255, 128, 64]  # Add a colored rectangle
    
    # Add some gradients
    img[20:80, 20:80, 0] = np.linspace(0, 255, 60).reshape(60, 1)  # Red gradient
    img[20:80, 20:80, 1] = np.linspace(0, 255, 60).reshape(1, 60)  # Green gradient
    
    # Test points at different locations
    coords = [(50, 50), (30, 30), (70, 70)]
    
    # Test with different patch sizes
    for patch_size in [5, 7, 9]:
        features = create_feature_matrix(img, coords, patch_size)
        assert features.shape[0] == len(coords)
        assert features.shape[1] > 20  # Should have many features for color images

def test_feature_matrix_edge_cases():
    """Test feature matrix creation with edge cases."""
    # Create test image
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    
    # Test points near image boundaries
    coords = [
        (0, 0),      # Top-left corner
        (49, 49),    # Bottom-right corner
        (0, 49),     # Top-right corner
        (49, 0),     # Bottom-left corner
        (25, 25)     # Center
    ]
    
    features = create_feature_matrix(img, coords, patch_size=7)
    assert features.shape[0] == len(coords)
    assert not np.any(np.isnan(features))  # No NaN values
    assert not np.any(np.isinf(features))  # No infinite values

def test_lasso_feathering_color_handling():
    """Test lasso feathering with color images and invalid inputs."""
    feathering = EnhancedLassoFeathering()
    
    # Test with invalid image shapes
    with pytest.raises(ValueError):
        # Test with 4-channel image
        invalid_img = np.zeros((50, 50, 4), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        feathering.apply_lasso_feathering(invalid_img, mask)
    
    with pytest.raises(ValueError):
        # Test with incompatible mask shape
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        invalid_mask = np.zeros((60, 60), dtype=np.uint8)
        feathering.apply_lasso_feathering(img, invalid_mask)

def test_color_aware_feathering():
    """Test color-aware feathering functionality."""
    feathering = EnhancedLassoFeathering()
    
    # Create test image with color gradients
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 100).reshape(1, -1)  # Red gradient
    img[:, :, 1] = np.linspace(0, 255, 100).reshape(-1, 1)  # Green gradient
    img[:, :, 2] = 128  # Constant blue
    
    # Create circular mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, (255,), -1)
    
    # Test with different alpha values
    for alpha in [0.01, 0.05, 0.1]:
        result = feathering.apply_color_aware_feathering(img, mask, alpha)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)  # Should modify the image
        
    # Test with invalid inputs
    with pytest.raises(ValueError):
        # Test with grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        feathering.apply_color_aware_feathering(gray_img, mask)

def test_selection_mask_creation():
    """Test selection mask creation with various point configurations."""
    feathering = EnhancedLassoFeathering()
    
    # Test with insufficient points
    points = [(10, 10), (20, 20)]  # Less than 3 points
    mask = feathering.create_selection_mask(100, 100, points)
    assert np.all(mask == 0)  # Should return empty mask
    
    # Test with valid polygon
    points = [(10, 10), (20, 10), (15, 20)]  # Triangle
    mask = feathering.create_selection_mask(100, 100, points)
    assert np.any(mask > 0)  # Should contain some selected pixels
    
    # Test with complex polygon
    points = [(10, 10), (50, 10), (50, 50), (10, 50)]  # Square
    mask = feathering.create_selection_mask(100, 100, points)
    assert np.any(mask > 0)  # Should contain some selected pixels
    assert mask.shape == (100, 100)  # Should match specified dimensions 

def test_feature_matrix_validation():
    """Test feature matrix creation with invalid inputs."""
    # Test with non-numpy array
    class FakeArray:
        def __init__(self):
            self.shape = (10, 10)
    with pytest.raises(ValueError, match="Image must be a numpy array"):
        create_feature_matrix(FakeArray(), [(0, 0)])
    
    # Test with empty coordinates
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="Coordinates list cannot be empty"):
        create_feature_matrix(img, [])
    
    # Test with invalid patch size
    with pytest.raises(ValueError, match="Patch size must be odd and >= 3"):
        create_feature_matrix(img, [(0, 0)], patch_size=2)

def test_feature_matrix_texture():
    """Test texture feature extraction in feature matrix creation."""
    # Create test image with specific texture patterns
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Add different patterns to each channel
    img[::2, ::2, 0] = 255  # Checkerboard in red channel
    img[::3, ::3, 1] = 255  # Different pattern in green channel
    img[::4, ::4, 2] = 255  # Different pattern in blue channel
    
    coords = [(25, 25)]  # Center point
    features = create_feature_matrix(img, coords, patch_size=7)
    
    assert features.shape[0] == 1
    assert features.shape[1] > 30  # Should have many features including texture

def test_color_aware_feathering_validation():
    """Test color-aware feathering with invalid inputs."""
    feathering = EnhancedLassoFeathering()
    
    # Test with grayscale image
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match=re.escape("Image must be a color image (3 channels)")):
        feathering.apply_color_aware_feathering(img, mask)
    
    # Test with invalid mask shape
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    invalid_mask = np.zeros((60, 60), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image and mask must have compatible shapes"):
        feathering.apply_color_aware_feathering(img, invalid_mask)

def test_selection_mask_complex():
    """Test selection mask creation with complex shapes."""
    feathering = EnhancedLassoFeathering()
    
    # Create a star-shaped selection
    center = (50, 50)
    points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5
        # Outer point
        points.append((
            int(center[0] + 40 * np.cos(angle)),
            int(center[1] + 40 * np.sin(angle))
        ))
        # Inner point
        angle += np.pi / 5
        points.append((
            int(center[0] + 20 * np.cos(angle)),
            int(center[1] + 20 * np.sin(angle))
        ))
    
    mask = feathering.create_selection_mask(100, 100, points)
    assert mask.shape == (100, 100)
    assert np.any(mask > 0)  # Should have selected pixels
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8

def test_gabor_features_validation():
    """Test Gabor feature computation with various inputs."""
    # Test with minimum size patch
    min_patch = np.ones((3, 3), dtype=np.uint8)
    features = compute_gabor_features(min_patch)
    assert len(features) == 16  # 4 orientations * 4 features
    
    # Test with larger patch
    large_patch = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    features = compute_gabor_features(large_patch, num_orientations=6)
    assert len(features) == 24  # 6 orientations * 4 features

def test_lbp_edge_cases():
    """Test Local Binary Pattern computation with edge cases."""
    # Test with minimum size patch
    min_patch = np.ones((3, 3), dtype=np.uint8)
    lbp = compute_lbp(min_patch)
    assert lbp.shape == (3, 3)
    assert lbp.dtype == np.uint8
    
    # Test with color patch
    color_patch = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    lbp = compute_lbp(color_patch)
    assert lbp.shape == (5, 5)
    assert lbp.dtype == np.uint8 

def test_comprehensive_coverage():
    """Test to cover remaining edge cases and functionality."""
    # Test color-aware feathering with various inputs
    feathering = EnhancedLassoFeathering()
    
    # Create test image with specific patterns
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = [255, 128, 64]  # Colored region
    img[30:40, 30:40] = [64, 255, 128]  # Different colored region
    
    # Create mask with specific pattern
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 255  # Selection area
    
    # Test color-aware feathering
    result = feathering.apply_color_aware_feathering(img, mask, alpha=0.05)
    assert result.shape == img.shape
    assert not np.array_equal(result, img)
    
    # Test selection mask with complex shape
    points = [(25, 25), (35, 25), (35, 35), (25, 35)]  # Square
    mask = feathering.create_selection_mask(50, 50, points)
    assert mask.shape == (50, 50)
    assert np.any(mask > 0)
    
    # Test Gabor features with various inputs
    patch = np.random.randint(0, 255, (15, 15), dtype=np.uint8)
    features = compute_gabor_features(patch, num_orientations=8)
    assert len(features) == 32  # 8 orientations * 4 features
    
    # Test LBP with various patterns
    test_patterns = [
        np.ones((5, 5), dtype=np.uint8) * 128,  # Uniform pattern
        np.random.randint(0, 255, (7, 7), dtype=np.uint8),  # Random pattern
        np.zeros((3, 3, 3), dtype=np.uint8)  # Color pattern
    ]
    
    for pattern in test_patterns:
        lbp = compute_lbp(pattern)
        assert lbp.shape == pattern.shape[:2]
        assert lbp.dtype == np.uint8
    
    # Test feature matrix with texture patterns
    img_texture = np.zeros((30, 30, 3), dtype=np.uint8)
    # Create checkerboard pattern
    img_texture[::2, ::2] = [255, 0, 0]
    img_texture[1::2, 1::2] = [0, 255, 0]
    
    coords = [(15, 15)]  # Center point
    features = create_feature_matrix(img_texture, coords, patch_size=5)
    assert features.shape[0] == 1
    assert features.shape[1] > 40  # Should have many features including texture

def test_advanced_feature_extraction():
    """Test advanced feature extraction with various inputs."""
    # Test color channel handling
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    img[10:20, 10:20, 0] = 255  # Red square
    img[15:25, 15:25, 1] = 255  # Green square overlapping
    img[5:15, 5:15, 2] = 255   # Blue square overlapping
    
    coords = [(15, 15)]  # Point at intersection of all squares
    features = create_feature_matrix(img, coords, patch_size=7)
    assert features.shape[0] == 1
    assert features.shape[1] > 50  # Should have many features including color interactions
    
    # Test texture feature extraction
    img_texture = np.zeros((40, 40, 3), dtype=np.uint8)
    # Create complex texture pattern
    for i in range(3):  # For each channel
        pattern = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        img_texture[:, :, i] = pattern
    
    coords = [(20, 20)]  # Center point
    features = create_feature_matrix(img_texture, coords, patch_size=9)
    assert features.shape[0] == 1
    assert features.shape[1] > 55  # Should have many features including texture
    
    # Test Gabor feature computation with edge cases
    small_patch = np.ones((3, 3), dtype=np.uint8)
    features_small = compute_gabor_features(small_patch, num_orientations=4)
    assert len(features_small) == 16  # 4 orientations * 4 features
    
    large_patch = np.random.randint(0, 255, (25, 25), dtype=np.uint8)
    features_large = compute_gabor_features(large_patch, num_orientations=6)
    assert len(features_large) == 24  # 6 orientations * 4 features

def test_final_coverage():
    """Test to cover remaining edge cases and functionality."""
    # Test color channel handling with various inputs
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    img[10:20, 10:20] = [255, 128, 64]  # Add colored region
    
    # Test with different patch sizes
    coords = [(15, 15)]
    for patch_size in [5, 7, 9]:
        features = create_feature_matrix(img, coords, patch_size)
        assert features.shape[0] == 1
        assert features.shape[1] > 40  # Should have many features
    
    # Test color-aware feathering with various inputs
    feathering = EnhancedLassoFeathering()
    
    # Create test image with specific patterns
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = [255, 128, 64]  # Colored region
    img[30:40, 30:40] = [64, 255, 128]  # Different colored region
    
    # Create mask with specific pattern
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 255  # Selection area
    
    # Test color-aware feathering with different parameters
    result = feathering.apply_color_aware_feathering(img, mask, alpha=0.05)
    assert result.shape == img.shape
    assert not np.array_equal(result, img)
    
    # Test selection mask with complex shape
    points = [(25, 25), (35, 25), (35, 35), (25, 35)]  # Square
    mask = feathering.create_selection_mask(50, 50, points)
    assert mask.shape == (50, 50)
    assert np.any(mask > 0)
    
    # Test Gabor features with various inputs
    patch = np.random.randint(0, 255, (15, 15), dtype=np.uint8)
    features = compute_gabor_features(patch, num_orientations=8)
    assert len(features) == 32  # 8 orientations * 4 features
    
    # Test texture feature extraction
    img_texture = np.zeros((40, 40, 3), dtype=np.uint8)
    # Create complex texture pattern
    for i in range(3):  # For each channel
        pattern = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        img_texture[:, :, i] = pattern
    
    coords = [(20, 20)]  # Center point
    features = create_feature_matrix(img_texture, coords, patch_size=9)
    assert features.shape[0] == 1
    assert features.shape[1] > 55  # Should have many features including texture

def test_remaining_coverage():
    """Test to cover remaining uncovered lines and edge cases."""
    # Test color channel handling (line 94)
    img_4ch = np.zeros((30, 30, 4), dtype=np.uint8)  # 4-channel image
    with pytest.raises(ValueError, match="Image must be grayscale or BGR"):
        feathering = EnhancedLassoFeathering()
        feathering.apply_lasso_feathering(img_4ch, np.zeros((30, 30)))

    # Test texture feature extraction (lines 120-122)
    img_color = np.zeros((30, 30, 3), dtype=np.uint8)
    img_color[10:20, 10:20] = [255, 128, 64]  # Add colored region
    coords = [(15, 15)]
    features = create_feature_matrix(img_color, coords, patch_size=7)
    assert features.shape[1] > 40  # Should have texture features

    # Test color-aware feathering validation and implementation (lines 382, 390-400)
    feathering = EnhancedLassoFeathering()
    img_lab = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255
    
    # Test with different alpha values and iterations
    result = feathering.apply_color_aware_feathering(
        img_lab.astype(np.float32) / 255.0,
        mask,
        alpha=0.1  # Increased alpha for better convergence
    )
    assert result.shape == img_lab.shape
    assert not np.array_equal(result, img_lab)

    # Test selection mask validation (line 422)
    points = [(10, 10), (20, 10)]  # Less than 3 points
    mask = feathering.create_selection_mask(30, 30, points)
    assert np.all(mask == 0)  # Should return empty mask

    # Test Gabor feature computation (line 450)
    patch = np.random.randint(0, 255, (15, 15), dtype=np.uint8)
    features = compute_gabor_features(patch, num_orientations=6)
    assert len(features) == 24  # 6 orientations * 4 features
    assert all(isinstance(f, float) for f in features)  # All features should be Python floats 

def test_texture_feature_extraction():
    """Test texture feature extraction with various edge cases."""
    # Test with small patches and color channels
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    # Create different patterns in each channel
    img[::2, ::2, 0] = 255  # Checkerboard in red
    img[::3, ::3, 1] = 255  # Different pattern in green
    img[:10, :10, 2] = 255  # Solid area in blue
    
    # Test points near edges and corners
    coords = [
        (1, 1),    # Near corner
        (10, 1),   # Near edge
        (10, 10),  # Center
        (18, 18)   # Near opposite corner
    ]
    
    # Test with minimum patch size
    features_min = create_feature_matrix(img, coords, patch_size=3)
    assert features_min.shape[0] == len(coords)
    assert features_min.shape[1] > 30  # Should include texture features
    
    # Test with larger patch size
    features_large = create_feature_matrix(img, coords, patch_size=7)
    assert features_large.shape[0] == len(coords)
    assert features_large.shape[1] > 30
    
    # Verify no NaN or infinite values
    assert not np.any(np.isnan(features_min))
    assert not np.any(np.isinf(features_min))
    assert not np.any(np.isnan(features_large))
    assert not np.any(np.isinf(features_large))

def test_edge_case_handling():
    """Test edge case handling in feature extraction and processing."""
    feathering = EnhancedLassoFeathering()
    
    # Test with 1-pixel wide image
    narrow_img = np.random.rand(20, 1, 3).astype(np.float32)
    narrow_mask = np.ones((20, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image must be at least 3x3 pixels"):
        feathering.apply_lasso_feathering(narrow_img, narrow_mask)
    
    # Test with 1-pixel high image
    short_img = np.random.rand(1, 20, 3).astype(np.float32)
    short_mask = np.ones((1, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image must be at least 3x3 pixels"):
        feathering.apply_lasso_feathering(short_img, short_mask)
    
    # Test with single-pixel image
    single_img = np.random.rand(1, 1, 3).astype(np.float32)
    single_mask = np.ones((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image must be at least 3x3 pixels"):
        feathering.apply_lasso_feathering(single_img, single_mask)

def test_feature_extraction_validation_comprehensive():
    """Test feature extraction validation and edge cases."""
    # Test with invalid coordinates
    img = np.random.rand(30, 30).astype(np.float32)
    invalid_coords = [(31, 31)]  # Outside image bounds
    with pytest.raises(ValueError, match="Coordinates outside image bounds"):
        create_feature_matrix(img, invalid_coords)
    
    # Test with minimum size image
    min_img = np.random.rand(3, 3).astype(np.float32)
    min_coords = [(1, 1)]
    features = create_feature_matrix(min_img, min_coords, patch_size=3)
    assert features.shape[0] == 1
    assert features.shape[1] > 10  # Should still extract basic features
    
    # Test with too small image
    tiny_img = np.random.rand(2, 2).astype(np.float32)
    with pytest.raises(ValueError, match="Image must be at least 3x3 pixels"):
        create_feature_matrix(tiny_img, [(0, 0)])
    
    # Test with invalid patch sizes
    with pytest.raises(ValueError, match="Patch size must be odd and >= 3"):
        create_feature_matrix(img, [(15, 15)], patch_size=2)  # Even size
    with pytest.raises(ValueError, match="Patch size must be odd and >= 3"):
        create_feature_matrix(img, [(15, 15)], patch_size=1)  # Too small

def test_gabor_feature_validation_extended():
    """Test Gabor feature computation with various edge cases."""
    # Test with invalid orientations
    patch = np.random.rand(10, 10).astype(np.float32)
    with pytest.raises(ValueError, match="Number of orientations must be positive"):
        compute_gabor_features(patch, num_orientations=0)
    
    # Test with different numbers of orientations
    for num_orientations in [2, 4, 6, 8]:
        features = compute_gabor_features(patch, num_orientations=num_orientations)
        assert len(features) == num_orientations * 4  # 4 features per orientation
        assert all(isinstance(f, float) for f in features)
        assert all(np.isfinite(f) for f in features)
    
    # Test with color patch
    color_patch = np.random.rand(10, 10, 3).astype(np.float32)
    features = compute_gabor_features(color_patch)
    assert len(features) == 16  # Default 4 orientations * 4 features
    
    # Test with normalized vs unnormalized patches
    patch_norm = patch / 255.0  # [0, 1] range
    patch_unnorm = (patch * 255).astype(np.uint8)  # [0, 255] range
    features_norm = compute_gabor_features(patch_norm)
    features_unnorm = compute_gabor_features(patch_unnorm)
    assert len(features_norm) == len(features_unnorm)

def test_selection_mask_validation_extended():
    """Test selection mask creation with edge cases."""
    feathering = EnhancedLassoFeathering()
    
    # Test with invalid dimensions
    with pytest.raises(ValueError, match="Width and height must be positive"):
        feathering.create_selection_mask(0, 50, [(10, 10), (20, 20), (15, 30)])
    with pytest.raises(ValueError, match="Width and height must be positive"):
        feathering.create_selection_mask(50, -1, [(10, 10), (20, 20), (15, 30)])
    
    # Test with various point configurations
    points_list = [
        [],  # Empty list
        [(10, 10)],  # Single point
        [(10, 10), (20, 20)],  # Two points
        [(10, 10), (20, 10), (15, 20)],  # Triangle
        [(10, 10), (20, 10), (20, 20), (10, 20)]  # Square
    ]
    
    for points in points_list:
        mask = feathering.create_selection_mask(50, 50, points)
        assert mask.shape == (50, 50)
        assert mask.dtype == np.uint8
        if len(points) < 3:
            assert np.all(mask == 0)  # Should be empty
        else:
            assert np.any(mask > 0)  # Should have some selected pixels 

def test_final_edge_cases():
    """Test remaining edge cases for complete coverage."""
    feathering = EnhancedLassoFeathering()
    
    # Test feature matrix creation edge cases
    img = np.zeros((20, 20, 3), dtype=np.float32)
    coords = [(5, 5), (15, 15)]
    
    # Test with sparse matrix conversion
    X = scipy.sparse.csr_matrix(np.random.rand(10, 5))
    X_dense = X.toarray() if scipy.sparse.issparse(X) else X
    assert isinstance(X_dense, np.ndarray)
    
    # Test feature scaling with outliers
    scaler = RobustScaler(quantile_range=(1, 99))
    data = np.random.rand(100, 5)
    data[0] = 1000  # Add outlier
    scaled = scaler.fit_transform(data)
    assert not np.any(np.abs(scaled) > 100)  # Outliers should be scaled down
    
    # Test Gabor feature computation with various inputs
    patch = np.random.rand(10, 10).astype(np.float32)
    for num_orientations in [3, 5, 7]:  # Test odd numbers of orientations
        features = compute_gabor_features(patch, num_orientations=num_orientations)
        assert len(features) == num_orientations * 4
        assert all(isinstance(f, float) for f in features)
    
    # Test selection mask with complex shapes
    points = [
        (10, 10), (20, 10), (20, 20), (15, 25), (10, 20)  # Pentagon
    ]
    mask = feathering.create_selection_mask(30, 30, points)
    assert mask.shape == (30, 30)
    assert np.any(mask > 0)
    
    # Test color processing with extreme patterns
    img_color = np.zeros((30, 30, 3), dtype=np.float32)
    # Create high contrast patterns
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, 30), np.linspace(0, 4*np.pi, 30))
    img_color[..., 0] = np.sin(x)
    img_color[..., 1] = np.cos(y)
    img_color[..., 2] = np.sin(x + y)
    
    mask_color = np.zeros((30, 30), dtype=np.uint8)
    mask_color[5:25, 5:25] = 255
    
    result = feathering.apply_color_aware_feathering(img_color, mask_color, alpha=0.1)
    assert result.shape == img_color.shape
    assert not np.array_equal(result, img_color)
    
    # Test with various edge strengths
    edge_strength = feathering.compute_edge_strength(img_color)
    assert edge_strength.shape == img_color.shape[:2]
    assert np.all(edge_strength >= 0) and np.all(edge_strength <= 1) 