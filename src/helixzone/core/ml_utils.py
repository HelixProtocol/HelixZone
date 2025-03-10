"""Machine learning utilities for HelixZone."""

import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import cv2
from typing import Dict, List, Union, Optional, Tuple, Any, TypeVar, cast
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
import scipy.sparse

# Type aliases for better readability
ImageType = TypeVar('ImageType', bound=np.ndarray)
MaskType = TypeVar('MaskType', bound=np.ndarray)
FeatureMatrix = np.ndarray
Coordinates = List[Tuple[int, int]]

def lasso_selection_performance(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: Optional[np.ndarray] = None
) -> Dict[str, List[Union[float, int]]]:
    """Evaluate Lasso regression performance across different alpha values.

    This function performs Lasso regression with different regularization strengths
    and evaluates the model's performance using MSE and R² metrics.

    Args:
        X: Training data of shape (n_samples, n_features)
        y: Target values of shape (n_samples,)
        alpha_range: Range of alpha values to test. Defaults to np.logspace(-4, 1, 50)

    Returns:
        Dictionary containing performance metrics:
            - 'alpha': List of alpha values tested
            - 'mse': Mean squared error for each alpha
            - 'r2': R² score for each alpha
            - 'n_features': Number of non-zero features for each alpha
            
    Raises:
        ValueError: If X and y have incompatible shapes
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
        
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Store results
    results = {
        'alpha': [],
        'mse': [],
        'r2': [],
        'n_features': []
    }

    # Test different alpha values
    for alpha in alpha_range:
        # Create and fit Lasso model
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = lasso.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_features = np.sum(lasso.coef_ != 0)

        # Store results
        results['alpha'].append(float(alpha))
        results['mse'].append(float(mse))
        results['r2'].append(float(r2))
        results['n_features'].append(int(n_features))

    return results

def create_feature_matrix(img: Any, coords: List[Tuple[int, int]], patch_size: int = 5) -> np.ndarray:
    """Create feature matrix from image patches.
    
    Args:
        img: Input image (must be a numpy array)
        coords: List of (y, x) coordinates
        patch_size: Size of patch to extract features from (must be odd and >= 3)
        
    Returns:
        Feature matrix of shape (len(coords), n_features)
        
    Raises:
        ValueError: If img is not a numpy array, coords is empty, or patch_size is invalid
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if not coords:
        raise ValueError("Coordinates list cannot be empty")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("Patch size must be odd and >= 3")
    if img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError("Image must be at least 3x3 pixels")
    
    # Validate coordinates are within image bounds
    for y, x in coords:
        if y < 0 or y >= img.shape[0] or x < 0 or x >= img.shape[1]:
            raise ValueError("Coordinates outside image bounds")
    
    features = []
    half_size = patch_size // 2
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    for y, x in coords:
        # Extract local patch with padding
        y_start = max(0, y - half_size)
        y_end = min(img.shape[0], y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(img.shape[1], x + half_size + 1)
        
        # Basic features
        feature_list = []
        
        # Position features (always included)
        feature_list.extend([
            float(x/img.shape[1]),
            float(y/img.shape[0]),
            float((x - img.shape[1]/2)/(img.shape[1]/2)),  # Normalized distance from center
            float((y - img.shape[0]/2)/(img.shape[0]/2))   # Normalized distance from center
        ])
        
        # Extract patches
        gray_patch = img_gray[y_start:y_end, x_start:x_end]
        
        # Ensure patch has minimum size for feature extraction
        if gray_patch.shape[0] < 3 or gray_patch.shape[1] < 3:
            # Pad patch to minimum size using constant padding
            pad_height = max(0, 3 - gray_patch.shape[0])
            pad_width = max(0, 3 - gray_patch.shape[1])
            gray_patch = np.pad(gray_patch, ((0, pad_height), (0, pad_width)), mode='constant')
        
        # Gradient features (always included)
        grad_y, grad_x = np.gradient(gray_patch)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        feature_list.extend([
            float(np.mean(grad_mag)),
            float(np.std(grad_mag)),
            float(np.mean(np.abs(grad_dir))),
            float(np.std(grad_dir))
        ])
        
        # Color features if available
        if len(img.shape) == 3:
            color_patch = img[y_start:y_end, x_start:x_end]
            for channel in range(img.shape[2]):
                channel_patch = color_patch[:, :, channel]
                # More detailed color statistics
                feature_list.extend([
                    float(np.mean(np.asarray(channel_patch))),
                    float(np.std(np.asarray(channel_patch))),
                    float(np.median(np.asarray(channel_patch))),
                    float(np.percentile(np.asarray(channel_patch), 25)),
                    float(np.percentile(np.asarray(channel_patch), 75)),
                    float(np.max(np.asarray(channel_patch)) - np.min(np.asarray(channel_patch)))  # Color range
                ])
                
                # Color gradients
                color_grad_y, color_grad_x = np.gradient(channel_patch)
                color_grad_mag = np.sqrt(color_grad_x**2 + color_grad_y**2)
                feature_list.extend([
                    float(np.mean(np.asarray(color_grad_mag))),
                    float(np.std(np.asarray(color_grad_mag)))
                ])
        
        # Texture features
        if len(img.shape) == 3:
            # Color-based texture
            for i in range(img.shape[2]):
                for j in range(i+1, img.shape[2]):
                    # Color channel differences
                    diff = color_patch[:, :, i] - color_patch[:, :, j]
                    feature_list.extend([
                        float(np.mean(np.abs(np.asarray(diff)))),
                        float(np.std(np.asarray(diff)))
                    ])
        
        # Add local binary pattern features
        lbp = compute_lbp(gray_patch)
        feature_list.extend([
            float(np.mean(np.asarray(lbp))),
            float(np.std(np.asarray(lbp))),
            float(np.percentile(np.asarray(lbp), 25)),
            float(np.percentile(np.asarray(lbp), 75))
        ])
        
        # Add Gabor features
        gabor_features = compute_gabor_features(gray_patch)
        feature_list.extend(gabor_features)
        
        features.append(feature_list)
    
    return np.array(features, dtype=np.float32)

class EnhancedLassoFeathering:
    """Enhanced Lasso-based feathering with advanced features."""
    
    def __init__(self):
        """Initialize the feathering object."""
        self.feature_scale = 1.0
        self.scaler = StandardScaler()
        self.debug_mode = False
        self.mask_value = 255  # Maximum value for mask intensity

    def compute_edge_strength(self, image: np.ndarray) -> np.ndarray:
        """Compute edge strength map.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Edge strength map as float32 array
            
        Raises:
            ValueError: If image is not a numpy array
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
            
        # Convert to uint8 for edge detection
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(img_uint8, 50, 150)
        edges_coarse = cv2.Canny(cv2.GaussianBlur(img_uint8, (5, 5), 0), 30, 90)
        
        # Combine edges
        edges = cv2.addWeighted(edges_fine.astype(np.float32), 0.7,
                               edges_coarse.astype(np.float32), 0.3, 0)
        
        # Normalize and smooth
        edges = edges / 255.0
        edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0)
        
        return edges

    def create_advanced_features(
        self,
        image: np.ndarray,
        coords: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Enhanced feature creation with sophisticated image characteristics.
        
        Args:
            image: Input image
            coords: List of (y, x) coordinates
            
        Returns:
            Feature matrix of shape (len(coords), n_features)
            
        Raises:
            ValueError: If image is not a numpy array or coords is empty
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if not coords:
            raise ValueError("Coordinates list cannot be empty")

        # Convert to float and normalize
        img_float = image.astype(float) / 255.0

        # Pre-compute edge information
        edge_map = self.compute_edge_strength(image)

        features = []
        for y, x in coords:
            # Define patch size based on local complexity
            patch_size = 5
            half_size = patch_size // 2

            # Extract patches safely with padding
            y_start = max(0, y - half_size)
            y_end = min(image.shape[0], y + half_size + 1)
            x_start = max(0, x - half_size)
            x_end = min(image.shape[1], x + half_size + 1)

            patch = img_float[y_start:y_end, x_start:x_end]
            edge_patch = edge_map[y_start:y_end, x_start:x_end]

            # Enhanced feature set
            feature_vector = [
                float(patch.mean()),                    # Mean intensity
                float(patch.std()),                     # Local contrast
                float(sobel(patch, axis=0).mean()),     # Gradient X
                float(sobel(patch, axis=1).mean()),     # Gradient Y
                float(edge_patch.mean()),               # Edge strength
                float(edge_patch.std()),                # Edge variation
                float(np.percentile(patch, 25)),        # Lower quartile
                float(np.percentile(patch, 75)),        # Upper quartile
                float(x / image.shape[1]),              # Normalized X
                float(y / image.shape[0]),              # Normalized Y
                float(np.sqrt((x/image.shape[1])**2 + (y/image.shape[0])**2))  # Radial distance
            ]

            features.append(feature_vector)

        return np.array(features, dtype=np.float32) * self.feature_scale

    def _apply_single_channel(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.01,
        content_aware: bool = False
    ) -> np.ndarray:
        """Apply feathering to a single channel using optimized ElasticNet regression.
        
        This method uses a sophisticated optimization strategy to ensure robust convergence:
        1. Features and targets are scaled using RobustScaler with quantile range (1, 99)
            to handle outliers while preserving important variations.
        2. A two-stage fitting process is used:
           - First fit with higher regularization (5x alpha) to get stable coefficients
           - Then refine with desired alpha, using the pre-fitted coefficients
        3. Sample weights reduce the impact of outliers (points > 2 std dev get 0.5 weight)
        4. In content-aware mode:
           - Alpha varies smoothly from 0.05x to 3.0x based on edge strength
           - Edge features are weighted at 1.2x to preserve details without overfitting
           - Final smoothing uses a 0.6/0.4 balance for stability
        
        The ElasticNet parameters are carefully tuned for convergence:
        - max_iter=10000 and tol=1e-5 balance accuracy with performance
        - l1_ratio=0.6 (content-aware) or 0.5 (basic) provides good sparsity
        - cyclic coordinate descent is more stable than random
        - warm_start and fit_intercept improve convergence
        
        Note: Some extreme cases (e.g., high-frequency color patterns) may still show
        convergence warnings, but the duality gaps remain proportional to tolerances,
        ensuring numerically stable results.

        Args:
            image: Input image channel as numpy array
            mask: Binary mask as numpy array
            alpha: Regularization strength (default: 0.01)
            content_aware: Whether to use content-aware feathering
            
        Returns:
            Feathered image channel
            
        Raises:
            ValueError: If inputs have incompatible shapes or invalid types
        """
        # Input validation
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays")
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have compatible shapes")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number")

        # Create feature matrix
        mask_coords = np.argwhere(mask > 0)
        coords = [(int(y), int(x)) for y, x in mask_coords]
        X = self.create_advanced_features(image, coords)
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X).toarray()
        X = np.asarray(X, dtype=np.float64)
        y = image[mask > 0].reshape(-1, 1)

        # Scale features with more robust quantile range and centering
        feature_scaler = RobustScaler(quantile_range=(1, 99), unit_variance=True, with_centering=True)
        X_scaled = feature_scaler.fit_transform(X)
        target_scaler = RobustScaler(quantile_range=(1, 99), unit_variance=True, with_centering=True)
        y_scaled = target_scaler.fit_transform(y)

        # Compute edge strength if in content-aware mode
        if content_aware:
            edge_strength = cv2.Sobel(image, cv2.CV_32F, 1, 1)
            edge_strength = np.abs(edge_strength)
            max_edge = np.max(edge_strength)
            if max_edge > 0:
                edge_strength = edge_strength / max_edge
            else:
                edge_strength = np.zeros_like(edge_strength)
            edge_mask = edge_strength[mask > 0]
            
            # Adjust alpha based on edge strength with smoother transitions
            base_alpha = alpha
            alpha_strong = base_alpha * 0.05  # Slightly increased minimum alpha
            alpha_weak = base_alpha * 3.0     # Reduced maximum alpha
            alpha_values = alpha_strong * edge_mask + alpha_weak * (1 - edge_mask)
            alpha_values = np.clip(alpha_values, base_alpha * 0.05, base_alpha * 3.0)
            
            # Add edge-preserving features
            grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
            edge_features = np.asarray([
                grad_x[mask > 0],
                grad_y[mask > 0],
                np.sqrt(grad_x[mask > 0]**2 + grad_y[mask > 0]**2)
            ]).T.astype(np.float64)
            edge_scaler = RobustScaler(quantile_range=(1, 99), unit_variance=True, with_centering=True)
            edge_features_scaled = edge_scaler.fit_transform(edge_features)
            X_scaled_array = np.asarray(X_scaled)
            edge_features_scaled_array = np.asarray(edge_features_scaled)
            X_scaled = np.concatenate((X_scaled_array, edge_features_scaled_array * 1.2), axis=1)
            
            # Use ElasticNet with optimized parameters for convergence
            model = ElasticNet(
                alpha=float(np.mean(alpha_values)),
                l1_ratio=0.6,  # Further reduced L1 for better convergence
                max_iter=10000,  # Increased iterations for complex cases
                tol=1e-5,       # Further relaxed tolerance
                warm_start=True,
                selection='cyclic',
                random_state=42,
                fit_intercept=True,  # Enable intercept fitting
                positive=False       # Allow negative coefficients
            )
        else:
            # Non-content-aware mode with optimized parameters
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=0.5,
                max_iter=10000,  # Increased iterations
                tol=1e-5,       # Relaxed tolerance
                warm_start=True,
                selection='cyclic',
                random_state=42,
                fit_intercept=True,
                positive=False
            )

        # Fit model with sample weights to handle outliers
        sample_weights = np.ones(len(y_scaled))
        outliers = np.abs(y_scaled) > 2.0
        sample_weights[outliers.ravel()] = 0.5
        
        # Pre-fit with higher regularization to get good initial coefficients
        pre_model = ElasticNet(
            alpha=float(alpha) * 5.0,
            l1_ratio=0.9,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            selection='cyclic',
            random_state=42
        )
        pre_model.fit(X_scaled, y_scaled.ravel(), sample_weight=sample_weights)
        
        # Use pre-fitted coefficients as starting point
        model.coef_ = pre_model.coef_
        model.intercept_ = pre_model.intercept_
        model.fit(X_scaled, y_scaled.ravel(), sample_weight=sample_weights)
        
        # Create prediction matrix
        full_coords = [(int(y), int(x)) for y, x in np.argwhere(np.ones_like(mask))]
        X_full = self.create_advanced_features(image, full_coords)
        if scipy.sparse.issparse(X_full):
            X_full = scipy.sparse.csr_matrix(X_full).toarray()
        X_full = np.asarray(X_full, dtype=np.float64)
        X_full_scaled = feature_scaler.transform(X_full)
        
        if content_aware:
            # Add edge features for full prediction
            grad_x_full = cv2.Sobel(image, cv2.CV_32F, 1, 0)
            grad_y_full = cv2.Sobel(image, cv2.CV_32F, 0, 1)
            edge_features_full = np.asarray([
                grad_x_full.ravel(),
                grad_y_full.ravel(),
                np.sqrt(grad_x_full.ravel()**2 + grad_y_full.ravel()**2)
            ]).T.astype(np.float64)
            edge_features_full_scaled = edge_scaler.transform(edge_features_full)
            X_full_scaled_array = np.asarray(X_full_scaled)
            edge_features_full_scaled_array = np.asarray(edge_features_full_scaled)
            X_full_scaled = np.concatenate((X_full_scaled_array, edge_features_full_scaled_array * 1.2), axis=1)

        y_pred_scaled = model.predict(X_full_scaled)
        y_pred_array = np.asarray(y_pred_scaled, dtype=np.float64).reshape(-1, 1)
        y_pred = target_scaler.inverse_transform(y_pred_array)
        
        # Reshape prediction to image size
        result = np.asarray(y_pred).reshape(image.shape)
        
        # Apply edge-preserving smoothing in content-aware mode with smoother transitions
        if content_aware:
            edge_weight = edge_strength * 0.6 + 0.4  # Even more balanced smoothing
            result = result * edge_weight + image * (1 - edge_weight)
        
        return result

    def apply_lasso_feathering(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.01,
        content_aware: bool = False,
        adaptive_width: bool = False
    ) -> np.ndarray:
        """Apply lasso feathering to an image.

        Args:
            image: Input image (grayscale or BGR) as numpy array
            mask: Binary mask as numpy array
            alpha: Regularization strength (higher values = smoother transitions)
            content_aware: Whether to adapt to image content
            adaptive_width: Whether to adapt feathering width to image complexity

        Returns:
            Feathered image with same shape and type as input

        Raises:
            ValueError: If inputs have incompatible shapes or invalid types
        """
        # Input validation
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays")
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have compatible shapes")
        if len(image.shape) > 3 or (len(image.shape) == 3 and image.shape[2] > 3):
            raise ValueError("Image must be grayscale or BGR")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number")
        if image.shape[0] < 3 or image.shape[1] < 3:
            raise ValueError("Image must be at least 3x3 pixels")

        # Process the image
        if len(image.shape) == 3:
            result = self.apply_color_aware_feathering(image, mask, alpha)
        else:
            result = self._apply_single_channel(image, mask, alpha, content_aware)

        return result

    def apply_color_aware_feathering(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.01
    ) -> np.ndarray:
        """Apply color-aware feathering in LAB color space with channel-specific optimization.

        This method processes the image in LAB color space for better perceptual results:
        - L channel (luminance) uses 2.0x alpha to control overall smoothness
        - A/B channels (color) use 0.5x alpha to preserve color transitions
        - All channels use content-aware mode for edge preservation
        
        The LAB color space is chosen because:
        1. Luminance (L) can be processed independently of color
        2. Color channels (A/B) are perceptually uniform
        3. Separate processing prevents color bleeding
        
        Note: Color processing may show more convergence warnings than grayscale
        due to the interaction between channels, but the results remain stable
        as the duality gaps stay proportional to tolerances.

        Args:
            image: BGR image as numpy array
            mask: Binary mask as numpy array
            alpha: Regularization strength (must be positive)

        Returns:
            Feathered image in BGR color space

        Raises:
            ValueError: If inputs have incompatible shapes or invalid types
        """
        # Input validation
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays")
        if len(image.shape) != 3:
            raise ValueError("Image must be a color image (3 channels)")
        if image.shape[2] != 3:
            raise ValueError("Image must be a color image (3 channels)")
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have compatible shapes")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number")

        # Convert to LAB color space
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        result_lab = np.zeros_like(image_lab)

        # Process each channel with appropriate alpha values
        result_lab[..., 0] = self._apply_single_channel(
            image_lab[..., 0], mask,
            alpha=alpha*2.0,  # L channel
            content_aware=True
        )
        result_lab[..., 1] = self._apply_single_channel(
            image_lab[..., 1], mask,
            alpha=alpha*0.5,  # A channel
            content_aware=True
        )
        result_lab[..., 2] = self._apply_single_channel(
            image_lab[..., 2], mask,
            alpha=alpha*0.5,  # B channel
            content_aware=True
        )

        # Convert back to BGR
        result_lab = (result_lab * 255.0).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        return result

    def create_selection_mask(
        self,
        width: int,
        height: int,
        points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Create a binary mask from a list of points.
        
        Args:
            width: Width of the mask
            height: Height of the mask
            points: List of (x, y) coordinates defining the polygon
            
        Returns:
            Binary mask as uint8 array
            
        Raises:
            ValueError: If dimensions are invalid or points list is too short
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        if len(points) < 3:
            return np.zeros((height, width), dtype=np.uint8)
            
        mask = np.zeros((height, width), dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], color=(self.mask_value,))
        return mask

def compute_gabor_features(patch: Union[np.ndarray, Any], num_orientations: int = 4) -> List[float]:
    """Compute Gabor filter responses.
    
    Args:
        patch: Input image patch (grayscale or BGR)
        num_orientations: Number of orientations for Gabor filters
        
    Returns:
        List of Gabor filter responses as Python floats
        
    Raises:
        ValueError: If patch is not a numpy array or num_orientations < 1
    """
    if not isinstance(patch, np.ndarray):
        raise ValueError("Patch must be a numpy array")
    if num_orientations < 1:
        raise ValueError("Number of orientations must be positive")
    
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32 and normalize to [0, 1]
    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch /= 255.0

    features: List[float] = []
    
    for theta in np.linspace(0, np.pi, num_orientations):
        kernel = cv2.getGaborKernel((5, 5), 1.0, theta, 5.0, 1.0, 0, ktype=cv2.CV_32F)
        response = cv2.filter2D(patch, cv2.CV_32F, kernel)
        # Convert OpenCV matrix to NumPy array and ensure float32
        response_np = np.asarray(response).astype(np.float32)
        # Convert numpy values to Python floats
        features.extend([
            float(response_np.mean().item()),
            float(response_np.std().item()),
            float(response_np.max().item()),
            float(response_np.min().item())
        ])
    
    return features

def compute_lbp(patch: Union[np.ndarray, Any]) -> np.ndarray:
    """Compute Local Binary Pattern features.
    
    Args:
        patch: Input image patch (grayscale or BGR)
        
    Returns:
        LBP features as uint8 array
        
    Raises:
        ValueError: If patch is not a numpy array or has invalid dimensions
    """
    if not isinstance(patch, np.ndarray):
        raise ValueError("Patch must be a numpy array")
    if len(patch.shape) not in (2, 3):
        raise ValueError("Patch must be 2D (grayscale) or 3D (BGR)")
    
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Initialize output array with int32 to handle intermediate calculations
    lbp = np.zeros_like(patch, dtype=np.int32)
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        raise ValueError("Patch must be at least 3x3 pixels")
        
    center = patch[1:-1, 1:-1]
    
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                # Calculate binary pattern
                pattern = (patch[i:i+patch.shape[0]-2, j:j+patch.shape[1]-2] > center) * (1 << ((i*3 + j) % 8))
                lbp[1:-1, 1:-1] += pattern
    
    # Convert back to uint8 after all calculations are done
    return lbp.astype(np.uint8) 