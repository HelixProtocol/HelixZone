"""Machine learning utilities for HelixZone."""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import cv2
from typing import Dict, List, Union, Optional, Tuple
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel

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
    """
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

def create_feature_matrix(img: np.ndarray, coords: List[Tuple[int, int]], patch_size: int = 5) -> np.ndarray:
    """Create enhanced feature matrix with texture and structure information."""
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
        feature_list.extend([float(f) for f in gabor_features])
        
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
        """Compute edge strength map."""
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
        """
        features = []

        # Convert to float and normalize
        img_float = image.astype(float) / 255.0

        # Pre-compute edge information
        edge_map = self.compute_edge_strength(image)

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
        content_aware: bool = False,
        adaptive_width: bool = False
    ) -> np.ndarray:
        """Apply feathering to a single channel."""
        # Create initial transition mask
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
        transition_mask = (dilated - eroded).astype(bool)

        # Get edge strength map for adaptive width
        if adaptive_width or content_aware:
            edge_strength = self.compute_edge_strength(image)
            if adaptive_width:
                # Enhance edge detection for more extreme differences
                edge_strength = cv2.Canny((image * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
                edge_strength = cv2.dilate(edge_strength, np.ones((3, 3), np.uint8))
                
                # Create binary mask of complex regions
                complex_mask = edge_strength > 0.5
                
                # Only apply adaptive width in transition regions
                complex_mask = complex_mask & transition_mask
            if content_aware:
                # For content-aware, use much stronger regularization in non-edge regions
                edge_strength = cv2.dilate(edge_strength.astype(np.float32), np.ones((3, 3), np.uint8))
                alpha_adj = np.where(edge_strength > 0.5,
                                   alpha * 0.1,  # Very low regularization near edges
                                   alpha * 5.0)  # Strong regularization in smooth regions
                alpha = float(np.mean(alpha_adj))  # Convert to float for Lasso

        # Create initial result with binary values
        result = mask.copy().astype(np.float32)

        # Only process transition regions
        if np.any(transition_mask):
            # Create coordinates for transition points
            y_coords, x_coords = np.where(transition_mask)
            coords = list(zip(y_coords, x_coords))

            # Create feature matrix for transition points
            X = self.create_advanced_features(image, coords)
            y = mask[transition_mask]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Fit model with stronger regularization for smoother transitions
            model = Lasso(alpha=alpha, max_iter=2000, tol=1e-4)
            model.fit(X_scaled, y)

            # Predict transition values
            y_pred = model.predict(X_scaled)
            result[transition_mask] = y_pred

        # Apply adaptive smoothing if requested
        if adaptive_width and np.any(transition_mask):
            # Create separate results for smooth and complex regions
            result_smooth = result.copy()
            result_complex = result.copy()
            
            # Apply multiple passes of smoothing for smooth regions
            sigmas_smooth = [40.0, 30.0, 20.0]  # Multiple passes with large sigmas
            for sigma in sigmas_smooth:
                result_smooth = gaussian_filter(result_smooth, sigma=sigma)
            
            # Apply minimal smoothing for complex regions
            result_complex = gaussian_filter(result_complex, sigma=1.0)  # Very sharp transitions
            
            # Create transition weights based on edge strength
            weights = np.zeros_like(result)
            weights[transition_mask] = 1.0
            weights = gaussian_filter(weights, sigma=5.0)  # Smooth the weights
            
            # Blend results based on region type and transition weights
            result = np.where(transition_mask,
                             np.where(complex_mask,
                                     result_complex,
                                     result_smooth),
                             result)

        return np.clip(result, 0, 1)

    def apply_lasso_feathering(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.01,
        content_aware: bool = False,
        adaptive_width: bool = False
    ) -> np.ndarray:
        """Apply Lasso-based feathering to an image.
        
        Args:
            image: Input image (grayscale or BGR)
            mask: Binary mask
            alpha: Regularization strength (higher = smoother)
            content_aware: Whether to use content-aware features
            adaptive_width: Whether to use adaptive transition width
            
        Returns:
            Feathered mask
        """
        # Input validation
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays")
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have compatible shapes")
        if len(image.shape) > 3 or (len(image.shape) == 3 and image.shape[2] > 3):
            raise ValueError("Image must be grayscale or BGR (3 channels)")
        
        if len(image.shape) == 3:
            # Process each channel separately
            channels = []
            for i in range(image.shape[2]):
                channel_result = self._apply_single_channel(
                    image[..., i],
                    mask,
                    alpha=alpha,
                    content_aware=content_aware,
                    adaptive_width=adaptive_width
                )
                channels.append(channel_result)
            result = np.stack(channels, axis=-1)
        else:
            result = self._apply_single_channel(
                image,
                mask,
                alpha=alpha,
                content_aware=content_aware,
                adaptive_width=adaptive_width
            )
        
        return np.clip(result, 0, 1)

    def apply_color_aware_feathering(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.01
    ) -> np.ndarray:
        """Apply feathering with color awareness."""
        if len(image.shape) != 3:
            raise ValueError("Image must be a color image (3 channels)")
        if image.shape[2] != 3:
            raise ValueError("Image must have 3 color channels")

        # Convert to LAB color space for better color difference measurement
        image_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        # Process each channel with different alpha values
        result_lab = np.zeros_like(image_lab)
        result_lab[..., 0] = self._apply_single_channel(image_lab[..., 0], mask, alpha=alpha*2.0)  # L channel
        result_lab[..., 1] = self._apply_single_channel(image_lab[..., 1], mask, alpha=alpha*0.5)  # A channel
        result_lab[..., 2] = self._apply_single_channel(image_lab[..., 2], mask, alpha=alpha*0.5)  # B channel

        # Convert back to BGR
        result = cv2.cvtColor((result_lab * 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

        return result

    def create_selection_mask(self, width: int, height: int, points: List[Tuple[int, int]]) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) < 3:
            return mask
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], color=(self.mask_value,))
        return mask

def compute_gabor_features(patch: np.ndarray, num_orientations: int = 4) -> np.ndarray:
    """Compute Gabor filter responses."""
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32 and normalize to [0, 1]
    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch /= 255.0

    features = []
    for theta in np.linspace(0, np.pi, num_orientations):
        kernel = cv2.getGaborKernel((5, 5), 1.0, theta, 5.0, 1.0, 0, ktype=cv2.CV_32F)
        response = cv2.filter2D(patch, cv2.CV_32F, kernel)
        # Convert OpenCV matrix to NumPy array and ensure float32
        response_np = np.asarray(response).astype(np.float32)
        features.extend([
            response_np.mean().item(),  # Convert to Python scalar
            response_np.std().item(),   # Convert to Python scalar
            response_np.max().item(),   # Convert to Python scalar
            response_np.min().item()    # Add min value for more features
        ])
    
    return np.array(features)

def compute_lbp(patch: np.ndarray) -> np.ndarray:
    """Compute Local Binary Pattern features."""
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    lbp = np.zeros_like(patch)
    center = patch[1:-1, 1:-1]
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                lbp[1:-1, 1:-1] += (patch[i:i+patch.shape[0]-2, j:j+patch.shape[1]-2] > center) * (1 << ((i*3 + j) % 8))
    return lbp 