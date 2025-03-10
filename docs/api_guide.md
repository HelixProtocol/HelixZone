# HelixZone API Guide

## Core Classes

### EnhancedLassoFeathering

The main class for performing advanced feathering operations with sophisticated edge detection and color processing.

```python
from helixzone.core.ml_utils import EnhancedLassoFeathering

feathering = EnhancedLassoFeathering()
```

#### Methods

##### apply_lasso_feathering
```python
def apply_lasso_feathering(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.01,
    content_aware: bool = False,
    adaptive_width: bool = False
) -> np.ndarray:
    """
    Apply lasso feathering to an image.

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
        ValueError: If image must be at least 3x3 pixels
        ValueError: If image must be grayscale or BGR
    """
```

##### apply_color_aware_feathering
```python
def apply_color_aware_feathering(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.01
) -> np.ndarray:
    """
    Apply color-aware feathering in LAB color space with channel-specific optimization.

    This method processes the image in LAB color space for better perceptual results:
    - L channel (luminance) uses 2.0x alpha to control overall smoothness
    - A/B channels (color) use 0.5x alpha to preserve color transitions
    - All channels use content-aware mode for edge preservation
    
    The LAB color space is chosen because:
    1. Luminance (L) can be processed independently of color
    2. Color channels (A/B) are perceptually uniform
    3. Separate processing prevents color bleeding

    Args:
        image: BGR image as numpy array
        mask: Binary mask as numpy array
        alpha: Base regularization strength (must be positive)

    Returns:
        Feathered image in BGR color space

    Raises:
        ValueError: If image is not a color image (3 channels)
        ValueError: If inputs have incompatible shapes
        ValueError: If alpha is not positive
    """
```

##### create_selection_mask
```python
def create_selection_mask(
    width: int,
    height: int,
    points: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Create a binary mask from a list of points.
    
    Args:
        width: Width of the mask
        height: Height of the mask
        points: List of (x, y) coordinates defining the polygon
        
    Returns:
        Binary mask as uint8 array
        
    Raises:
        ValueError: If dimensions are invalid or points list is too short
    """
```

## Implementation Details

### Feature Extraction

The feathering process uses sophisticated feature extraction including:
- Multi-scale edge detection combining fine and coarse edges
- Local contrast and gradient information
- Color-aware processing in LAB space
- Adaptive feature weighting based on image content

### Optimization Strategy

The implementation uses an optimized ElasticNet regression approach:
1. Features and targets are scaled using RobustScaler with quantile range (1, 99)
2. Two-stage fitting process:
   - Initial fit with higher regularization for stability
   - Refinement with desired alpha using pre-fitted coefficients
3. Sample weights reduce impact of outliers
4. Content-aware mode:
   - Alpha varies from 0.05x to 3.0x based on edge strength
   - Edge features weighted at 1.2x
   - Final smoothing uses 0.6/0.4 balance

## Error Handling

```python
try:
    result = feathering.apply_lasso_feathering(image, mask)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

## Examples

### Basic Usage
```python
import cv2
import numpy as np
from helixzone.core.ml_utils import EnhancedLassoFeathering

# Load image and create mask
image = cv2.imread("input.png")
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(mask, (100, 100), 50, 255, -1)

# Initialize and apply feathering
feathering = EnhancedLassoFeathering()
result = feathering.apply_lasso_feathering(
    image,
    mask,
    alpha=0.01,
    content_aware=True
)

# Save result
cv2.imwrite("output.png", result)
```

### Color-Aware Processing
```python
# Load inputs
image = cv2.imread("portrait.png")
mask = cv2.imread("mask.png", 0)

# Apply color-aware feathering
result = feathering.apply_color_aware_feathering(
    image,
    mask,
    alpha=0.01  # Controls overall smoothness
)

# Composite result
output = cv2.multiply(image, result)
```

## Advanced Features

### GPU Acceleration

```python
# Enable GPU acceleration
feathering.enable_gpu()

# Check GPU availability
is_gpu_available = feathering.has_gpu_support()

# Set specific device
feathering.set_gpu_device(0)  # Use first GPU
```

### Batch Processing

```python
def process_batch(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    **kwargs
) -> List[np.ndarray]:
    """
    Process multiple images efficiently.
    
    Args:
        images: List of input images
        masks: List of input masks
        **kwargs: Parameters passed to apply_lasso_feathering
        
    Returns:
        List of processed masks
    """
    return [
        feathering.apply_lasso_feathering(img, mask, **kwargs)
        for img, mask in zip(images, masks)
    ]
```

## Performance Tips

### Memory Optimization
```python
# For large images, use streaming mode
feathering.enable_streaming_mode(chunk_size=(1024, 1024))

# Clear GPU memory if needed
feathering.clear_gpu_memory()
```

### Multi-Threading
```python
# Enable multi-threading
feathering.set_num_threads(4)  # Use 4 threads

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_single, images, masks))
``` 