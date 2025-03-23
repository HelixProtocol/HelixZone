"""Edge detection module for HelixZone."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, cast
from typing_extensions import Final
import numpy as np
from numpy.typing import NDArray
import cv2
import warnings

# Type aliases
ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
EdgeMap = NDArray[np.bool_]

class EdgeDetector:
    """Edge detection with GPU acceleration support."""

    def __init__(self) -> None:
        self._initialized = False
        self._has_cuda = False
        self._has_opencl = False

    def initialize(self) -> None:
        """Initialize edge detector."""
        if self._initialized:
            return

        try:
            import cupy as cp
            self._has_cuda = True
        except ImportError:
            warnings.warn("CUDA not available, falling back to CPU processing")

        try:
            import pyopencl as cl
            self._has_opencl = True
        except ImportError:
            warnings.warn("OpenCL not available, falling back to CPU processing")

        self._initialized = True

    def detect_edges(
        self,
        img: ImageArray,
        threshold1: float = 100.0,
        threshold2: float = 200.0,
        aperture_size: int = 3,
        l2_gradient: bool = False
    ) -> EdgeMap:
        """Detect edges in an image using Canny edge detection."""
        if not self._initialized:
            self.initialize()

        if self._has_cuda:
            try:
                import cupy as cp
                gpu_img = cp.asarray(img)
                if len(img.shape) == 3:
                    gpu_img = cp.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                
                # Apply Gaussian blur
                gpu_img = cp.asarray(cv2.GaussianBlur(cp.asnumpy(gpu_img), (5, 5), 0))
                
                # Compute gradients
                sobelx = cp.asarray(cv2.Sobel(cp.asnumpy(gpu_img), cv2.CV_32F, 1, 0, ksize=3))
                sobely = cp.asarray(cv2.Sobel(cp.asnumpy(gpu_img), cv2.CV_32F, 0, 1, ksize=3))
                
                # Compute gradient magnitude and direction
                magnitude = cp.sqrt(sobelx * sobelx + sobely * sobely)
                direction = cp.arctan2(sobely, sobelx)
                
                # Non-maximum suppression
                suppressed = cp.zeros_like(magnitude)
                for i in range(1, magnitude.shape[0] - 1):
                    for j in range(1, magnitude.shape[1] - 1):
                        angle = direction[i, j] * 180.0 / np.pi
                        angle = angle % 180.0
                        
                        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                            q = magnitude[i, j+1]
                            r = magnitude[i, j-1]
                        elif 22.5 <= angle < 67.5:
                            q = magnitude[i+1, j-1]
                            r = magnitude[i-1, j+1]
                        elif 67.5 <= angle < 112.5:
                            q = magnitude[i+1, j]
                            r = magnitude[i-1, j]
                        else:
                            q = magnitude[i-1, j-1]
                            r = magnitude[i+1, j+1]
                            
                        if magnitude[i, j] >= q and magnitude[i, j] >= r:
                            suppressed[i, j] = magnitude[i, j]
                
                # Double thresholding
                strong_edges = suppressed > threshold2
                weak_edges = (suppressed >= threshold1) & (suppressed <= threshold2)
                edges = cp.zeros_like(strong_edges, dtype=bool)
                edges[strong_edges] = True
                
                # Edge tracking by hysteresis
                for i in range(1, edges.shape[0] - 1):
                    for j in range(1, edges.shape[1] - 1):
                        if weak_edges[i, j]:
                            if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                                edges[i, j] = True
                
                return cp.asnumpy(edges)
            except Exception as e:
                warnings.warn(f"CUDA processing failed, falling back to CPU: {e}")

        # Fall back to CPU processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        return cv2.Canny(
            gray,
            threshold1=threshold1,
            threshold2=threshold2,
            apertureSize=aperture_size,
            L2gradient=l2_gradient
        ).astype(bool)

    def save_results(
        self,
        edges: EdgeMap,
        output_path: str,
        original_img: Optional[ImageArray] = None
    ) -> None:
        """Save edge detection results."""
        # Convert boolean array to uint8
        edge_img = edges.astype(np.uint8) * 255
        
        if original_img is not None:
            # Create a 3-channel image with red edges
            result = original_img.copy()
            result[edges] = [0, 0, 255]  # Red color for edges
            cv2.imwrite(output_path, result)
        else:
            cv2.imwrite(output_path, edge_img) 