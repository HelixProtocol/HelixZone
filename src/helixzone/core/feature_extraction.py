"""Feature extraction module for HelixZone.

This module provides functionality for extracting image features using both CPU and GPU.
Features are extracted from image patches around specified coordinates using various
techniques including:
- HOG (Histogram of Oriented Gradients)
- SIFT-like descriptors
- Color histograms
- Texture features
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, cast
from typing_extensions import TypeAlias, Final
import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import Tensor
from .type_defs import ImageArray, GrayImage, FloatArray, Array
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings

# Type aliases
Mat = Union[ImageArray, FloatArray]
Coordinate = Tuple[int, int]
FeatureVector = NDArray[np.float32]
FeatureMap = Dict[str, NDArray[np.float32]]

@dataclass
class FeatureExtractionParams:
    """Parameters for feature extraction."""
    patch_size: Tuple[int, int] = field(default=(64, 64))
    num_orientations: int = field(default=9)
    pixels_per_cell: Tuple[int, int] = field(default=(8, 8))
    cells_per_block: Tuple[int, int] = field(default=(3, 3))
    color_bins: int = field(default=32)
    use_gpu: bool = field(default=False)

class FeatureExtractor:
    """Feature extraction with GPU acceleration support."""

    def __init__(self) -> None:
        self._initialized = False
        self._has_cuda = False
        self._has_opencl = False
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def initialize(self) -> None:
        """Initialize feature extractor."""
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

    def extract_features(
        self,
        img: ImageArray,
        compute_hog: bool = True,
        compute_color: bool = True,
        compute_texture: bool = True
    ) -> FeatureMap:
        """Extract features from an image."""
        if not self._initialized:
            self.initialize()

        features: FeatureMap = {}

        if self._has_cuda:
            try:
                import cupy as cp
                gpu_img = cp.asarray(img)

                if compute_hog:
                    # HOG features are computed on CPU since CUDA HOG is not available
                    features['hog'] = np.array(self._hog.compute(img), dtype=np.float32).flatten()

                if compute_color and len(img.shape) == 3:
                    features['color'] = self._compute_color_histogram_gpu(gpu_img)

                if compute_texture:
                    features['texture'] = self._compute_texture_features_gpu(gpu_img)

                return features
            except Exception as e:
                warnings.warn(f"CUDA processing failed, falling back to CPU: {e}")

        # Fall back to CPU processing
        if compute_hog:
            features['hog'] = np.array(self._hog.compute(img), dtype=np.float32).flatten()

        if compute_color and len(img.shape) == 3:
            features['color'] = self._compute_color_histogram_cpu(img)

        if compute_texture:
            features['texture'] = self._compute_texture_features_cpu(img)

        return features

    def _compute_color_histogram_cpu(self, img: ImageArray) -> FloatArray:
        """Compute color histogram using CPU."""
        # Cast img to Mat type for calcHist
        img_mat = cast('cv2.Mat', img)
        hist = cv2.calcHist([img_mat], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return hist.flatten().astype(np.float32)

    def _compute_texture_features_cpu(self, img: ImageArray) -> FloatArray:
        """Compute texture features using CPU."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        direction = np.arctan2(grad_y, grad_x)

        # Create feature vector
        features = np.concatenate([
            magnitude.flatten(),
            direction.flatten()
        ]).astype(np.float32)

        return features

    def _compute_color_histogram_gpu(self, gpu_img: NDArray[np.uint8]) -> FloatArray:
        """Compute color histogram using GPU."""
        import cupy as cp
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(cp.asnumpy(gpu_img), cv2.COLOR_BGR2HSV)
        hsv_gpu = cp.asarray(hsv)

        # Compute histogram for each channel
        hist_h = cp.histogram(hsv_gpu[:, :, 0].flatten(), bins=8, range=(0, 180))
        hist_s = cp.histogram(hsv_gpu[:, :, 1].flatten(), bins=8, range=(0, 256))
        hist_v = cp.histogram(hsv_gpu[:, :, 2].flatten(), bins=8, range=(0, 256))

        # Normalize histograms
        hist_h = hist_h[0].astype(np.float32) / hist_h[0].sum()
        hist_s = hist_s[0].astype(np.float32) / hist_s[0].sum()
        hist_v = hist_v[0].astype(np.float32) / hist_v[0].sum()

        # Combine histograms
        return cp.asnumpy(cp.concatenate([hist_h, hist_s, hist_v]))

    def _compute_texture_features_gpu(self, gpu_img: NDArray[np.uint8]) -> FloatArray:
        """Compute texture features using GPU."""
        import cupy as cp

        # Convert to grayscale if needed
        if len(gpu_img.shape) == 3:
            gray = cv2.cvtColor(cp.asnumpy(gpu_img), cv2.COLOR_BGR2GRAY)
            gray_gpu = cp.asarray(gray)
        else:
            gray_gpu = gpu_img

        # Compute gradients
        grad_x = cp.asarray(cv2.Sobel(cp.asnumpy(gray_gpu), cv2.CV_32F, 1, 0, ksize=3))
        grad_y = cp.asarray(cv2.Sobel(cp.asnumpy(gray_gpu), cv2.CV_32F, 0, 1, ksize=3))

        # Compute gradient magnitude and direction
        magnitude = cp.sqrt(grad_x * grad_x + grad_y * grad_y)
        direction = cp.arctan2(grad_y, grad_x)

        # Create feature vector
        features = cp.concatenate([
            magnitude.flatten(),
            direction.flatten()
        ]).astype(np.float32)

        return cp.asnumpy(features)

    def _extract_patches_gpu(
        self,
        image: Tensor,
        coords: List[Coordinate],
        patch_size: int
    ) -> Tensor:
        """Extract patches around coordinates using GPU.
        
        Args:
            image: Input image tensor on GPU
            coords: List of (x, y) coordinates
            patch_size: Size of patches to extract
            
        Returns:
            Tensor of patches with shape (N, C, H, W)
        """
        half_size = patch_size // 2
        height, width = image.shape[:2]
        num_patches = len(coords)
        
        # Create patch indices
        y_indices = torch.arange(-half_size, half_size + 1, device=image.device)
        x_indices = torch.arange(-half_size, half_size + 1, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Add coordinates
        coords_tensor = torch.tensor(coords, device=image.device)
        patch_y = grid_y.unsqueeze(0) + coords_tensor[:, 1:2]
        patch_x = grid_x.unsqueeze(0) + coords_tensor[:, 0:1]
        
        # Clip to image bounds
        patch_y = torch.clamp(patch_y, 0, height - 1)
        patch_x = torch.clamp(patch_x, 0, width - 1)
        
        # Extract patches
        patches = image[patch_y, patch_x]
        return patches

    def _extract_patches_cpu(
        self,
        image: ImageArray,
        coords: List[Coordinate],
        patch_size: int
    ) -> ImageArray:
        """Extract patches around coordinates using CPU.
        
        Args:
            image: Input image array
            coords: List of (x, y) coordinates
            patch_size: Size of patches to extract
            
        Returns:
            Array of patches with shape (N, H, W, C)
        """
        half_size = patch_size // 2
        height, width = image.shape[:2]
        num_patches = len(coords)
        
        # Initialize output array
        if len(image.shape) == 3:
            patches = np.zeros((num_patches, patch_size, patch_size, image.shape[2]), dtype=np.uint8)
        else:
            patches = np.zeros((num_patches, patch_size, patch_size), dtype=np.uint8)
        
        # Extract each patch
        for i, (x, y) in enumerate(coords):
            # Calculate patch bounds
            y1 = max(0, y - half_size)
            y2 = min(height, y + half_size + 1)
            x1 = max(0, x - half_size)
            x2 = min(width, x + half_size + 1)
            
            # Extract and pad if necessary
            patch = image[y1:y2, x1:x2]
            if patch.shape[:2] != (patch_size, patch_size):
                if len(image.shape) == 3:
                    padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=np.uint8)
                else:
                    padded = np.zeros((patch_size, patch_size), dtype=np.uint8)
                py1 = half_size - (y - y1)
                py2 = py1 + (y2 - y1)
                px1 = half_size - (x - x1)
                px2 = px1 + (x2 - x1)
                padded[py1:py2, px1:px2] = patch
                patches[i] = padded
            else:
                patches[i] = patch
        
        return patches 