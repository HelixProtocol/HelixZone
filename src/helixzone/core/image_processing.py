"""Image processing module for HelixZone."""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from skimage import color
from concurrent.futures import ThreadPoolExecutor  # for parallel processing
from .gpu_manager import GPUResourceMonitor, MultiGPUManager
from .logging_manager import LoggingManager
from .type_defs import (
    ImageSource, EdgeDetectionParams, ProcessingResults,
    is_image_array, is_gray_image, is_color_image
)

@dataclass
class ImageStats:
    """Statistics for an image."""
    mean: float
    std: float
    min: float
    max: float
    histogram: Any
    dominant_colors: List[Tuple[int, int, int]]

class ColorSpaceConverter:
    """Handles color space conversions with proper normalization."""
    
    @staticmethod
    def rgb_to_lab(image: Any) -> Any:
        """Convert RGB to LAB color space."""
        # Ensure proper input range [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return color.rgb2lab(image)
        
    @staticmethod
    def lab_to_rgb(image: Any) -> Any:
        """Convert LAB to RGB color space."""
        rgb = color.lab2rgb(image)
        return (rgb * 255).astype(np.uint8)
        
    @staticmethod
    def ensure_color(image: Any) -> Any:
        """Ensure image is in color format (3 channels)."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def rgb_to_gray(image: Any) -> Any:
        """Convert RGB image to grayscale."""
        if not is_color_image(image):
            raise ValueError("Input must be a color image")
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    @staticmethod
    def gray_to_rgb(image: Any) -> Any:
        """Convert grayscale image to RGB."""
        if not is_gray_image(image):
            raise ValueError("Input must be a grayscale image")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

class ImageProcessor:
    """Processes images with GPU acceleration when available."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        logger: Optional[LoggingManager] = None
    ) -> None:
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = logger or LoggingManager()
        self.multi_gpu = MultiGPUManager()
        self.gpu_monitor = GPUResourceMonitor(multi_gpu=self.multi_gpu)
        self.color_converter = ColorSpaceConverter()
    
    def analyze_image(self, image: Any) -> ImageStats:
        """Analyze image statistics."""
        if not is_image_array(image):
            raise ValueError("Expected a uint8 numpy array")
        
        # Convert to float32 for accurate calculations
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0
            
        # Calculate histogram
        hist_list = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_list.append(hist.flatten())
        histogram = np.array(hist_list)
        
        # Find dominant colors using k-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        
        # Initialize arrays for k-means
        labels = np.zeros(pixels.shape[0], dtype=np.int32)
        centers = np.zeros((n_colors, 3), dtype=np.float32)
        
        # Run k-means
        ret, labels, centers = cv2.kmeans(
            data=pixels,
            K=n_colors,
            bestLabels=labels,
            criteria=criteria,
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert centers to RGB colors
        centers_array = centers
        dominant_colors: List[Tuple[int, int, int]] = [
            (int(center[0]), int(center[1]), int(center[2]))
            for center in centers_array
        ]
        
        return ImageStats(
            mean=float(img_float.mean()),
            std=float(img_float.std()),
            min=float(img_float.min()),
            max=float(img_float.max()),
            histogram=histogram,
            dominant_colors=dominant_colors
        )
    
    def enhance_image(
        self,
        image: Any,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0
    ) -> Any:
        """Enhance image appearance."""
        if not is_image_array(image):
            raise ValueError("Expected a uint8 numpy array")
            
        with self.logger.track_operation("image_enhancement"):
            if self.use_gpu:
                return self._enhance_image_gpu(image, brightness, contrast, saturation)
            return self._enhance_image_cpu(image, brightness, contrast, saturation)
            
    def _enhance_image_gpu(
        self,
        image: Any,
        brightness: float,
        contrast: float,
        saturation: float
    ) -> Any:
        """Enhance image using GPU acceleration."""
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().cuda()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.max() > 1.0:
            image_tensor /= 255.0
            
        # Apply brightness
        image_tensor = image_tensor * brightness
        
        # Apply contrast
        mean = image_tensor.mean()
        image_tensor = (image_tensor - mean) * contrast + mean
        
        # Apply saturation in HSV space
        if len(image_tensor.shape) == 3:
            # Convert to HSV
            image_hsv = color.rgb2hsv(image_tensor.cpu().numpy())
            image_hsv[..., 1] = np.clip(image_hsv[..., 1] * saturation, 0, 1)
            # Convert back to RGB
            image_tensor = torch.from_numpy(
                color.hsv2rgb(image_hsv)
            ).float().cuda()
            
        # Ensure proper range
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert back to numpy
        result = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return result
        
    def _enhance_image_cpu(
        self,
        image: Any,
        brightness: float,
        contrast: float,
        saturation: float
    ) -> Any:
        """Enhance image on CPU."""
        # Convert to float32
        image_float = image.astype(np.float32)
        if image_float.max() > 1.0:
            image_float /= 255.0
            
        # Apply brightness
        image_float = image_float * brightness
        
        # Apply contrast
        mean = image_float.mean()
        image_float = (image_float - mean) * contrast + mean
        
        # Apply saturation in HSV space
        if len(image_float.shape) == 3:
            image_hsv = color.rgb2hsv(image_float)
            image_hsv[..., 1] = np.clip(image_hsv[..., 1] * saturation, 0, 1)
            image_float = color.hsv2rgb(image_hsv)
            
        # Ensure proper range and convert back to uint8
        image_float = np.clip(image_float, 0, 1)
        return (image_float * 255).astype(np.uint8)
        
    def apply_filter(
        self,
        image: Any,
        kernel_size: int = 3,
        sigma: float = 1.0,
        filter_type: str = 'gaussian'
    ) -> Any:
        """Apply various types of filters."""
        if not is_image_array(image):
            raise ValueError("Expected a uint8 numpy array")
            
        with self.logger.track_operation("filter_application"):
            if self.use_gpu:
                return self._apply_filter_gpu(image, kernel_size, sigma, filter_type)
            return self._apply_filter_cpu(image, kernel_size, sigma, filter_type)
            
    def _apply_filter_gpu(
        self,
        image: Any,
        kernel_size: int,
        sigma: float,
        filter_type: str
    ) -> Any:
        """Apply filter using GPU acceleration."""
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().cuda()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.max() > 1.0:
            image_tensor /= 255.0
            
        if filter_type == 'gaussian':
            # Create Gaussian kernel
            kernel = self._create_gaussian_kernel(kernel_size, sigma).cuda()
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            
            # Apply filter
            if len(image_tensor.shape) == 3:
                result = torch.zeros_like(image_tensor)
                for c in range(image_tensor.shape[2]):
                    channel = image_tensor[..., c].unsqueeze(0).unsqueeze(0)
                    result[..., c] = torch.nn.functional.conv2d(
                        channel,
                        kernel,
                        padding=kernel_size//2
                    ).squeeze()
            else:
                result = torch.nn.functional.conv2d(
                    image_tensor.unsqueeze(0).unsqueeze(0),
                    kernel,
                    padding=kernel_size//2
                ).squeeze()
                
        elif filter_type == 'median':
            # Use torch.median with sliding window
            pad_size = kernel_size // 2
            padded = torch.nn.functional.pad(
                image_tensor,
                (pad_size, pad_size, pad_size, pad_size),
                mode='reflect'
            )
            result = torch.zeros_like(image_tensor)
            
            for i in range(image_tensor.shape[0]):
                for j in range(image_tensor.shape[1]):
                    window = padded[
                        i:i+kernel_size,
                        j:j+kernel_size
                    ]
                    result[i, j] = torch.median(window.view(-1))
                    
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
        # Convert back to numpy
        result = torch.clamp(result, 0, 1)
        return (result.cpu().numpy() * 255).astype(np.uint8)
        
    def _apply_filter_cpu(
        self,
        image: Any,
        kernel_size: int,
        sigma: float,
        filter_type: str
    ) -> Any:
        """Apply filter on CPU."""
        if filter_type == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif filter_type == 'median':
            return cv2.medianBlur(image, kernel_size)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
    def _create_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float
    ) -> Tensor:
        """Create a Gaussian kernel."""
        # Create a 1D Gaussian kernel
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        gaussian = torch.exp(-(x**2)/(2*sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        # Create 2D kernel
        kernel_2d = gaussian.view(-1, 1) * gaussian.view(1, -1)
        return kernel_2d
        
    def detect_edges(self, image: Any, params: EdgeDetectionParams) -> ProcessingResults:
        """Detect edges in image using Canny algorithm."""
        if not is_image_array(image):
            raise ValueError("Expected a uint8 numpy array")
        
        # Convert to grayscale if needed
        if is_color_image(image):
            gray = self.color_converter.rgb_to_gray(image)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), params['sigma'])
        
        # Detect edges
        edges = cv2.Canny(
            blurred,
            params['low_threshold'],
            params['high_threshold']
        )
        
        return ProcessingResults(
            original=image,
            processed=edges.astype(np.float32) / 255.0,
            params=params
        ) 