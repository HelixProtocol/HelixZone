"""Image processing module for HelixZone."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, cast, Literal, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
from numpy.typing import NDArray
from .type_defs import (
    ImageArray,
    FloatArray,
    EdgeDetectionParams,
    GPUBackend
)

# Type definitions
ImgArray = NDArray[np.uint8]
FloatImgArray = NDArray[np.float32]
ColorMode = Literal['rgb', 'grayscale']

@dataclass
class ImageProcessingParams:
    """Parameters for image processing."""
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    normalize: bool = True
    denoise: bool = False
    denoise_strength: int = 10
    sharpen: bool = False
    sharpen_strength: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    gamma: float = 1.0
    mode: ColorMode = 'rgb'

@dataclass
class ProcessingResult:
    """Result of image processing."""
    processed_image: ImageArray
    metadata: Dict[str, Union[int, float, str]] = field(default_factory=dict)

class ImageProcessor:
    """Image processing class with GPU acceleration support."""

    def __init__(self, backend: GPUBackend = 'cpu') -> None:
        self.backend = backend
        self._initialized = False

    def initialize(self) -> None:
        """Initialize processor with selected backend."""
        if self._initialized:
            return

        if self.backend == 'cuda':
            try:
                import cupy as cp
                self._initialized = True
            except ImportError:
                self.backend = 'cpu'
        elif self.backend == 'opencl':
            try:
                import pyopencl as cl
                self._initialized = True
            except ImportError:
                self.backend = 'cpu'
        
        self._initialized = True

    def _to_uint8(self, img: NDArray[Any]) -> ImgArray:
        """Convert any image array to uint8."""
        if img.dtype == np.uint8:
            return img
        return (np.clip(img * 255.0, 0, 255)).astype(np.uint8)

    def _to_float32(self, img: NDArray[Any]) -> FloatImgArray:
        """Convert any image array to float32."""
        if img.dtype == np.float32:
            return img
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def process_image(
        self,
        image: ImageArray,
        params: Optional[ImageProcessingParams] = None
    ) -> ProcessingResult:
        """Process image with given parameters."""
        if not self._initialized:
            self.initialize()

        params = params or ImageProcessingParams()
        processed = self._to_float32(image)

        # Resize if requested
        if params.resize_width is not None and params.resize_height is not None:
            processed = self._to_float32(
                cv2.resize(
                    src=self._to_uint8(processed),
                    dsize=(params.resize_width, params.resize_height),
                    interpolation=cv2.INTER_LANCZOS4
                )
            )

        # Denoise if requested
        if params.denoise:
            if len(processed.shape) == 3:
                uint8_img = self._to_uint8(processed)
                denoised = cv2.fastNlMeansDenoisingColored(
                    src=uint8_img,
                    dst=np.zeros_like(uint8_img),
                    h=params.denoise_strength,
                    hColor=params.denoise_strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                processed = self._to_float32(denoised)
            else:
                uint8_img = self._to_uint8(processed)
                denoised = cv2.fastNlMeansDenoising(
                    src=uint8_img,
                    dst=np.zeros_like(uint8_img),
                    h=params.denoise_strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                processed = self._to_float32(denoised)

        # Apply contrast and brightness
        processed = self._to_float32(
            cv2.convertScaleAbs(
                src=self._to_uint8(processed),
                alpha=params.contrast,
                beta=params.brightness
            )
        )

        # Apply gamma correction
        if params.gamma != 1.0:
            processed = np.power(processed, 1.0/params.gamma)

        # Sharpen if requested
        if params.sharpen:
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ], dtype=np.float32) * params.sharpen_strength
            processed = self._to_float32(
                cv2.filter2D(
                    src=self._to_uint8(processed),
                    ddepth=-1,
                    kernel=kernel
                )
            )

        # Convert color space if needed
        if params.mode == 'grayscale' and len(processed.shape) == 3:
            processed = self._to_float32(
                cv2.cvtColor(
                    self._to_uint8(processed),
                    cv2.COLOR_RGB2GRAY
                )
            )
        elif params.mode == 'rgb' and len(processed.shape) == 2:
            processed = self._to_float32(
                cv2.cvtColor(
                    self._to_uint8(processed),
                    cv2.COLOR_GRAY2RGB
                )
            )

        # Normalize to [0, 255] range
        if params.normalize:
            uint8_img = self._to_uint8(processed)
            processed = cv2.normalize(
                src=uint8_img,
                dst=np.zeros_like(uint8_img),
                alpha=0,
                beta=255,
                normType=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
        else:
            processed = self._to_uint8(processed)

        # Collect metadata
        metadata = {
            'width': processed.shape[1],
            'height': processed.shape[0],
            'channels': processed.shape[2] if len(processed.shape) == 3 else 1,
            'mode': params.mode,
            'backend': self.backend
        }

        return ProcessingResult(processed_image=processed, metadata=metadata)

    def enhance_details(
        self,
        image: ImageArray,
        strength: float = 1.0,
        radius: int = 2
    ) -> ImageArray:
        """Enhance image details using unsharp masking."""
        if not self._initialized:
            self.initialize()

        # Convert to float32
        img_float = self._to_float32(image)

        # Create gaussian blur
        blurred = self._to_float32(
            cv2.GaussianBlur(
                src=self._to_uint8(img_float),
                ksize=(0, 0),
                sigmaX=radius
            )
        )
        
        # Calculate unsharp mask
        mask = img_float - blurred
        
        # Apply mask with strength
        sharpened = img_float + mask * strength
        
        # Clip values and convert back to uint8
        return self._to_uint8(sharpened)

    def adjust_colors(
        self,
        image: ImageArray,
        saturation: float = 1.0,
        temperature: float = 0.0,
        tint: float = 0.0
    ) -> ImageArray:
        """Adjust image colors."""
        if not self._initialized:
            self.initialize()

        # Convert to float32
        img_float = self._to_float32(image)

        # Convert to HSV for saturation adjustment
        if len(img_float.shape) == 3:
            hsv = self._to_float32(
                cv2.cvtColor(
                    self._to_uint8(img_float),
                    cv2.COLOR_RGB2HSV
                )
            )
            hsv[..., 1] *= saturation
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
            img_float = self._to_float32(
                cv2.cvtColor(
                    self._to_uint8(hsv),
                    cv2.COLOR_HSV2RGB
                )
            )

        # Apply temperature (blue-yellow balance)
        if temperature != 0:
            temp_matrix = np.array([
                [1 + temperature * 0.1, 0, 0],
                [0, 1, 0],
                [0, 0, 1 - temperature * 0.1]
            ])
            img_float = self._to_float32(
                cv2.transform(
                    self._to_uint8(img_float),
                    temp_matrix
                )
            )

        # Apply tint (green-magenta balance)
        if tint != 0:
            tint_matrix = np.array([
                [1, 0, 0],
                [0, 1 + tint * 0.1, 0],
                [0, 0, 1 - tint * 0.1]
            ])
            img_float = self._to_float32(
                cv2.transform(
                    self._to_uint8(img_float),
                    tint_matrix
                )
            )

        # Clip values and convert back to uint8
        return self._to_uint8(img_float)

    def apply_vignette(
        self,
        image: ImageArray,
        strength: float = 0.5,
        radius: float = 1.0
    ) -> ImageArray:
        """Apply vignette effect to image."""
        if not self._initialized:
            self.initialize()

        # Create vignette mask
        height, width = image.shape[:2]
        center_x = width / 2
        center_y = height / 2
        X = np.arange(width)
        Y = np.arange(height)
        X, Y = np.meshgrid(X, Y)
        
        # Calculate distances from center
        distances = np.sqrt(
            ((X - center_x) / (width / 2)) ** 2 +
            ((Y - center_y) / (height / 2)) ** 2
        )
        
        # Create mask with smooth falloff
        mask = np.clip((1 - distances * radius) ** (2 / strength), 0, 1)
        
        # Expand mask to match image channels
        if len(image.shape) == 3:
            mask = np.dstack([mask] * image.shape[2])
        
        # Apply mask
        return self._to_uint8(self._to_float32(image) * mask) 