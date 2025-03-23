"""Advanced image format support for HelixZone.

This module provides support for various image formats beyond what Qt and PIL support 
natively, including camera RAW files and HDR images. The functionality depends on 
several optional dependencies which are imported conditionally.

Optional Dependencies:
    - rawpy: Used for camera RAW format support (CR2, NEF, ARW, etc.)
    - exifread: Used for advanced EXIF metadata handling
    - OpenImageIO: Used for HDR image support (EXR, HDR)
    - colour: Used for advanced color management and transformations
    - colour_demosaicing: Used for advanced RAW demosaicing algorithms

Each dependency is imported within a try-except block and a corresponding HAS_*
flag is set to indicate if the dependency is available. The code will automatically
fall back to basic functionality if these dependencies are not installed.

To install all optional dependencies:
    pip install rawpy exifread OpenImageIO colour colour_demosaicing

Class Overview:
    - RawFormatHandler: Handles camera RAW files using rawpy
    - HdrFormatHandler: Handles HDR image formats using OpenImageIO or OpenCV
    - ColorProfile: Manages color profiles for images
    - ColorManager: Provides color space transformations and management
    - FormatSupport: Main class that coordinates all format handlers
"""

import logging
import os
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from io import BytesIO

from PyQt6.QtGui import QImage, QColor
from PyQt6.QtCore import QByteArray, QBuffer

# Try to import optional dependencies for RAW and HDR support
try:
    import rawpy  # type: ignore
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    import exifread  # type: ignore
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False

try:
    import OpenImageIO as oiio  # type: ignore
    HAS_OPENIMAGEIO = True
except ImportError:
    HAS_OPENIMAGEIO = False

try:
    import colour_demosaicing  # type: ignore
    HAS_COLOUR_DEMOSAICING = True
except ImportError:
    HAS_COLOUR_DEMOSAICING = False

try:
    import cv2
    import colour  # type: ignore
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False

logger = logging.getLogger(__name__)


class RawProcessingMode(Enum):
    """Processing modes for RAW images."""
    LINEAR = auto()  # Linear processing, no tone mapping
    SRGB = auto()    # sRGB output with tone mapping
    CUSTOM = auto()  # Custom processing parameters


@dataclass
class RawProcessingParams:
    """Parameters for RAW image processing."""
    mode: RawProcessingMode = RawProcessingMode.SRGB
    demosaic_algorithm: str = "AHD"  # Adaptive Homogeneity-Directed
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    highlight_recovery: bool = True
    black_level: int = 0
    white_level: Optional[int] = None
    auto_white_balance: bool = True
    white_balance_coeffs: Optional[Tuple[float, float, float]] = None
    output_color_space: str = "sRGB"


class RawFormatHandler:
    """Handler for camera RAW formats."""
    
    # List of supported RAW extensions
    SUPPORTED_EXTENSIONS = [
        ".arw",   # Sony
        ".cr2",   # Canon
        ".cr3",   # Canon
        ".dng",   # Adobe Digital Negative
        ".nef",   # Nikon
        ".orf",   # Olympus
        ".pef",   # Pentax
        ".raf",   # Fujifilm
        ".rw2",   # Panasonic
        ".srw",   # Samsung
        ".x3f",   # Sigma
    ]
    
    @staticmethod
    def is_raw_file(file_path: str) -> bool:
        """Check if a file is a supported RAW format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a supported RAW format
        """
        ext = os.path.splitext(file_path.lower())[1]
        return ext in RawFormatHandler.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from a RAW file.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        if not HAS_RAWPY or not HAS_EXIFREAD:
            logger.warning("RAW support requires rawpy and exifread packages")
            return metadata
        
        try:
            # Extract basic metadata using rawpy
            with rawpy.imread(file_path) as raw:
                metadata['width'] = raw.sizes.width
                metadata['height'] = raw.sizes.height
                metadata['channels'] = 3  # RAW files process to RGB
                metadata['bits_per_channel'] = 16
                metadata['has_alpha'] = False
                
                # Get black and white levels
                metadata['black_level'] = raw.black_level_per_channel[0]
                metadata['white_level'] = raw.white_level
                
                # Get camera model and make
                metadata['camera_model'] = raw.camera_model
                metadata['camera_make'] = raw.camera_make
                
            # Extract EXIF metadata
            if HAS_EXIFREAD:
                with open(file_path, 'rb') as f:
                    exif_tags = exifread.process_file(f, details=False)
                    metadata['exif'] = {}
                    
                    # Extract relevant EXIF tags
                    for tag, value in exif_tags.items():
                        if tag != 'JPEGThumbnail':
                            metadata['exif'][tag] = str(value)
            
        except Exception as e:
            logger.error(f"Error extracting RAW metadata: {str(e)}")
            
        return metadata
    
    @staticmethod
    def load_raw_image(file_path: str, params: Optional[RawProcessingParams] = None) -> Optional[QImage]:
        """Load a RAW image with the specified processing parameters.
        
        Args:
            file_path: Path to the RAW file
            params: Processing parameters
            
        Returns:
            Processed QImage or None if failed
        """
        if not HAS_RAWPY:
            logger.error("RAW support requires rawpy package")
            return None
        
        if params is None:
            params = RawProcessingParams()
        
        try:
            with rawpy.imread(file_path) as raw:
                # Determine the postprocessing settings based on the mode
                if params.mode == RawProcessingMode.LINEAR:
                    # Linear processing (no tone curve)
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        output_bps=16,
                        no_auto_bright=True,
                        use_camera_wb=not params.auto_white_balance,
                        user_wb=params.white_balance_coeffs,
                        bright=params.brightness,
                        demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, params.demosaic_algorithm)
                    )
                else:
                    # sRGB or custom processing with tone curve
                    rgb = raw.postprocess(
                        gamma=(2.2, 4.5),  # Standard sRGB gamma approximation
                        output_bps=8,
                        no_auto_bright=False,
                        use_camera_wb=not params.auto_white_balance,
                        user_wb=params.white_balance_coeffs,
                        bright=params.brightness,
                        demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, params.demosaic_algorithm)
                    )
                
                # Convert to QImage
                height, width, channels = rgb.shape
                bytesPerLine = channels * width
                
                if rgb.dtype == np.uint8:
                    # 8-bit RGB
                    qimg = QImage(rgb.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
                else:
                    # 16-bit RGB, need to convert to 8-bit for QImage
                    rgb_8bit = (rgb / 256).astype(np.uint8)
                    qimg = QImage(rgb_8bit.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
                
                # Return a copy to ensure it's valid after raw is closed
                return qimg.copy()
        
        except Exception as e:
            logger.error(f"Error loading RAW image: {str(e)}")
            return None


class HdrFormatHandler:
    """Handler for HDR image formats."""
    
    # List of supported HDR extensions
    SUPPORTED_EXTENSIONS = [
        ".hdr",   # Radiance HDR
        ".exr",   # OpenEXR
        ".pfm",   # Portable Float Map
    ]
    
    @staticmethod
    def is_hdr_file(file_path: str) -> bool:
        """Check if a file is a supported HDR format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a supported HDR format
        """
        ext = os.path.splitext(file_path.lower())[1]
        return ext in HdrFormatHandler.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from an HDR file.
        
        Args:
            file_path: Path to the HDR file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        if not HAS_OPENIMAGEIO:
            logger.warning("HDR support requires OpenImageIO package")
            return metadata
        
        try:
            # Open the file with OpenImageIO
            input_file = oiio.ImageInput.open(file_path)
            if input_file:
                # Get basic image specs
                spec = input_file.spec()
                metadata['width'] = spec.width
                metadata['height'] = spec.height
                metadata['channels'] = spec.nchannels
                metadata['format'] = str(spec.format)
                metadata['bits_per_channel'] = spec.format.size() * 8
                metadata['has_alpha'] = spec.alpha_channel >= 0
                
                # Get metadata attributes
                for name in spec.extra_attribs:
                    metadata[name] = spec.getattribute(name)
                
                input_file.close()
        
        except Exception as e:
            logger.error(f"Error extracting HDR metadata: {str(e)}")
            
        return metadata
    
    @staticmethod
    def load_hdr_image(file_path: str, tone_map: bool = True, exposure: float = 0.0) -> Optional[QImage]:
        """Load an HDR image with optional tone mapping.
        
        Args:
            file_path: Path to the HDR file
            tone_map: Whether to apply tone mapping
            exposure: Exposure adjustment (in EV)
            
        Returns:
            Processed QImage or None if failed
        """
        # Prefer OpenImageIO if available
        if HAS_OPENIMAGEIO:
            try:
                # Open the HDR file
                input_file = oiio.ImageInput.open(file_path)
                if not input_file:
                    logger.error(f"Could not open HDR file: {file_path}")
                    return None
                
                # Read the image data
                spec = input_file.spec()
                img = oiio.ImageBuf(spec)
                if not img.read(input_file):
                    logger.error(f"Could not read HDR file: {file_path}")
                    input_file.close()
                    return None
                
                # Apply exposure adjustment
                if exposure != 0.0:
                    exposure_factor = 2.0 ** exposure
                    img = oiio.ImageBufAlgo.mul(img, exposure_factor)
                
                # Apply tone mapping if requested
                if tone_map:
                    # Simple Reinhard tone mapping
                    img = oiio.ImageBufAlgo.tonemap_reinhardish(img, intensity=0.5, contrast=1.0)
                
                # Convert to 8-bit RGBA
                rgb_8bit = oiio.ImageBufAlgo.to_format(img, oiio.TypeDesc.UINT8)
                
                # Convert to QImage
                pixels = rgb_8bit.get_pixels(oiio.TypeDesc.UINT8)
                if pixels is None:
                    return None
                
                # Reshape the pixel array
                pixels_np = np.frombuffer(pixels, dtype=np.uint8).reshape(
                    (spec.height, spec.width, spec.nchannels))
                
                # Create QImage (handle RGB vs RGBA)
                if spec.nchannels == 4:
                    qimg = QImage(pixels_np.data, spec.width, spec.height, 
                                 spec.width * 4, QImage.Format.Format_RGBA8888)
                else:
                    # Convert to RGB if it's not already
                    if spec.nchannels != 3:
                        rgb_img = oiio.ImageBufAlgo.channels(rgb_8bit, (0, 1, 2))
                        pixels = rgb_img.get_pixels(oiio.TypeDesc.UINT8)
                        pixels_np = np.frombuffer(pixels, dtype=np.uint8).reshape(
                            (spec.height, spec.width, 3))
                    
                    qimg = QImage(pixels_np.data, spec.width, spec.height, 
                                 spec.width * 3, QImage.Format.Format_RGB888)
                
                input_file.close()
                return qimg.copy()  # Return a copy to ensure it's valid after data is gone
            
            except Exception as e:
                logger.error(f"Error loading HDR image with OpenImageIO: {str(e)}")
                return None
        
        # Fallback to OpenCV if available
        elif HAS_COLOUR:
            try:
                # Read HDR image with OpenCV
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.error(f"Could not open HDR file with OpenCV: {file_path}")
                    return None
                
                # Apply exposure adjustment
                if exposure != 0.0:
                    exposure_factor = 2.0 ** exposure
                    img = img * exposure_factor
                
                # Apply tone mapping if requested
                if tone_map:
                    tone_mapper = cv2.createTonemap(gamma=2.2)
                    img = tone_mapper.process(img)
                
                # Convert to 8-bit
                img_8bit = np.clip(img * 255, 0, 255).astype(np.uint8)
                
                # Convert from BGR to RGB
                img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)
                
                # Create QImage
                height, width, channels = img_rgb.shape
                bytesPerLine = channels * width
                qimg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
                
                return qimg.copy()  # Return a copy to ensure it's valid after data is gone
            
            except Exception as e:
                logger.error(f"Error loading HDR image with OpenCV: {str(e)}")
                return None
        
        else:
            logger.error("HDR support requires OpenImageIO or OpenCV with colour package")
            return None


class ColorProfile:
    """Color profile for managing image color spaces."""
    
    def __init__(self, profile_data: Optional[bytes] = None, name: str = ""):
        """Initialize a color profile.
        
        Args:
            profile_data: ICC profile data as bytes
            name: Profile name
        """
        self.data = profile_data
        self.name = name
        self._is_valid = profile_data is not None and len(profile_data) > 0
    
    @property
    def is_valid(self) -> bool:
        """Check if the profile is valid."""
        return self._is_valid
    
    @staticmethod
    def from_file(file_path: str) -> 'ColorProfile':
        """Load a color profile from an ICC file.
        
        Args:
            file_path: Path to the ICC profile file
            
        Returns:
            ColorProfile instance
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                name = os.path.basename(file_path)
                return ColorProfile(data, name)
        except Exception as e:
            logger.error(f"Error loading ICC profile: {str(e)}")
            return ColorProfile(None, "")
    
    @staticmethod
    def sRGB() -> 'ColorProfile':
        """Get the standard sRGB profile."""
        # We could include a built-in sRGB profile here, or use one provided by Qt
        return ColorProfile(None, "sRGB")
    
    @staticmethod
    def from_qimage(image: QImage) -> 'ColorProfile':
        """Extract color profile from a QImage.
        
        Args:
            image: QImage with possibly embedded color profile
            
        Returns:
            ColorProfile instance
        """
        profile_data = image.colorProfile()
        if profile_data.size() > 0:
            return ColorProfile(profile_data.data(), "Embedded")
        return ColorProfile(None, "")


class ColorManager:
    """Manager for color profiles and conversions."""
    
    def __init__(self):
        """Initialize the color manager."""
        self.profiles = {}
        self._init_default_profiles()
    
    def _init_default_profiles(self):
        """Initialize default color profiles."""
        # Add standard profiles
        self.profiles['sRGB'] = ColorProfile.sRGB()
        
        # TODO: Add more standard profiles
    
    def add_profile(self, name: str, profile: ColorProfile):
        """Add a color profile to the manager.
        
        Args:
            name: Profile name
            profile: ColorProfile instance
        """
        self.profiles[name] = profile
    
    def get_profile(self, name: str) -> Optional[ColorProfile]:
        """Get a color profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            ColorProfile instance or None if not found
        """
        return self.profiles.get(name)
    
    def apply_profile(self, image: QImage, profile: ColorProfile) -> QImage:
        """Apply a color profile to an image.
        
        Args:
            image: QImage to process
            profile: ColorProfile to apply
            
        Returns:
            Processed QImage
        """
        if not profile.is_valid:
            return image
        
        # TODO: Implement proper color conversion
        # For now, just set the color profile on the image
        if profile.data:
            # Create a QByteArray from the profile data
            profile_data = QByteArray(profile.data)
            
            # Create a new image with the profile
            new_image = image.copy()
            new_image.setColorProfile(profile_data)
            return new_image
        
        return image


class FormatSupport:
    """Main class for providing extended format support."""
    
    def __init__(self):
        """Initialize the format support."""
        self.raw_handler = RawFormatHandler()
        self.hdr_handler = HdrFormatHandler()
        self.color_manager = ColorManager()
        
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check and log available dependencies."""
        dependencies = [
            ("RawPy", HAS_RAWPY, "RAW image support"),
            ("ExifRead", HAS_EXIFREAD, "EXIF metadata reading"),
            ("OpenImageIO", HAS_OPENIMAGEIO, "HDR image support"),
            ("Colour", HAS_COLOUR, "Advanced color management"),
            ("Colour Demosaicing", HAS_COLOUR_DEMOSAICING, "Advanced RAW demosaicing")
        ]
        
        logger.info("Format support dependencies:")
        for name, available, purpose in dependencies:
            status = "Available" if available else "Not available"
            logger.info(f"  {name}: {status} ({purpose})")
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if a file is a supported extended format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a supported format
        """
        return (
            self.raw_handler.is_raw_file(file_path) or
            self.hdr_handler.is_hdr_file(file_path)
        )
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of metadata
        """
        if self.raw_handler.is_raw_file(file_path):
            return self.raw_handler.get_metadata(file_path)
        elif self.hdr_handler.is_hdr_file(file_path):
            return self.hdr_handler.get_metadata(file_path)
        return {}
    
    def load_image(self, file_path: str, options: Dict[str, Any] = None) -> Optional[QImage]:
        """Load an image from a file.
        
        Args:
            file_path: Path to the file
            options: Loading options
            
        Returns:
            QImage or None if failed
        """
        if options is None:
            options = {}
        
        if self.raw_handler.is_raw_file(file_path):
            # Parse RAW options
            raw_params = RawProcessingParams()
            
            # Apply custom options if provided
            if 'raw_processing_mode' in options:
                raw_params.mode = options['raw_processing_mode']
            if 'demosaic_algorithm' in options:
                raw_params.demosaic_algorithm = options['demosaic_algorithm']
            if 'brightness' in options:
                raw_params.brightness = options['brightness']
            if 'contrast' in options:
                raw_params.contrast = options['contrast']
            if 'saturation' in options:
                raw_params.saturation = options['saturation']
            if 'highlight_recovery' in options:
                raw_params.highlight_recovery = options['highlight_recovery']
            if 'auto_white_balance' in options:
                raw_params.auto_white_balance = options['auto_white_balance']
            
            return self.raw_handler.load_raw_image(file_path, raw_params)
            
        elif self.hdr_handler.is_hdr_file(file_path):
            # Parse HDR options
            tone_map = options.get('tone_map', True)
            exposure = options.get('exposure', 0.0)
            
            return self.hdr_handler.load_hdr_image(file_path, tone_map, exposure)
            
        return None


# Singleton instance
_instance = None

def get_format_support() -> FormatSupport:
    """Get the singleton FormatSupport instance.
    
    Returns:
        FormatSupport instance
    """
    global _instance
    if _instance is None:
        _instance = FormatSupport()
    return _instance 