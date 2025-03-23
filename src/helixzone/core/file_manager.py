"""
File manager for optimized loading and saving of images.

This module provides functions for loading and saving images in various
formats, with support for background processing and progress reporting.
"""

import logging
import os
import io
import time
from typing import Optional, Tuple, Dict, Any, Callable, List, BinaryIO, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path

from PyQt6.QtGui import QImage, QPixmap, QImageReader, QImageWriter
from PyQt6.QtCore import Qt, QSize, QRect
import numpy as np
import cv2
from PIL import Image, ImageQt, ExifTags

from .task_manager import TaskManager, get_task_manager, Task, TaskProgress
from .memory_manager import get_memory_manager
from .tiled_processor import TiledProcessor, get_tiled_processor

try:
    from .format_support import get_format_support, ColorProfile
    HAS_FORMAT_SUPPORT = True
except ImportError:
    HAS_FORMAT_SUPPORT = False

logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported file formats."""
    UNKNOWN = 0
    PNG = 1
    JPEG = 2
    TIFF = 3
    BMP = 4
    GIF = 5
    PSD = 6
    SVG = 7
    WEBP = 8
    RAW = 9
    HDR = 10
    EXR = 11
    HEIF = 12
    AVIF = 13
    
    @staticmethod
    def from_extension(ext: str) -> 'FileFormat':
        """Get file format from extension.
        
        Args:
            ext: File extension (with or without leading dot)
            
        Returns:
            Corresponding FileFormat enum value
        """
        # Remove leading dot if present
        if ext.startswith('.'):
            ext = ext[1:]
        
        # Normalize to lowercase
        ext = ext.lower()
        
        # Map extension to format
        format_map = {
            'png': FileFormat.PNG,
            'jpg': FileFormat.JPEG,
            'jpeg': FileFormat.JPEG,
            'tif': FileFormat.TIFF,
            'tiff': FileFormat.TIFF,
            'bmp': FileFormat.BMP,
            'gif': FileFormat.GIF,
            'psd': FileFormat.PSD,
            'svg': FileFormat.SVG,
            'webp': FileFormat.WEBP,
            'raw': FileFormat.RAW,
            'cr2': FileFormat.RAW,
            'nef': FileFormat.RAW,
            'arw': FileFormat.RAW,
            'dng': FileFormat.RAW,
            'hdr': FileFormat.HDR,
            'exr': FileFormat.EXR,
            'heif': FileFormat.HEIF,
            'heic': FileFormat.HEIF,
            'avif': FileFormat.AVIF
        }
        
        return format_map.get(ext, FileFormat.UNKNOWN)
    
    @staticmethod
    def get_extension(format: 'FileFormat') -> str:
        """Get default extension for a file format.
        
        Args:
            format: FileFormat enum value
            
        Returns:
            Default extension for the format (without leading dot)
        """
        format_map = {
            FileFormat.PNG: 'png',
            FileFormat.JPEG: 'jpg',
            FileFormat.TIFF: 'tiff',
            FileFormat.BMP: 'bmp',
            FileFormat.GIF: 'gif',
            FileFormat.PSD: 'psd',
            FileFormat.SVG: 'svg',
            FileFormat.WEBP: 'webp',
            FileFormat.RAW: 'raw',
            FileFormat.HDR: 'hdr',
            FileFormat.EXR: 'exr',
            FileFormat.HEIF: 'heif',
            FileFormat.AVIF: 'avif'
        }
        
        return format_map.get(format, 'unknown')
    
    @staticmethod
    def get_description(format: 'FileFormat') -> str:
        """Get human-readable description for a file format.
        
        Args:
            format: FileFormat enum value
            
        Returns:
            Human-readable description
        """
        format_map = {
            FileFormat.PNG: 'PNG Image',
            FileFormat.JPEG: 'JPEG Image',
            FileFormat.TIFF: 'TIFF Image',
            FileFormat.BMP: 'Bitmap Image',
            FileFormat.GIF: 'GIF Image',
            FileFormat.PSD: 'Photoshop Document',
            FileFormat.SVG: 'Scalable Vector Graphics',
            FileFormat.WEBP: 'WebP Image',
            FileFormat.RAW: 'Camera Raw Image',
            FileFormat.HDR: 'High Dynamic Range Image',
            FileFormat.EXR: 'OpenEXR Image',
            FileFormat.HEIF: 'High Efficiency Image Format',
            FileFormat.AVIF: 'AV1 Image Format'
        }
        
        return format_map.get(format, 'Unknown Format')
    
    @staticmethod
    def get_filter_string(format: 'FileFormat') -> str:
        """Get file dialog filter string for a format.
        
        Args:
            format: FileFormat enum value
            
        Returns:
            Filter string for file dialogs
        """
        extension = FileFormat.get_extension(format)
        description = FileFormat.get_description(format)
        
        return f"{description} (*.{extension})"
    
    @staticmethod
    def get_all_filters() -> str:
        """Get file dialog filter string for all supported formats.
        
        Returns:
            Combined filter string for file dialogs
        """
        filters = []
        
        # Add all formats filter
        extensions = []
        for format in FileFormat:
            if format != FileFormat.UNKNOWN:
                extensions.append(f"*.{FileFormat.get_extension(format)}")
        
        all_filter = f"All Supported Formats ({' '.join(extensions)})"
        filters.append(all_filter)
        
        # Add individual format filters
        for format in FileFormat:
            if format != FileFormat.UNKNOWN:
                filters.append(FileFormat.get_filter_string(format))
        
        # Add all files filter
        filters.append("All Files (*.*)")
        
        return ";;".join(filters)


@dataclass
class ImageMetadata:
    """Metadata for an image file."""
    width: int
    height: int
    channels: int
    bits_per_channel: int
    has_alpha: bool
    format: FileFormat
    dpi: Tuple[float, float] = (72.0, 72.0)
    exif: Dict[str, Any] = None
    color_profile: Optional[bytes] = None
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image dimensions as (width, height) tuple."""
        return (self.width, self.height)


class FileManager:
    """Manager for file operations with support for optimized loading and saving."""
    
    def __init__(self):
        """Initialize the file manager."""
        self.metadata_cache = {}
        self.recent_files = []
        self.max_recent_files = 10
        
        # Initialize format support if available
        self.format_support = get_format_support() if HAS_FORMAT_SUPPORT else None
        
        # Load recent files
        self._load_recent_files()
        
    def get_metadata(self, path: str, refresh: bool = False) -> Optional[ImageMetadata]:
        """Get metadata for an image file.
        
        Args:
            path: Path to the image file
            refresh: Whether to refresh cached metadata
            
        Returns:
            ImageMetadata instance or None if failed
        """
        # Return cached metadata if available and not refreshing
        if not refresh and path in self.metadata_cache:
            return self.metadata_cache[path]
        
        # Check file existence
        if not os.path.isfile(path):
            logger.error(f"File does not exist: {path}")
            return None
        
        # Try to get metadata using extended format support
        if self.format_support and self.format_support.is_supported_format(path):
            raw_metadata = self.format_support.get_metadata(path)
            
            if raw_metadata:
                # Convert to our ImageMetadata format
                file_format = FileFormat.from_extension(os.path.splitext(path)[1])
                
                metadata = ImageMetadata(
                    width=raw_metadata.get('width', 0),
                    height=raw_metadata.get('height', 0),
                    channels=raw_metadata.get('channels', 0),
                    bits_per_channel=raw_metadata.get('bits_per_channel', 8),
                    has_alpha=raw_metadata.get('has_alpha', False),
                    format=file_format,
                    dpi=raw_metadata.get('dpi', (72.0, 72.0)),
                    exif=raw_metadata.get('exif', None),
                    color_profile=raw_metadata.get('color_profile', None)
                )
                
                # Cache and return the metadata
                self.metadata_cache[path] = metadata
                return metadata
        
        # Fall back to regular metadata extraction
        try:
            # Determine file format based on extension
            file_format = FileFormat.from_extension(os.path.splitext(path)[1])
            
            # Use QImageReader to get basic metadata
            reader = QImageReader(path)
            
            if not reader.canRead():
                logger.error(f"Cannot read image: {path}")
                return None
            
            # Get basic image information
            size = reader.size()
            
            # Get color space information
            image_format = reader.imageFormat()
            has_alpha = image_format in [
                QImage.Format.Format_RGBA8888,
                QImage.Format.Format_ARGB32,
                QImage.Format.Format_RGBA64,
                QImage.Format.Format_RGBA16FPx4
            ]
            
            # Determine bit depth and channels
            if image_format in [QImage.Format.Format_RGB16, QImage.Format.Format_RGBA16]:
                bits_per_channel = 16
            elif image_format in [QImage.Format.Format_RGBA16FPx4, QImage.Format.Format_RGBX16FPx4]:
                bits_per_channel = 16  # Float 16
            elif image_format in [QImage.Format.Format_RGBA32FPx4, QImage.Format.Format_RGBX32FPx4]:
                bits_per_channel = 32  # Float 32
            else:
                bits_per_channel = 8
            
            # Channels: 1 for grayscale, 3 for RGB, 4 for RGBA
            if image_format in [QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16]:
                channels = 1
            elif has_alpha:
                channels = 4
            else:
                channels = 3
            
            # Get DPI information
            dpi_x = reader.logicalDpiX()
            dpi_y = reader.logicalDpiY()
            
            # Create metadata object
            metadata = ImageMetadata(
                width=size.width(),
                height=size.height(),
                channels=channels,
                bits_per_channel=bits_per_channel,
                has_alpha=has_alpha,
                format=file_format,
                dpi=(dpi_x, dpi_y)
            )
            
            # Try to extract EXIF data (simplified approach)
            try:
                # Load the image to access its text keys
                img = QImage(path)
                
                # Get EXIF as text entries
                exif_data = {}
                for key in img.textKeys():
                    exif_data[key] = img.text(key)
                
                if exif_data:
                    metadata.exif = exif_data
                
                # Extract color profile if present
                profile_data = img.colorProfile()
                if not profile_data.isEmpty():
                    metadata.color_profile = bytes(profile_data.data())
                
            except Exception as e:
                logger.warning(f"Error extracting extended metadata: {str(e)}")
            
            # Cache and return the metadata
            self.metadata_cache[path] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            return None

    def load_image_async(self, path: str, 
                      on_progress: Optional[Callable[[TaskProgress], None]] = None,
                      on_complete: Optional[Callable[[Optional[QImage]], None]] = None,
                      on_error: Optional[Callable[[Exception], None]] = None) -> str:
        """Load an image asynchronously.
        
        Args:
            path: Path to the image file
            on_progress: Callback for progress updates
            on_complete: Callback for completion with the loaded image
            on_error: Callback for errors
            
        Returns:
            Task ID
        """
        task_manager = get_task_manager()
        memory_manager = get_memory_manager()
        
        # Get metadata (for size estimation)
        metadata = self.get_metadata(path)
        
        def _load_task():
            """Task function to load the image."""
            progress = TaskProgress()
            progress.set_message("Preparing to load image...")
            
            try:
                # Check if this is a special format that needs enhanced support
                is_special_format = False
                
                if self.format_support:
                    is_special_format = self.format_support.is_supported_format(path)
                
                if is_special_format:
                    # Handle RAW, HDR, etc. with the format support module
                    progress.set_message("Loading specialized format...")
                    progress.set_indeterminate(True)
                    
                    # Use format support to load the image
                    image = self.format_support.load_image(path)
                    
                    if image is None:
                        raise IOError(f"Failed to load image: {path}")
                    
                    # Add to recent files
                    self._add_recent_file(path)
                    
                    if on_complete:
                        on_complete(image)
                    
                    progress.set_message("Image loaded successfully")
                    progress.set_percent(100)
                    return image
                
                # Regular image loading
                # Determine if we should use tiled loading for large images
                use_tiled_loading = False
                estimated_memory = 0
                
                if metadata:
                    # Estimate memory requirement (width * height * channels * bytes per channel)
                    channels = max(metadata.channels, 3)  # At least RGB
                    bytes_per_channel = max(1, metadata.bits_per_channel // 8)
                    estimated_memory = metadata.width * metadata.height * channels * bytes_per_channel
                    
                    # Use tiled loading if the image is large (> 100MB)
                    use_tiled_loading = estimated_memory > 100 * 1024 * 1024
                
                # Reserve memory if we know the size
                if estimated_memory > 0:
                    if not memory_manager.reserve(estimated_memory):
                        logger.warning(f"Low memory for loading image: {path}")
                        # Continue anyway, but this may cause problems
                
                if use_tiled_loading:
                    # For large images, use tiled loading
                    progress.set_message("Loading large image in tiles...")
                    
                    # This is a simplified approach - a full implementation would use
                    # a TiledProcessor to load the image in chunks
                    reader = QImageReader(path)
                    
                    # Set clip rect to load a small preview first
                    if metadata:
                        preview_width = min(metadata.width, 1000)
                        preview_height = min(metadata.height, 1000)
                        preview_rect = QRect(0, 0, preview_width, preview_height)
                        reader.setScaledSize(QSize(preview_width, preview_height))
                    
                    # Load the preview
                    progress.set_message("Loading preview...")
                    progress.set_percent(10)
                    preview = reader.read()
                    
                    # Now load the full image
                    progress.set_message("Loading full image...")
                    reader = QImageReader(path)
                    
                    # Load in chunks reporting progress
                    # (In a real implementation, this would load tiles separately)
                    progress.set_percent(20)
                    full_image = reader.read()
                    
                    if full_image.isNull():
                        error = reader.errorString()
                        raise IOError(f"Failed to load image: {error}")
                    
                    # Add to recent files
                    self._add_recent_file(path)
                    
                    progress.set_message("Image loaded successfully")
                    progress.set_percent(100)
                    
                    if on_complete:
                        on_complete(full_image)
                    
                    return full_image
                    
                else:
                    # For small images, load directly
                    progress.set_message("Loading image...")
                    
                    reader = QImageReader(path)
                    image = reader.read()
                    
                    if image.isNull():
                        error = reader.errorString()
                        raise IOError(f"Failed to load image: {error}")
                    
                    # Add to recent files
                    self._add_recent_file(path)
                    
                    progress.set_message("Image loaded successfully")
                    progress.set_percent(100)
                    
                    if on_complete:
                        on_complete(image)
                    
                    return image
                
            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                
                if on_error:
                    on_error(e)
                
                # Re-raise to mark task as failed
                raise
            
            finally:
                # Release reserved memory
                if estimated_memory > 0:
                    memory_manager.release(estimated_memory)
        
        # Create a task for loading
        return task_manager.create_task(
            "Load Image",
            _load_task,
            on_progress=on_progress
        )
    
    def load_image(self, path: str) -> Optional[QImage]:
        """Load an image synchronously.
        
        Args:
            path: Path to the image file
            
        Returns:
            Loaded QImage or None if loading failed
        """
        try:
            # Get metadata for rough size estimation
            metadata = self.get_metadata(path, refresh=True)
            if not metadata:
                raise ValueError(f"Could not read image metadata for {path}")
            
            # Load the image
            with Image.open(path) as img:
                # Convert to QImage
                qimage = ImageQt.ImageQt(img)
                
                # Add to recent files
                self._add_recent_file(path)
                
                return qimage
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def save_image_async(self, image: QImage, path: str, format: Optional[FileFormat] = None, 
                       quality: int = 90,
                       options: Optional[Dict[str, Any]] = None,
                       on_progress: Optional[Callable[[TaskProgress], None]] = None,
                       on_complete: Optional[Callable[[bool], None]] = None,
                       on_error: Optional[Callable[[Exception], None]] = None) -> str:
        """Save an image asynchronously.
        
        Args:
            image: QImage to save
            path: Path to save to
            format: File format (determined from path if None)
            quality: Quality for lossy formats (0-100)
            options: Additional save options
            on_progress: Callback for progress updates
            on_complete: Callback for completion with success flag
            on_error: Callback for errors
            
        Returns:
            Task ID
        """
        task_manager = get_task_manager()
        
        # Determine format from path if not specified
        if format is None:
            ext = os.path.splitext(path)[1].lower()
            format = FileFormat.from_extension(ext)
        
        # Default options if none provided
        if options is None:
            options = {}
        
        def _save_task():
            """Task function to save the image."""
            progress = TaskProgress()
            progress.set_message("Preparing to save image...")
            
            try:
                # Initialize writer
                writer = QImageWriter(path)
                
                # Handle color profile preservation
                preserve_color_profile = options.get('preserve_color_profile', True)
                if preserve_color_profile:
                    # Check if image has a color profile
                    profile_data = image.colorProfile()
                    if not profile_data.isEmpty():
                        writer.setText("ColorSpace", "ICC")
                        # The color profile will be automatically included
                
                # Handle metadata preservation
                preserve_metadata = options.get('preserve_metadata', True)
                preserve_exif = options.get('preserve_exif', True)
                
                if preserve_metadata and preserve_exif:
                    # Copy EXIF and other metadata from the original image
                    for key in image.textKeys():
                        writer.setText(key, image.text(key))
                        
                    # Add additional metadata if specified
                    if 'author' in options and options['author']:
                        writer.setText("Author", options['author'])
                        
                    if 'copyright' in options and options['copyright']:
                        writer.setText("Copyright", options['copyright'])
                
                # Set format-specific options
                if format == FileFormat.JPEG:
                    writer.setQuality(quality)
                    
                    # JPEG progressive option
                    if 'jpeg_progressive' in options:
                        writer.setProgressiveScanWrite(options['jpeg_progressive'])
                        
                    # JPEG optimization option
                    if 'jpeg_optimize' in options:
                        writer.setOptimizedWrite(options['jpeg_optimize'])
                        
                elif format == FileFormat.PNG:
                    # PNG compression level
                    if 'png_compression' in options:
                        compression_level = options['png_compression']
                        # Map to 0-9 range
                        if compression_level == 0:  # Default
                            writer.setCompression(5)
                        elif compression_level == 1:  # Fast
                            writer.setCompression(3)
                        else:  # Best compression
                            writer.setCompression(9)
                    
                    # PNG interlacing
                    if 'png_interlaced' in options:
                        writer.setTransformation(
                            QImageWriter.Transformation.TransformationFlag.Transformation_PremultiplyAlpha
                        )
                
                elif format == FileFormat.WEBP:
                    writer.setQuality(quality)
                    
                    # WebP lossless option
                    if 'webp_lossless' in options:
                        if options['webp_lossless']:
                            # For lossless WebP, set quality to 100
                            writer.setQuality(100)
                
                # Set quality for other formats that support it
                elif format in [FileFormat.HEIF, FileFormat.AVIF]:
                    writer.setQuality(quality)
                
                # Check if we need to resize the image
                use_original_size = options.get('use_original_size', True)
                if not use_original_size:
                    width = options.get('width', image.width())
                    height = options.get('height', image.height())
                    maintain_aspect_ratio = options.get('maintain_aspect_ratio', True)
                    
                    if maintain_aspect_ratio:
                        # Calculate aspect ratio
                        original_aspect = image.width() / float(image.height())
                        new_aspect = width / float(height)
                        
                        if original_aspect > new_aspect:
                            # Width constrained
                            height = int(width / original_aspect)
                        else:
                            # Height constrained
                            width = int(height * original_aspect)
                    
                    # Only resize if the dimensions are different
                    if width != image.width() or height != image.height():
                        progress.set_message("Resizing image...")
                        progress.set_percent(10)
                        
                        # Resize the image with high quality
                        resized_image = image.scaled(
                            width, height,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        image = resized_image
                
                # Special handling for HDR formats
                if format in [FileFormat.HDR, FileFormat.EXR] and self.format_support:
                    # Export with extended format support
                    progress.set_message(f"Saving as {format.name}...")
                    progress.set_indeterminate(True)
                    
                    # Extended format support will handle HDR export
                    success = self.format_support.save_image(image, path, format, options)
                    
                    if not success:
                        raise IOError(f"Failed to save {format.name} image")
                    
                    progress.set_message("Image saved successfully")
                    progress.set_percent(100)
                    
                    if on_complete:
                        on_complete(True)
                    
                    return True
                
                # Standard save process
                progress.set_message(f"Saving image as {format.name}...")
                progress.set_percent(50)
                
                success = writer.write(image)
                
                if not success:
                    error = writer.errorString()
                    raise IOError(f"Failed to save image: {error}")
                
                progress.set_message("Image saved successfully")
                progress.set_percent(100)
                
                if on_complete:
                    on_complete(True)
                
                return True
            
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                
                if on_error:
                    on_error(e)
                
                # Re-raise to mark task as failed
                raise
        
        # Create a task for saving
        return task_manager.create_task(
            "Save Image",
            _save_task,
            on_progress=on_progress
        )
    
    def save_image(self, image: QImage, path: str, format: Optional[FileFormat] = None, 
                quality: int = 90) -> bool:
        """Save an image synchronously.
        
        Args:
            image: The QImage to save
            path: Path where to save the image
            format: Optional file format (if None, derived from path extension)
            quality: Save quality for lossy formats (0-100)
            
        Returns:
            True if save succeeded, False otherwise
        """
        try:
            # Derive format from path if not specified
            if format is None:
                _, ext = os.path.splitext(path)
                format = FileFormat.from_extension(ext)
            
            # Convert QImage to PIL Image
            buffer = QImage(image)  # Create a copy to be safe
            ptr = buffer.constBits()
            ptr.setsize(buffer.byteCount())
            arr = np.frombuffer(ptr, np.uint8).reshape((buffer.height(), buffer.width(), 4))
            
            # Create PIL Image from array
            pil_image = Image.fromarray(arr)
            
            # Save based on format
            if format == FileFormat.JPEG:
                # Save as JPEG with quality setting
                pil_image.save(path, format='JPEG', quality=quality)
            elif format == FileFormat.PNG:
                # Save as PNG with compression level
                compression = 9 - (quality // 10)  # Convert quality (0-100) to compression (9-0)
                pil_image.save(path, format='PNG', compress_level=compression)
            elif format == FileFormat.TIFF:
                # Save as TIFF
                pil_image.save(path, format='TIFF')
            elif format == FileFormat.BMP:
                # Save as BMP
                pil_image.save(path, format='BMP')
            elif format == FileFormat.GIF:
                # Save as GIF
                pil_image.save(path, format='GIF')
            elif format == FileFormat.WEBP:
                # Save as WebP with quality setting
                pil_image.save(path, format='WEBP', quality=quality)
            else:
                # Default to PNG for unsupported formats
                pil_image.save(path, format='PNG')
            
            # Add to recent files
            self._add_recent_file(path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def get_recent_files(self) -> List[str]:
        """Get list of recently opened files.
        
        Returns:
            List of file paths, most recent first
        """
        return self.recent_files.copy()
    
    def clear_recent_files(self) -> None:
        """Clear the list of recent files."""
        self.recent_files = []
    
    def _add_recent_file(self, path: str) -> None:
        """Add a file to the recent files list.
        
        Args:
            path: Path to the file
        """
        # Remove the path if it already exists
        if path in self.recent_files:
            self.recent_files.remove(path)
            
        # Add to the front of the list
        self.recent_files.insert(0, path)
        
        # Trim the list if needed
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
    
    def _load_recent_files(self) -> None:
        """Load the list of recent files from configuration."""
        try:
            # Get application config 
            from ..core.config import load_config
            config = load_config()
            
            # Get recent files from config if available
            if 'recent_files' in config and isinstance(config['recent_files'], list):
                self.recent_files = [
                    path for path in config['recent_files'] 
                    if isinstance(path, str) and os.path.exists(path)
                ][:self.max_recent_files]
            
            logger.debug(f"Loaded {len(self.recent_files)} recent files")
        except Exception as e:
            logger.warning(f"Failed to load recent files: {e}")
            self.recent_files = []


# Singleton pattern
_file_manager = None

def get_file_manager() -> FileManager:
    """Get the global file manager instance.
    
    Returns:
        The file manager instance
    """
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager 