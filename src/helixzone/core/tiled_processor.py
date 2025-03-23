"""Tiled processing for efficient handling of large images.

This module provides functionality for processing large images by breaking them
into smaller tiles, which helps reduce memory usage and improve performance.
"""

import numpy as np
import cv2
from typing import Callable, Dict, Tuple, List, Any, Optional, Union, Iterator, TypeVar
from dataclasses import dataclass
import logging
from numpy.typing import NDArray
import threading
from concurrent.futures import ThreadPoolExecutor
from .memory_manager import memory_manager

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generic tile processing
T = TypeVar('T')
ImageType = NDArray[np.uint8]
TileFunction = Callable[[ImageType, Dict[str, Any]], ImageType]
ResultType = TypeVar('ResultType')

@dataclass
class Tile:
    """Represents a tile within a larger image."""
    x: int
    y: int
    width: int
    height: int
    data: Optional[ImageType] = None
    
    @property
    def slice(self) -> Tuple[slice, slice]:
        """Get the slice for this tile."""
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of this tile."""
        return (self.height, self.width)
    
    @property
    def size(self) -> int:
        """Get the number of pixels in this tile."""
        return self.width * self.height
    
    def __repr__(self) -> str:
        return f"Tile(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


class TiledProcessor:
    """Process large images by breaking them into smaller tiles.
    
    This class allows processing of large images that might not fit in memory
    by splitting them into smaller tiles, processing each tile independently,
    and then combining the results.
    """
    
    def __init__(
        self,
        max_tile_size_mb: float = 100.0,
        min_tile_size: int = 256,
        max_tile_size: int = 2048,
        overlap: int = 16
    ):
        """Initialize the tiled processor.
        
        Args:
            max_tile_size_mb: Maximum tile size in megabytes
            min_tile_size: Minimum tile width/height in pixels
            max_tile_size: Maximum tile width/height in pixels
            overlap: Number of pixels to overlap between tiles
        """
        self.max_tile_size_mb = max_tile_size_mb
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size
        self.overlap = overlap
        self._thread_executor = ThreadPoolExecutor(max_workers=8)
    
    def _calculate_optimal_tile_size(
        self,
        image_shape: Tuple[int, ...],
        channels: int = 4,
        bytes_per_pixel: int = 4
    ) -> Tuple[int, int]:
        """Calculate the optimal tile size based on memory constraints.
        
        Args:
            image_shape: Shape of the image (height, width, [channels])
            channels: Number of channels in the image
            bytes_per_pixel: Bytes per pixel (e.g., 4 for float32, 1 for uint8)
            
        Returns:
            Optimal tile size as (width, height)
        """
        # Get total image size
        height, width = image_shape[:2]
        
        # Calculate bytes per pixel including channels
        bpp = bytes_per_pixel * channels
        
        # Calculate max pixels per tile
        max_pixels = (self.max_tile_size_mb * 1024 * 1024) / bpp
        
        # Calculate tile dimensions (try to keep tiles square)
        tile_dim = int(np.sqrt(max_pixels))
        tile_dim = max(self.min_tile_size, min(tile_dim, self.max_tile_size))
        
        return (tile_dim, tile_dim)
    
    def generate_tiles(
        self,
        image_shape: Tuple[int, ...],
        tile_size: Optional[Tuple[int, int]] = None
    ) -> List[Tile]:
        """Generate tiles for processing an image.
        
        Args:
            image_shape: Shape of the image (height, width, [channels])
            tile_size: Optional tile size override (width, height)
            
        Returns:
            List of tiles
        """
        height, width = image_shape[:2]
        channels = 3 if len(image_shape) > 2 else 1
        
        # Determine tile size
        if tile_size is None:
            tile_width, tile_height = self._calculate_optimal_tile_size(
                image_shape, channels, 4  # Assume worst case of float32
            )
        else:
            tile_width, tile_height = tile_size
        
        # Calculate number of tiles in each dimension
        tiles_x = max(1, (width + tile_width - 1) // tile_width)
        tiles_y = max(1, (height + tile_height - 1) // tile_height)
        
        logger.info(f"Processing image {width}x{height} in {tiles_x}x{tiles_y} tiles "
                   f"of size {tile_width}x{tile_height}")
        
        # Generate tiles
        tiles = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile dimensions with overlap
                tile_x = max(0, x * tile_width - self.overlap if x > 0 else 0)
                tile_y = max(0, y * tile_height - self.overlap if y > 0 else 0)
                
                # Calculate right and bottom edges with overlap
                right_overlap = self.overlap if x < tiles_x - 1 else 0
                bottom_overlap = self.overlap if y < tiles_y - 1 else 0
                
                tile_right = min(width, (x + 1) * tile_width + right_overlap)
                tile_bottom = min(height, (y + 1) * tile_height + bottom_overlap)
                
                # Create tile
                tile = Tile(
                    x=tile_x,
                    y=tile_y,
                    width=tile_right - tile_x,
                    height=tile_bottom - tile_y
                )
                tiles.append(tile)
        
        return tiles
    
    def load_tile_data(self, image: NDArray[np.uint8], tile: Tile) -> Tile:
        """Load data for a specific tile from the image.
        
        Args:
            image: Source image
            tile: Tile to load data for
            
        Returns:
            Tile with data loaded
        """
        # Copy the tile data from the image
        tile.data = image[tile.slice].copy()
        return tile
    
    def process_image(
        self,
        image: NDArray[np.uint8],
        process_fn: TileFunction,
        process_args: Dict[str, Any],
        tile_size: Optional[Tuple[int, int]] = None,
        parallel: bool = True
    ) -> NDArray[np.uint8]:
        """Process a large image in tiles.
        
        Args:
            image: Input image
            process_fn: Function to process each tile
            process_args: Arguments to pass to the processing function
            tile_size: Optional tile size override
            parallel: Whether to process tiles in parallel
            
        Returns:
            Processed image
        """
        # Validate input
        if image is None or image.size == 0:
            logger.error("Cannot process empty image")
            return np.zeros((1, 1), dtype=np.uint8)
            
        # Track memory usage
        with memory_manager.monitor_allocation("tiled_processing"):
            # Generate tiles
            tiles = self.generate_tiles(image.shape, tile_size)
            
            # Create output image
            output = np.zeros_like(image)
            
            # Process tiles
            if parallel:
                # Process tiles in parallel
                futures = []
                for tile in tiles:
                    # Load tile data
                    tile_with_data = self.load_tile_data(image, tile)
                    
                    # Submit processing task
                    future = self._thread_executor.submit(
                        self._process_tile,
                        tile_with_data,
                        process_fn,
                        process_args
                    )
                    futures.append((future, tile))
                
                # Collect results
                for future, tile in futures:
                    processed_tile = future.result()
                    
                    # Remove overlap if needed
                    # For now we use the full tile
                    if processed_tile.data is not None:
                        output[tile.slice] = processed_tile.data
            else:
                # Process tiles sequentially
                for tile in tiles:
                    # Load tile data
                    tile_with_data = self.load_tile_data(image, tile)
                    
                    # Process tile
                    processed_tile = self._process_tile(
                        tile_with_data, process_fn, process_args
                    )
                    
                    # Copy to output
                    if processed_tile.data is not None:
                        output[tile.slice] = processed_tile.data
            
            return output
    
    def _process_tile(
        self,
        tile: Tile,
        process_fn: TileFunction,
        process_args: Dict[str, Any]
    ) -> Tile:
        """Process a single tile.
        
        Args:
            tile: Tile to process
            process_fn: Function to process the tile
            process_args: Arguments for the processing function
            
        Returns:
            Processed tile
        """
        try:
            if tile.data is None:
                logger.warning(f"Tile has no data: {tile}")
                return tile
                
            # Process the tile
            processed_data = process_fn(tile.data, process_args)
            
            # Update tile data
            tile.data = processed_data
            return tile
        except Exception as e:
            logger.error(f"Error processing tile {tile}: {e}")
            # Return original tile
            return tile
    
    def close(self):
        """Close the thread executor."""
        self._thread_executor.shutdown()


# Create a tiled processor for general use
tiled_processor = TiledProcessor()


def get_tiled_processor() -> TiledProcessor:
    """Get the global tiled processor instance.
    
    Returns:
        The tiled processor instance
    """
    return tiled_processor


# Utility functions for common processing tasks
def process_large_image(
    image: NDArray[np.uint8],
    processor: TileFunction,
    args: Dict[str, Any],
    max_memory_mb: float = 100.0
) -> NDArray[np.uint8]:
    """Process a large image using tiled processing.
    
    Args:
        image: Input image
        processor: Function to process each tile
        args: Arguments for the processing function
        max_memory_mb: Maximum memory per tile in MB
        
    Returns:
        Processed image
    """
    # Create a tiled processor with the specified memory limit
    processor_instance = TiledProcessor(max_tile_size_mb=max_memory_mb)
    
    try:
        # Process the image
        result = processor_instance.process_image(image, processor, args)
        return result
    finally:
        # Clean up
        processor_instance.close()
        
        # Force memory cleanup
        memory_manager.cleanup_unused_memory()


class ProgressiveTiledLoader:
    """Progressively load and process large images.
    
    This class supports loading large images in a progressive manner,
    showing lower resolution versions first and then refining as more
    data is loaded.
    """
    
    def __init__(
        self,
        max_resolution: Tuple[int, int] = (8192, 8192),
        resolution_levels: int = 3
    ):
        """Initialize the progressive loader.
        
        Args:
            max_resolution: Maximum resolution to support
            resolution_levels: Number of resolution levels
        """
        self.max_resolution = max_resolution
        self.resolution_levels = resolution_levels
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._current_task = None
    
    def load_image_progressive(
        self,
        file_path: str,
        callback: Callable[[NDArray[np.uint8], int], None]
    ) -> None:
        """Load an image progressively, calling back with each resolution level.
        
        Args:
            file_path: Path to the image file
            callback: Function called with (image, level) for each resolution level
        """
        with self._lock:
            # Cancel any existing task
            self._cancel_event.set()
            if self._current_task is not None:
                self._current_task.join(timeout=1.0)
            
            # Reset cancel event
            self._cancel_event.clear()
            
            # Start new task
            self._current_task = threading.Thread(
                target=self._load_progressive,
                args=(file_path, callback),
                daemon=True
            )
            self._current_task.start()
    
    def _load_progressive(
        self,
        file_path: str,
        callback: Callable[[NDArray[np.uint8], int], None]
    ) -> None:
        """Internal method to load an image progressively.
        
        Args:
            file_path: Path to the image file
            callback: Function called with (image, level) for each resolution level
        """
        try:
            # First try to get image dimensions without loading the full image
            metadata = cv2.imread(file_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if metadata is None:
                logger.error(f"Failed to read image metadata: {file_path}")
                return
                
            # Determine image dimensions
            full_height, full_width = metadata.shape[:2]
            
            # Check if cancelled
            if self._cancel_event.is_set():
                return
                
            # Calculate resolution levels
            levels = []
            for i in range(self.resolution_levels):
                scale = 1.0 / (2 ** (self.resolution_levels - i - 1))
                width = max(1, int(full_width * scale))
                height = max(1, int(full_height * scale))
                levels.append((width, height, scale))
            
            # Add full resolution
            levels.append((full_width, full_height, 1.0))
            
            # Load each level
            for i, (width, height, scale) in enumerate(levels):
                # Check if cancelled
                if self._cancel_event.is_set():
                    return
                    
                # Load image at this resolution
                if scale < 1.0:
                    # Calculate IMREAD_REDUCED flag
                    # OpenCV supports 1/2, 1/4, or 1/8
                    if scale <= 0.125:
                        reduction = cv2.IMREAD_REDUCED_COLOR_8
                    elif scale <= 0.25:
                        reduction = cv2.IMREAD_REDUCED_COLOR_4
                    else:  # scale <= 0.5
                        reduction = cv2.IMREAD_REDUCED_COLOR_2
                    
                    # Load at reduced resolution
                    img = cv2.imread(file_path, reduction)
                    
                    # Resize to exact target size if needed
                    if img is not None and (img.shape[1] != width or img.shape[0] != height):
                        img = cv2.resize(img, (width, height))
                else:
                    # Load at full resolution
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                
                # Check if loaded
                if img is None:
                    logger.error(f"Failed to load image at level {i}: {file_path}")
                    continue
                
                # Invoke callback
                callback(img, i)
                
        except Exception as e:
            logger.error(f"Error in progressive loading: {e}")
    
    def cancel(self) -> None:
        """Cancel any ongoing loading operation."""
        self._cancel_event.set()
        
    def dispose(self) -> None:
        """Release resources."""
        self.cancel()
        with self._lock:
            if self._current_task is not None:
                self._current_task.join(timeout=1.0)
                self._current_task = None


# Create a global instance
progressive_loader = ProgressiveTiledLoader() 