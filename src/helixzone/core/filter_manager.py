"""Filter management for image operations.

This module manages filter operations on images, handling both
synchronous and asynchronous filtering with progress reporting.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
import numpy as np
import cv2

from .task_manager import task_manager, Task, TaskStatus
from .async_processor import (
    async_apply_filter, 
    async_gaussian_blur,
    async_edge_detection,
    async_histogram_equalization,
    async_threshold
)
from .image_processing import ColorSpaceConverter
from ..gui.progress_dialog import TaskProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


class FilterResult:
    """Result of a filter operation, with original and filtered images."""
    
    def __init__(
        self,
        original_image: np.ndarray,
        filtered_image: np.ndarray,
        filter_name: str,
        parameters: Dict[str, Any]
    ):
        """Initialize filter result.
        
        Args:
            original_image: Original image
            filtered_image: Filtered image
            filter_name: Name of the filter applied
            parameters: Parameters used for the filter
        """
        self.original_image = original_image
        self.filtered_image = filtered_image
        self.filter_name = filter_name
        self.parameters = parameters
        self.creation_time = time.time()
        self.id = str(uuid.uuid4())
    
    @property
    def has_result(self) -> bool:
        """Check if there is a valid filtered result.
        
        Returns:
            True if there is a valid filtered result
        """
        return (
            self.filtered_image is not None and 
            self.filtered_image.size > 0
        )


class FilterManager:
    """Manages filter operations with progress reporting and caching.
    
    Features:
    - Applies filters to images
    - Caches recent filter results
    - Supports both synchronous and asynchronous operation
    - Reports progress for long-running operations
    - Manages filter history
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'FilterManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = FilterManager()
        return cls._instance
    
    def __init__(self, max_cache_size: int = 10):
        """Initialize filter manager.
        
        Args:
            max_cache_size: Maximum number of filter results to cache
        """
        self._cache: Dict[str, FilterResult] = {}
        self._max_cache_size = max_cache_size
        self._recent_filters: List[str] = []
        self._active_filter_tasks: Set[str] = set()
        self._progress_tracker = TaskProgressTracker(title="Applying Filter")
        
        logger.info(f"Filter manager initialized with cache size {max_cache_size}")
    
    def _cache_result(self, result: FilterResult) -> None:
        """Cache a filter result.
        
        Args:
            result: Filter result to cache
        """
        # Add to cache
        self._cache[result.id] = result
        
        # Add to recent filters
        if result.id in self._recent_filters:
            self._recent_filters.remove(result.id)
        self._recent_filters.append(result.id)
        
        # Trim cache if needed
        if len(self._recent_filters) > self._max_cache_size:
            oldest_id = self._recent_filters.pop(0)
            if oldest_id in self._cache:
                del self._cache[oldest_id]
        
        logger.debug(f"Cached filter result {result.id} ({result.filter_name})")
    
    def get_cached_result(self, result_id: str) -> Optional[FilterResult]:
        """Get a cached filter result.
        
        Args:
            result_id: ID of the filter result
            
        Returns:
            Filter result or None if not found
        """
        return self._cache.get(result_id)
    
    def clear_cache(self) -> None:
        """Clear the filter cache."""
        self._cache.clear()
        self._recent_filters.clear()
        logger.info("Filter cache cleared")
    
    def apply_filter_async(
        self,
        image: np.ndarray,
        filter_name: str,
        parameters: Dict[str, Any],
        on_complete: Optional[Callable[[FilterResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        show_progress: bool = True
    ) -> str:
        """Apply a filter asynchronously.
        
        Args:
            image: Image to filter
            filter_name: Name of the filter to apply
            parameters: Parameters for the filter
            on_complete: Callback when filter completes
            on_error: Callback when filter fails
            show_progress: Whether to show a progress dialog
            
        Returns:
            Task ID for the filter operation
        """
        # Handle empty images
        if image is None or image.size == 0:
            logger.warning("Empty image passed to apply_filter_async")
            if on_error:
                on_error(ValueError("Empty image"))
            return ""
        
        # Make a copy of the image
        original_image = image.copy()
        
        # Create a callback for when the filter completes
        def filter_complete(filtered_image: np.ndarray) -> None:
            """Handle filter completion."""
            # Create filter result
            result = FilterResult(
                original_image=original_image,
                filtered_image=filtered_image,
                filter_name=filter_name,
                parameters=parameters.copy()
            )
            
            # Cache the result
            self._cache_result(result)
            
            # Remove from active tasks
            if task_id in self._active_filter_tasks:
                self._active_filter_tasks.remove(task_id)
            
            # Call the completion callback
            if on_complete:
                on_complete(result)
            
            logger.info(f"Filter {filter_name} completed (task: {task_id})")
        
        # Create a callback for when the filter fails
        def filter_error(error: Exception) -> None:
            """Handle filter error."""
            # Remove from active tasks
            if task_id in self._active_filter_tasks:
                self._active_filter_tasks.remove(task_id)
            
            # Call the error callback
            if on_error:
                on_error(error)
            
            logger.error(f"Filter {filter_name} failed: {error}")
        
        # Select the appropriate filter based on name
        task_id = ""
        
        try:
            if filter_name == "gaussian_blur":
                # Extract parameters
                kernel_size = parameters.get("kernel_size", 5)
                sigma = parameters.get("sigma", 0.0)
                
                # Apply filter
                task_id = async_gaussian_blur(
                    original_image,
                    kernel_size=kernel_size,
                    sigma=sigma,
                    task_name=f"Gaussian Blur (k={kernel_size})",
                    on_complete=filter_complete,
                    on_error=filter_error
                )
            
            elif filter_name == "edge_detection":
                # Extract parameters
                low_threshold = parameters.get("low_threshold", 50)
                high_threshold = parameters.get("high_threshold", 150)
                aperture_size = parameters.get("aperture_size", 3)
                
                # Apply filter
                task_id = async_edge_detection(
                    original_image,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    aperture_size=aperture_size,
                    task_name=f"Edge Detection ({low_threshold}-{high_threshold})",
                    on_complete=filter_complete,
                    on_error=filter_error
                )
            
            elif filter_name == "histogram_equalization":
                # Extract parameters
                clip_limit = parameters.get("clip_limit", 2.0)
                tile_grid_size = parameters.get("tile_grid_size", (8, 8))
                
                # Apply filter
                task_id = async_histogram_equalization(
                    original_image,
                    clip_limit=clip_limit,
                    tile_grid_size=tile_grid_size,
                    task_name=f"Histogram Equalization (clip={clip_limit})",
                    on_complete=filter_complete,
                    on_error=filter_error
                )
            
            elif filter_name == "threshold":
                # Extract parameters
                threshold_value = parameters.get("threshold_value", 127)
                max_value = parameters.get("max_value", 255)
                threshold_type = parameters.get("threshold_type", cv2.THRESH_BINARY)
                
                # Apply filter
                task_id = async_threshold(
                    original_image,
                    threshold_value=threshold_value,
                    max_value=max_value,
                    threshold_type=threshold_type,
                    task_name=f"Threshold (value={threshold_value})",
                    on_complete=filter_complete,
                    on_error=filter_error
                )
            
            elif filter_name == "custom":
                # Extract parameters
                filter_func = parameters.get("filter_func")
                filter_args = parameters.get("filter_args", ())
                filter_kwargs = parameters.get("filter_kwargs", {})
                
                if not filter_func:
                    raise ValueError("Custom filter requires filter_func parameter")
                
                # Apply filter
                task_id = async_apply_filter(
                    original_image,
                    filter_func,
                    filter_args=filter_args,
                    filter_kwargs=filter_kwargs,
                    task_name=f"Custom Filter ({filter_func.__name__})",
                    on_complete=filter_complete,
                    on_error=filter_error
                )
            
            else:
                # Unknown filter
                raise ValueError(f"Unknown filter: {filter_name}")
            
            # Add to active tasks
            self._active_filter_tasks.add(task_id)
            
            # Show progress dialog if requested
            if show_progress:
                self._progress_tracker.track_task(task_id)
            
            logger.info(f"Started async filter {filter_name} (task: {task_id})")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error starting filter {filter_name}: {e}")
            if on_error:
                on_error(e)
            return ""
    
    def apply_filter(
        self,
        image: np.ndarray,
        filter_name: str,
        parameters: Dict[str, Any],
        show_progress: bool = True
    ) -> Optional[FilterResult]:
        """Apply a filter synchronously, with a blocking operation.
        
        Args:
            image: Image to filter
            filter_name: Name of the filter to apply
            parameters: Parameters for the filter
            show_progress: Whether to show a progress dialog
            
        Returns:
            Filter result or None if operation failed
        """
        result = [None]
        error = [None]
        done = [False]
        
        # Create callbacks
        def on_complete(filter_result: FilterResult) -> None:
            """Handle filter completion."""
            result[0] = filter_result
            done[0] = True
        
        def on_error(e: Exception) -> None:
            """Handle filter error."""
            error[0] = e
            done[0] = True
        
        # Start the filter
        task_id = self.apply_filter_async(
            image,
            filter_name,
            parameters,
            on_complete=on_complete,
            on_error=on_error,
            show_progress=show_progress
        )
        
        if not task_id:
            logger.error("Failed to start filter")
            return None
        
        # Wait for the filter to complete
        while not done[0]:
            # Get task status
            task = task_manager.get_task(task_id)
            if task is None:
                logger.error(f"Task {task_id} not found")
                return None
            
            # Check if task is complete
            if task.is_complete:
                # Task complete but callbacks not called yet
                if task.status == TaskStatus.FAILED and task.error:
                    error[0] = task.error
                done[0] = True
                break
            
            # Sleep a bit
            time.sleep(0.1)
        
        # Check for error
        if error[0]:
            logger.error(f"Filter failed: {error[0]}")
            return None
        
        return result[0]
    
    def cancel_filter(self, task_id: str) -> bool:
        """Cancel a filter operation.
        
        Args:
            task_id: ID of the filter task
            
        Returns:
            True if cancelled, False if already complete or not found
        """
        # Cancel the task
        result = task_manager.cancel_task(task_id)
        
        # Remove from active tasks
        if task_id in self._active_filter_tasks:
            self._active_filter_tasks.remove(task_id)
        
        return result
    
    def cancel_all_filters(self) -> int:
        """Cancel all active filter operations.
        
        Returns:
            Number of filters cancelled
        """
        cancelled = 0
        for task_id in list(self._active_filter_tasks):
            if self.cancel_filter(task_id):
                cancelled += 1
        
        return cancelled
    
    def get_filter_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a filter operation.
        
        Args:
            task_id: ID of the filter task
            
        Returns:
            Task status or None if not found
        """
        task = task_manager.get_task(task_id)
        if task is None:
            return None
        return task.status
    
    def get_filter_names(self) -> List[str]:
        """Get list of available filter names.
        
        Returns:
            List of filter names supported by the filter manager
        """
        return [
            "gaussian_blur",
            "edge_detection",
            "histogram_equalization",
            "threshold",
            "custom"
        ]
    
    def has_filter(self, filter_name: str) -> bool:
        """Check if a filter is supported.
        
        Args:
            filter_name: Name of the filter to check
            
        Returns:
            True if the filter is supported, False otherwise
        """
        return filter_name in self.get_filter_names()
    
    def validate_filter_params(self, filter_name: str, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters for a specific filter.
        
        Args:
            filter_name: Name of the filter
            parameters: Parameters to validate
            
        Returns:
            List of error messages, empty if valid
        """
        errors = []
        
        if not self.has_filter(filter_name):
            errors.append(f"Unknown filter: {filter_name}")
            return errors
        
        # Validate based on filter type
        if filter_name == "gaussian_blur":
            # Check kernel size
            kernel_size = parameters.get("kernel_size", 5)
            if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
                errors.append("Kernel size must be a positive odd integer")
            
            # Check sigma
            sigma = parameters.get("sigma", 0.0)
            if not isinstance(sigma, (int, float)) or sigma < 0:
                errors.append("Sigma must be a non-negative number")
        
        elif filter_name == "edge_detection":
            # Check thresholds
            low_threshold = parameters.get("low_threshold", 50)
            if not isinstance(low_threshold, int) or low_threshold < 0 or low_threshold > 255:
                errors.append("Low threshold must be an integer between 0 and 255")
            
            high_threshold = parameters.get("high_threshold", 150)
            if not isinstance(high_threshold, int) or high_threshold < 0 or high_threshold > 255:
                errors.append("High threshold must be an integer between 0 and 255")
            
            if low_threshold >= high_threshold:
                errors.append("Low threshold must be less than high threshold")
            
            # Check aperture size
            aperture_size = parameters.get("aperture_size", 3)
            if not isinstance(aperture_size, int) or aperture_size not in [3, 5, 7]:
                errors.append("Aperture size must be 3, 5, or 7")
        
        elif filter_name == "histogram_equalization":
            # Check clip limit
            clip_limit = parameters.get("clip_limit", 2.0)
            if not isinstance(clip_limit, (int, float)) or clip_limit < 0:
                errors.append("Clip limit must be a non-negative number")
            
            # Check tile grid size
            tile_grid_size = parameters.get("tile_grid_size", (8, 8))
            if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2:
                errors.append("Tile grid size must be a tuple of two integers")
            else:
                if not all(isinstance(x, int) and x > 0 for x in tile_grid_size):
                    errors.append("Tile grid dimensions must be positive integers")
        
        elif filter_name == "threshold":
            # Check threshold value
            threshold_value = parameters.get("threshold_value", 127)
            if not isinstance(threshold_value, int) or threshold_value < 0 or threshold_value > 255:
                errors.append("Threshold value must be an integer between 0 and 255")
            
            # Check max value
            max_value = parameters.get("max_value", 255)
            if not isinstance(max_value, int) or max_value < 0 or max_value > 255:
                errors.append("Max value must be an integer between 0 and 255")
        
        elif filter_name == "custom":
            # Check filter function
            filter_func = parameters.get("filter_func")
            if not filter_func or not callable(filter_func):
                errors.append("Custom filter requires a callable filter_func")
        
        return errors
    
    def get_active_filters(self) -> List[Task]:
        """Get all active filter tasks.
        
        Returns:
            List of active filter tasks
        """
        return [
            task_manager.get_task(task_id)
            for task_id in self._active_filter_tasks
            if task_manager.get_task(task_id) is not None
        ]


# Create global filter manager
filter_manager = FilterManager.get_instance() 