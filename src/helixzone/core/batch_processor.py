"""
Batch processing module for handling multiple images or operations efficiently.

This module provides functionality for batch processing multiple images
or operations, with support for parallel processing and progress tracking.
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import time

from PyQt6.QtGui import QImage
from PyQt6.QtCore import QObject, pyqtSignal

from .task_manager import TaskManager, get_task_manager, Task, TaskProgress
from .file_manager import get_file_manager, FileFormat
from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class BatchItemStatus(Enum):
    """Status of a batch item."""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class BatchItem:
    """Represents a single item in a batch operation."""
    id: str
    name: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    operation: Optional[str] = None
    parameters: Dict[str, Any] = None
    status: BatchItemStatus = BatchItemStatus.PENDING
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Any = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}


class BatchOperation:
    """Base class for batch operations."""
    
    name = "Batch Operation"
    description = "Base batch operation"
    
    def process_item(self, item: BatchItem) -> Any:
        """Process a single batch item.
        
        Args:
            item: The batch item to process
            
        Returns:
            The processing result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this operation.
        
        Returns:
            Dictionary describing the parameters for this operation
        """
        return {}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate the parameters for this operation.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of error messages, empty if valid
        """
        return []
    
    def get_estimated_memory(self, item: BatchItem) -> int:
        """Estimate memory requirements for this operation.
        
        Args:
            item: The batch item to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        return 0


class BatchResizeOperation(BatchOperation):
    """Batch resize operation."""
    
    name = "Resize Images"
    description = "Resize multiple images to specified dimensions"
    
    def process_item(self, item: BatchItem) -> QImage:
        """Resize an image to the specified dimensions.
        
        Args:
            item: The batch item to process
            
        Returns:
            Resized QImage
        """
        from PyQt6.QtCore import Qt
        
        file_manager = get_file_manager()
        
        # Load the input image
        input_image = file_manager.load_image(item.input_path)
        if input_image is None:
            raise ValueError(f"Failed to load image: {item.input_path}")
        
        # Get resize parameters
        width = item.parameters.get('width', input_image.width())
        height = item.parameters.get('height', input_image.height())
        keep_aspect_ratio = item.parameters.get('keep_aspect_ratio', True)
        resize_mode = item.parameters.get('resize_mode', 'smooth')
        
        # Determine transformation mode
        if resize_mode == 'fast':
            mode = Qt.TransformationMode.FastTransformation
        else:
            mode = Qt.TransformationMode.SmoothTransformation
        
        # Determine aspect ratio mode
        if keep_aspect_ratio:
            aspect_mode = Qt.AspectRatioMode.KeepAspectRatio
        else:
            aspect_mode = Qt.AspectRatioMode.IgnoreAspectRatio
        
        # Resize the image
        resized_image = input_image.scaled(width, height, aspect_mode, mode)
        
        return resized_image
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this operation.
        
        Returns:
            Dictionary describing the parameters for resize operation
        """
        return {
            'width': {
                'type': 'integer',
                'description': 'Target width in pixels',
                'minimum': 1,
                'maximum': 10000,
                'default': 1024
            },
            'height': {
                'type': 'integer',
                'description': 'Target height in pixels',
                'minimum': 1,
                'maximum': 10000,
                'default': 768
            },
            'keep_aspect_ratio': {
                'type': 'boolean',
                'description': 'Maintain original aspect ratio',
                'default': True
            },
            'resize_mode': {
                'type': 'string',
                'description': 'Quality of the resize operation',
                'enum': ['smooth', 'fast'],
                'default': 'smooth'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate the parameters for resize operation.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of error messages, empty if valid
        """
        errors = []
        
        # Check required parameters
        if 'width' not in parameters:
            errors.append("Width is required")
        elif not isinstance(parameters['width'], int) or parameters['width'] < 1:
            errors.append("Width must be a positive integer")
        
        if 'height' not in parameters:
            errors.append("Height is required")
        elif not isinstance(parameters['height'], int) or parameters['height'] < 1:
            errors.append("Height must be a positive integer")
        
        return errors
    
    def get_estimated_memory(self, item: BatchItem) -> int:
        """Estimate memory requirements for resize operation.
        
        Args:
            item: The batch item to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        # Basic estimation: input image + output image
        file_manager = get_file_manager()
        metadata = file_manager.get_metadata(item.input_path)
        
        if metadata:
            width = item.parameters.get('width', metadata.width)
            height = item.parameters.get('height', metadata.height)
            channels = metadata.channels
            
            # Estimate memory for both input and output images
            input_memory = metadata.width * metadata.height * channels * 4  # 4 bytes per pixel in QImage
            output_memory = width * height * channels * 4
            
            return input_memory + output_memory
        
        # Default estimate if metadata not available
        return 100 * 1024 * 1024  # 100 MB


class BatchFormatConvertOperation(BatchOperation):
    """Batch format conversion operation."""
    
    name = "Convert Format"
    description = "Convert images to a different format"
    
    def process_item(self, item: BatchItem) -> QImage:
        """Convert an image to the specified format.
        
        Args:
            item: The batch item to process
            
        Returns:
            Converted QImage
        """
        file_manager = get_file_manager()
        
        # Load the input image
        input_image = file_manager.load_image(item.input_path)
        if input_image is None:
            raise ValueError(f"Failed to load image: {item.input_path}")
        
        # The format conversion happens during save, not here
        # Just return the loaded image
        return input_image
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this operation.
        
        Returns:
            Dictionary describing the parameters for format conversion
        """
        return {
            'format': {
                'type': 'string',
                'description': 'Target format',
                'enum': ['PNG', 'JPEG', 'TIFF', 'BMP', 'GIF', 'WEBP'],
                'default': 'PNG'
            },
            'quality': {
                'type': 'integer',
                'description': 'Quality for lossy formats (0-100)',
                'minimum': 0,
                'maximum': 100,
                'default': 90
            },
            'preserve_metadata': {
                'type': 'boolean',
                'description': 'Preserve metadata (EXIF, ICC profile)',
                'default': True
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate the parameters for format conversion.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of error messages, empty if valid
        """
        errors = []
        
        # Check format
        if 'format' not in parameters:
            errors.append("Format is required")
        elif parameters['format'] not in ['PNG', 'JPEG', 'TIFF', 'BMP', 'GIF', 'WEBP']:
            errors.append("Invalid format specified")
        
        # Check quality for lossy formats
        if parameters.get('format') in ['JPEG', 'WEBP']:
            if 'quality' not in parameters:
                errors.append("Quality is required for JPEG/WEBP formats")
            elif not isinstance(parameters['quality'], int) or parameters['quality'] < 0 or parameters['quality'] > 100:
                errors.append("Quality must be an integer between 0 and 100")
        
        return errors
    
    def get_estimated_memory(self, item: BatchItem) -> int:
        """Estimate memory requirements for format conversion.
        
        Args:
            item: The batch item to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        # Basic estimation: just the input image in memory
        file_manager = get_file_manager()
        metadata = file_manager.get_metadata(item.input_path)
        
        if metadata:
            # Estimate memory for input image
            return metadata.width * metadata.height * metadata.channels * 4  # 4 bytes per pixel in QImage
        
        # Default estimate if metadata not available
        return 50 * 1024 * 1024  # 50 MB


class BatchFilterOperation(BatchOperation):
    """Batch filter application operation."""
    
    name = "Apply Filter"
    description = "Apply a filter to multiple images"
    
    def process_item(self, item: BatchItem) -> QImage:
        """Apply a filter to an image.
        
        Args:
            item: The batch item to process
            
        Returns:
            Filtered QImage
        """
        from .filter_manager import filter_manager
        
        file_manager = get_file_manager()
        
        # Load the input image
        input_image = file_manager.load_image(item.input_path)
        if input_image is None:
            raise ValueError(f"Failed to load image: {item.input_path}")
        
        # Get filter parameters
        filter_name = item.parameters.get('filter_name')
        filter_params = item.parameters.get('filter_params', {})
        
        if not filter_name:
            raise ValueError("No filter specified")
        
        # Check if filter exists
        if not filter_manager.has_filter(filter_name):
            raise ValueError(f"Filter not found: {filter_name}")
        
        # Apply the filter
        result = filter_manager.apply_filter(input_image, filter_name, filter_params)
        
        return result
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this operation.
        
        Returns:
            Dictionary describing the parameters for filter application
        """
        from .filter_manager import filter_manager
        
        # Get available filters
        available_filters = filter_manager.get_filter_names()
        
        return {
            'filter_name': {
                'type': 'string',
                'description': 'Name of the filter to apply',
                'enum': available_filters
            },
            'filter_params': {
                'type': 'object',
                'description': 'Parameters for the filter',
                'default': {}
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate the parameters for filter application.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of error messages, empty if valid
        """
        from .filter_manager import filter_manager
        
        errors = []
        
        # Check filter name
        if 'filter_name' not in parameters:
            errors.append("Filter name is required")
        elif not filter_manager.has_filter(parameters['filter_name']):
            errors.append(f"Filter not found: {parameters['filter_name']}")
        
        # Validate filter parameters
        if 'filter_params' in parameters and parameters['filter_name']:
            filter_errors = filter_manager.validate_filter_params(
                parameters['filter_name'], 
                parameters['filter_params']
            )
            errors.extend(filter_errors)
        
        return errors
    
    def get_estimated_memory(self, item: BatchItem) -> int:
        """Estimate memory requirements for filter application.
        
        Args:
            item: The batch item to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        # Basic estimation: input image + output image
        file_manager = get_file_manager()
        metadata = file_manager.get_metadata(item.input_path)
        
        if metadata:
            # Estimate memory for both input and output images
            memory = metadata.width * metadata.height * metadata.channels * 4 * 2  # 4 bytes per pixel in QImage, 2 images
            
            # Some filters might use more memory
            filter_name = item.parameters.get('filter_name', '')
            if filter_name.lower() in ['blur', 'gaussian blur', 'motion blur']:
                # Convolution filters can use more memory for intermediate buffers
                memory *= 1.5
            
            return int(memory)
        
        # Default estimate if metadata not available
        return 100 * 1024 * 1024  # 100 MB


class BatchProcessor(QObject):
    """Processor for batch operations on multiple images."""
    
    # Signals
    batch_started = pyqtSignal(str)  # Batch ID
    batch_completed = pyqtSignal(str)  # Batch ID
    batch_failed = pyqtSignal(str, str)  # Batch ID, error message
    batch_cancelled = pyqtSignal(str)  # Batch ID
    item_started = pyqtSignal(str, str)  # Batch ID, item ID
    item_progress = pyqtSignal(str, str, float, str)  # Batch ID, item ID, progress percentage, message
    item_completed = pyqtSignal(str, str)  # Batch ID, item ID
    item_failed = pyqtSignal(str, str, str)  # Batch ID, item ID, error message
    
    def __init__(self):
        """Initialize the batch processor."""
        super().__init__()
        
        self.task_manager = get_task_manager()
        self.file_manager = get_file_manager()
        self.memory_manager = get_memory_manager()
        
        # Dictionary of registered operations
        self.operations: Dict[str, BatchOperation] = {}
        
        # Dictionary of active batches
        self.active_batches: Dict[str, List[BatchItem]] = {}
        
        # Register default operations
        self.register_operation('resize', BatchResizeOperation())
        self.register_operation('convert_format', BatchFormatConvertOperation())
        self.register_operation('apply_filter', BatchFilterOperation())
    
    def register_operation(self, operation_id: str, operation: BatchOperation) -> None:
        """Register a batch operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation: BatchOperation instance
        """
        self.operations[operation_id] = operation
        logger.info(f"Registered batch operation: {operation_id} ({operation.name})")
    
    def get_operations(self) -> Dict[str, BatchOperation]:
        """Get all registered operations.
        
        Returns:
            Dictionary of operation ID to BatchOperation
        """
        return self.operations.copy()
    
    def get_operation(self, operation_id: str) -> Optional[BatchOperation]:
        """Get a specific operation by ID.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            BatchOperation or None if not found
        """
        return self.operations.get(operation_id)
    
    def create_batch(self, operation_id: str, items: List[Dict[str, Any]], 
                   global_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new batch operation.
        
        Args:
            operation_id: Operation identifier
            items: List of items to process (dictionaries with input_path, output_path, parameters)
            global_parameters: Optional parameters to apply to all items
            
        Returns:
            Batch ID
            
        Raises:
            ValueError: If operation not found or invalid parameters
        """
        # Check if operation exists
        if operation_id not in self.operations:
            raise ValueError(f"Operation not found: {operation_id}")
        
        operation = self.operations[operation_id]
        
        # Create a new batch ID
        batch_id = str(uuid.uuid4())
        
        # Create batch items
        batch_items = []
        for i, item_dict in enumerate(items):
            # Merge global and item-specific parameters
            parameters = {}
            if global_parameters:
                parameters.update(global_parameters)
            if 'parameters' in item_dict:
                parameters.update(item_dict['parameters'])
            
            # Create batch item
            item = BatchItem(
                id=f"{batch_id}_{i}",
                name=item_dict.get('name', f"Item {i+1}"),
                input_path=item_dict.get('input_path'),
                output_path=item_dict.get('output_path'),
                operation=operation_id,
                parameters=parameters,
                status=BatchItemStatus.PENDING
            )
            
            # Validate parameters for this item
            errors = operation.validate_parameters(parameters)
            if errors:
                raise ValueError(f"Invalid parameters for item {i+1}: {', '.join(errors)}")
            
            batch_items.append(item)
        
        # Store the batch
        self.active_batches[batch_id] = batch_items
        
        logger.info(f"Created batch {batch_id} with {len(batch_items)} items")
        
        return batch_id
    
    def start_batch(self, batch_id: str, 
                  max_concurrent_items: int = 2,
                  on_batch_progress: Optional[Callable[[float, str], None]] = None) -> None:
        """Start processing a batch.
        
        Args:
            batch_id: Batch identifier
            max_concurrent_items: Maximum number of concurrent items to process
            on_batch_progress: Optional callback for batch progress updates
            
        Raises:
            ValueError: If batch not found
        """
        # Check if batch exists
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch not found: {batch_id}")
        
        batch_items = self.active_batches[batch_id]
        
        # Create a task to process the batch
        task_id = str(uuid.uuid4())
        
        def _batch_task():
            """Background task to process the batch."""
            try:
                # Emit batch started signal
                self.batch_started.emit(batch_id)
                
                # Process items
                failed_items = 0
                completed_items = 0
                total_items = len(batch_items)
                
                # Create a list to track active processing tasks
                active_tasks = []
                
                # Process items until all done
                while completed_items + failed_items < total_items:
                    # Check how many slots are available
                    slots_available = max_concurrent_items - len(active_tasks)
                    
                    # Start new tasks if slots are available
                    if slots_available > 0:
                        for item in batch_items:
                            if (item.status == BatchItemStatus.PENDING and 
                                slots_available > 0):
                                # Start processing this item
                                item.status = BatchItemStatus.PROCESSING
                                self.item_started.emit(batch_id, item.id)
                                
                                # Create task for this item
                                item_task_id = str(uuid.uuid4())
                                active_tasks.append(item_task_id)
                                
                                # Define callbacks
                                def on_item_progress(progress):
                                    """Handle item progress updates."""
                                    item.progress = progress.percent
                                    item.message = progress.message
                                    self.item_progress.emit(batch_id, item.id, progress.percent, progress.message)
                                    
                                    # Update overall batch progress
                                    batch_progress = (completed_items + failed_items) / total_items * 100.0
                                    batch_progress += item.progress / total_items
                                    
                                    if on_batch_progress:
                                        on_batch_progress(batch_progress, f"Processing {completed_items + failed_items + 1}/{total_items}")
                                
                                def on_item_complete(result):
                                    """Handle item completion."""
                                    nonlocal completed_items
                                    
                                    # Remove from active tasks
                                    if item_task_id in active_tasks:
                                        active_tasks.remove(item_task_id)
                                    
                                    # Save result if output path specified
                                    if result and item.output_path:
                                        # Determine format
                                        format = None
                                        quality = 90
                                        
                                        # If this is a format conversion operation, get format from parameters
                                        if item.operation == 'convert_format':
                                            format_name = item.parameters.get('format', 'PNG')
                                            for f in FileFormat:
                                                if f.name == format_name:
                                                    format = f
                                                    break
                                            
                                            quality = item.parameters.get('quality', 90)
                                        
                                        # Save the result
                                        success = self.file_manager.save_image(
                                            result,
                                            item.output_path,
                                            format=format,
                                            quality=quality
                                        )
                                        
                                        if not success:
                                            # Failed to save
                                            item.status = BatchItemStatus.FAILED
                                            item.error = f"Failed to save output to {item.output_path}"
                                            failed_items += 1
                                            self.item_failed.emit(batch_id, item.id, item.error)
                                            return
                                    
                                    # Update item status
                                    item.status = BatchItemStatus.COMPLETED
                                    item.result = result
                                    item.progress = 100.0
                                    completed_items += 1
                                    self.item_completed.emit(batch_id, item.id)
                                
                                def on_item_error(err):
                                    """Handle item error."""
                                    nonlocal failed_items
                                    
                                    # Remove from active tasks
                                    if item_task_id in active_tasks:
                                        active_tasks.remove(item_task_id)
                                    
                                    # Update item status
                                    item.status = BatchItemStatus.FAILED
                                    item.error = str(err)
                                    failed_items += 1
                                    self.item_failed.emit(batch_id, item.id, item.error)
                                
                                # Create the task
                                operation = self.operations[item.operation]
                                
                                def process_item_task():
                                    """Task to process a single item."""
                                    try:
                                        # Update progress - starting
                                        progress = TaskProgress(percent=0, message="Starting processing...")
                                        self.task_manager.update_task_progress(item_task_id, progress)
                                        
                                        # Process the item
                                        result = operation.process_item(item)
                                        
                                        # Update progress - completed
                                        progress = TaskProgress(percent=100, message="Processing complete")
                                        self.task_manager.update_task_progress(item_task_id, progress)
                                        
                                        return result
                                    except Exception as e:
                                        logger.error(f"Error processing batch item {item.id}: {e}")
                                        raise
                                
                                # Submit the task
                                task = Task(
                                    id=item_task_id,
                                    name=f"Process {item.name}",
                                    callback=process_item_task
                                )
                                
                                self.task_manager.submit_task(
                                    task,
                                    on_progress=on_item_progress,
                                    on_complete=on_item_complete,
                                    on_error=on_item_error
                                )
                                
                                # Decrement available slots
                                slots_available -= 1
                    
                    # Wait a bit before checking again
                    time.sleep(0.1)
                
                # Batch completed
                if failed_items > 0:
                    error_message = f"{failed_items} out of {total_items} items failed"
                    self.batch_failed.emit(batch_id, error_message)
                else:
                    self.batch_completed.emit(batch_id)
                
                # Final batch progress update
                if on_batch_progress:
                    on_batch_progress(100.0, f"Batch completed ({completed_items}/{total_items} successful)")
                
                return {
                    'total_items': total_items,
                    'completed_items': completed_items,
                    'failed_items': failed_items
                }
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {e}")
                self.batch_failed.emit(batch_id, str(e))
                
                if on_batch_progress:
                    on_batch_progress(0.0, f"Batch failed: {str(e)}")
                
                raise
        
        # Create and submit the task
        batch_task = Task(
            id=task_id,
            name=f"Batch Processing: {len(batch_items)} items",
            callback=_batch_task
        )
        
        self.task_manager.submit_task(batch_task)
        
        logger.info(f"Started batch {batch_id} with task {task_id}")
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if batch was cancelled, False if not found
        """
        if batch_id not in self.active_batches:
            return False
        
        # Cancel all pending items
        batch_items = self.active_batches[batch_id]
        for item in batch_items:
            if item.status == BatchItemStatus.PENDING:
                item.status = BatchItemStatus.SKIPPED
        
        # The task manager will handle cancelling in-progress items
        # through the task cancellation mechanism
        
        # Emit signal
        self.batch_cancelled.emit(batch_id)
        
        logger.info(f"Cancelled batch {batch_id}")
        
        return True
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Dictionary with batch status or None if not found
        """
        if batch_id not in self.active_batches:
            return None
        
        batch_items = self.active_batches[batch_id]
        
        # Count items by status
        total_items = len(batch_items)
        pending_items = sum(1 for item in batch_items if item.status == BatchItemStatus.PENDING)
        processing_items = sum(1 for item in batch_items if item.status == BatchItemStatus.PROCESSING)
        completed_items = sum(1 for item in batch_items if item.status == BatchItemStatus.COMPLETED)
        failed_items = sum(1 for item in batch_items if item.status == BatchItemStatus.FAILED)
        skipped_items = sum(1 for item in batch_items if item.status == BatchItemStatus.SKIPPED)
        
        # Calculate progress
        if total_items > 0:
            # Base progress on completed/failed/skipped items
            progress = (completed_items + failed_items + skipped_items) / total_items * 100.0
            
            # Add partial progress from items being processed
            for item in batch_items:
                if item.status == BatchItemStatus.PROCESSING:
                    progress += (item.progress / 100.0) * (1.0 / total_items) * 100.0
        else:
            progress = 0.0
        
        return {
            'batch_id': batch_id,
            'total_items': total_items,
            'pending_items': pending_items,
            'processing_items': processing_items,
            'completed_items': completed_items,
            'failed_items': failed_items,
            'skipped_items': skipped_items,
            'progress': progress,
            'is_complete': pending_items == 0 and processing_items == 0,
            'items': batch_items
        }
    
    def remove_batch(self, batch_id: str) -> bool:
        """Remove a completed or failed batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if batch was removed, False if not found or still running
        """
        if batch_id not in self.active_batches:
            return False
        
        # Check if batch is still running
        batch_items = self.active_batches[batch_id]
        for item in batch_items:
            if item.status in [BatchItemStatus.PENDING, BatchItemStatus.PROCESSING]:
                return False
        
        # Remove the batch
        del self.active_batches[batch_id]
        
        logger.info(f"Removed batch {batch_id}")
        
        return True


# Singleton pattern
_batch_processor = None

def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance.
    
    Returns:
        The batch processor instance
    """
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor 