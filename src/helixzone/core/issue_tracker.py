"""Issue tracking module for monitoring and debugging item processing."""

from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
import time
from ..core.debugger import ProjectDebugger

# Get the global debugger instance
debugger = ProjectDebugger(
    project_name="helixzone",
    log_dir="debug_logs"
)

@dataclass
class ProcessingStats:
    """Statistics for item processing."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    item_times: Dict[int, float] = field(default_factory=dict)
    failures: Dict[int, str] = field(default_factory=dict)

class IssueTracker:
    """Tracks and debugs item processing with detailed error handling."""
    
    def __init__(self):
        self.debugger = debugger
        self.stats = ProcessingStats()
        
    @debugger.trace_function
    def process_item(self, item: Any) -> None:
        """Process a single item with performance tracking.
        
        Args:
            item: The item to process
            
        Raises:
            ValueError: If item is invalid
            RuntimeError: If processing fails
        """
        start_time = time.time()
        
        try:
            # Log item details
            self.debugger.variable_dump(
                prefix="item_processing",
                item=item
            )
            
            # Validate item
            if item is None:
                raise ValueError("Item cannot be None")
                
            # TODO: Add actual item processing logic here
            # This is a placeholder for demonstration
            if isinstance(item, dict):
                # Process dictionary items
                for key, value in item.items():
                    # Simulate processing
                    _ = f"{key}:{value}"
            elif isinstance(item, (list, tuple)):
                # Process sequence items
                for value in item:
                    # Simulate processing
                    _ = str(value)
            else:
                # Process simple items
                _ = str(item)
                
            # Update success stats
            self.stats.processed_items += 1
            self.stats.item_times[self.stats.processed_items] = time.time() - start_time
            
        except Exception as e:
            # Update failure stats
            self.stats.failed_items += 1
            self.stats.failures[self.stats.failed_items] = str(e)
            raise
    
    @debugger.trace_function
    def process_with_tracking(self, items: List[Any]) -> None:
        """Process multiple items with comprehensive error tracking.
        
        Args:
            items: List of items to process
        """
        # Initialize stats
        self.stats = ProcessingStats(total_items=len(items))
        
        # Log batch info
        self.debugger.variable_dump(
            prefix="batch_processing",
            total_items=len(items),
            start_time=self.stats.start_time
        )
        
        for i, item in enumerate(items):
            with self.debugger.error_boundary(f"item_{i}"):
                try:
                    # Process each item
                    self.process_item(item)
                    
                    # Log progress
                    self.debugger.checkpoint(
                        f"Processed item {i + 1}/{len(items)}",
                        include_memory=False  # Reduce overhead for frequent checkpoints
                    )
                    
                except Exception as e:
                    # Log failure but continue with next item
                    self.debugger.error_logger.error(
                        f"Item {i} failed: {str(e)}"
                    )
        
        # Log final stats
        self._log_final_stats()
    
    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        elapsed = time.time() - self.stats.start_time
        
        stats_summary = {
            'total_items': self.stats.total_items,
            'processed_items': self.stats.processed_items,
            'failed_items': self.stats.failed_items,
            'success_rate': (self.stats.processed_items / self.stats.total_items) * 100,
            'total_time': elapsed,
            'avg_time_per_item': elapsed / self.stats.total_items if self.stats.total_items > 0 else 0,
            'failures': self.stats.failures
        }
        
        self.debugger.state_logger.info(
            f"Processing completed: {stats_summary}"
        )
        
        # Create final checkpoint
        self.debugger.checkpoint(
            "Batch processing completed",
            include_memory=True  # Include memory stats for final checkpoint
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'total_items': self.stats.total_items,
            'processed_items': self.stats.processed_items,
            'failed_items': self.stats.failed_items,
            'item_times': self.stats.item_times,
            'failures': self.stats.failures,
            'elapsed_time': time.time() - self.stats.start_time
        } 