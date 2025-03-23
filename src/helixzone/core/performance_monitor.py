"""Performance monitoring module for critical code sections."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import psutil
import numpy as np
from contextlib import contextmanager
from ..core.debugger import ProjectDebugger

# Get the global debugger instance
debugger = ProjectDebugger(
    project_name="helixzone",
    log_dir="debug_logs"
)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    gpu_metrics: Dict[str, List[float]] = field(default_factory=dict)

class PerformanceMonitor:
    """Monitors and analyzes performance of critical code sections."""
    
    def __init__(self):
        self.debugger = debugger
        self.metrics = PerformanceMetrics()
        self._process = psutil.Process()
        
    @contextmanager
    def measure_block(self, block_name: str):
        """Context manager for measuring a block of code.
        
        Args:
            block_name: Name of the code block being measured
        """
        start_time = time.time()
        start_cpu = self._process.cpu_percent()
        start_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = self._process.cpu_percent()
            end_memory = self._process.memory_info().rss / 1024 / 1024
            
            # Record metrics
            self.metrics.execution_times.append(end_time - start_time)
            self.metrics.cpu_usage.append((end_cpu + start_cpu) / 2)
            self.metrics.memory_usage.append(end_memory - start_memory)
            self.metrics.timestamps.append(end_time)
            
            # Log performance data
            self.debugger.perf_logger.info(
                f"Block '{block_name}' metrics: "
                f"time={end_time - start_time:.3f}s, "
                f"cpu={end_cpu:.1f}%, "
                f"memory_delta={end_memory - start_memory:.1f}MB"
            )
    
    @debugger.trace_function
    def performance_critical_code(self, input_data: Any) -> Tuple[Any, Dict[str, float]]:
        """Execute and monitor performance-critical code.
        
        Args:
            input_data: Data to process
            
        Returns:
            Tuple of (result, performance_metrics)
        """
        self.debugger.checkpoint("Starting performance check")
        
        try:
            with self.measure_block("critical_section"):
                # Log input characteristics
                self.debugger.variable_dump(
                    prefix="perf_critical",
                    input_type=type(input_data).__name__,
                    input_size=len(input_data) if hasattr(input_data, '__len__') else 'N/A'
                )
                
                # TODO: Replace with actual performance-critical code
                if isinstance(input_data, np.ndarray):
                    # Example: Matrix operations
                    result = np.fft.fft2(input_data)
                elif isinstance(input_data, (list, tuple)):
                    # Example: Sequence processing
                    result = sorted(input_data)
                else:
                    result = input_data
                
                # Calculate performance metrics
                metrics = self._calculate_metrics()
                
                return result, metrics
                
        finally:
            self.debugger.checkpoint("Ending performance check")
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate summary statistics from collected metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.metrics.execution_times:
            return {}
            
        metrics = {
            'avg_execution_time': np.mean(self.metrics.execution_times),
            'max_execution_time': np.max(self.metrics.execution_times),
            'min_execution_time': np.min(self.metrics.execution_times),
            'std_execution_time': np.std(self.metrics.execution_times),
            'avg_cpu_usage': np.mean(self.metrics.cpu_usage),
            'peak_cpu_usage': np.max(self.metrics.cpu_usage),
            'avg_memory_delta': np.mean(self.metrics.memory_usage),
            'peak_memory_delta': np.max(self.metrics.memory_usage)
        }
        
        # Log detailed metrics
        self.debugger.state_logger.info(
            f"Performance metrics: {metrics}"
        )
        
        return metrics
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get historical performance data.
        
        Returns:
            Dictionary containing lists of metrics over time
        """
        return {
            'execution_times': self.metrics.execution_times,
            'cpu_usage': self.metrics.cpu_usage,
            'memory_usage': self.metrics.memory_usage,
            'timestamps': self.metrics.timestamps
        }
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = PerformanceMetrics()
        self.debugger.checkpoint("Performance metrics reset") 