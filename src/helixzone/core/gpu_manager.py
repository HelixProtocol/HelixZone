"""GPU resource management for HelixZone."""

from __future__ import annotations
from typing import Optional, Dict, Any, Final, Generator, Protocol, TypeVar, cast, Union, overload
import torch
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
from .gpu_metrics import GPUMetricsCollector, GPUMetrics, MetricsDict
from .type_defs import (
    ImageArray,
    FloatArray,
    EdgeDetectionParams,
    GPUBackend
)
import numpy as np
import cv2
from numpy.typing import NDArray

# Constants
MEMORY_CRITICAL_THRESHOLD: Final[float] = 95.0  # Percentage
MEMORY_WARNING_THRESHOLD: Final[float] = 85.0  # Percentage

class GPUMemoryError(Exception):
    """Raised when GPU memory is critically low."""
    pass

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MemoryDict = Dict[str, Union[int, float]]
EdgeMap = Dict[str, NDArray[np.float32]]

@dataclass
class BatchSizerConfig:
    """Configuration for batch size optimization."""
    initial_batch_size: int = 100
    min_batch_size: int = 10
    max_batch_size: int = 1000
    growth_factor: float = 1.5
    shrink_factor: float = 0.5
    target_memory_utilization: float = 80.0
    target_gpu_utilization: float = 80.0

class BatchSizer:
    """Dynamic batch size optimizer based on GPU metrics."""
    
    def __init__(self, config: Optional[BatchSizerConfig] = None) -> None:
        self.config = config or BatchSizerConfig()
        self.batch_size = self.config.initial_batch_size
        self._performance_history: Dict[int, float] = {}

    def update(self, metrics: Optional[GPUMetrics], processing_time: float) -> None:
        """Update batch size based on GPU metrics."""
        if metrics is None:
            return
            
        self._performance_history[self.batch_size] = processing_time
            
        # Calculate headroom with safety margins
        memory_usage = metrics.memory_utilization / 100.0
        memory_headroom = (self.config.target_memory_utilization / 100.0 - memory_usage) * 0.9  # 10% safety margin
        
        util_usage = metrics.utilization / 100.0
        util_headroom = (self.config.target_gpu_utilization / 100.0 - util_usage) * 0.9  # 10% safety margin
        
        # Adjust batch size based on both metrics with more conservative thresholds
        if memory_headroom > 0.2 and util_headroom > 0.2:  # More conservative growth
            # Can increase batch size
            self.batch_size = min(
                int(self.batch_size * (1 + (self.config.growth_factor - 1) * 0.5)),  # Slower growth
                self.config.max_batch_size
            )
        elif memory_usage > 0.85 or metrics.utilization > 85:  # Earlier intervention
            # Need to decrease batch size
            self.batch_size = max(
                int(self.batch_size * self.config.shrink_factor),
                self.config.min_batch_size
            )
            
        # Clear old history entries to prevent memory growth
        if len(self._performance_history) > 1000:
            # Keep only the most recent 100 entries
            sorted_history = sorted(self._performance_history.items(), key=lambda x: x[1])
            self._performance_history = dict(sorted_history[:100])

    def get_optimal_batch_size(self) -> int:
        """Get the batch size with the best performance from history."""
        if not self._performance_history:
            return self.batch_size
            
        return min(
            self._performance_history.items(),
            key=lambda x: x[1]
        )[0]

class MultiGPUManager:
    """Manages multiple GPU devices for distributed processing."""
    
    def __init__(self) -> None:
        self._metrics_collector = GPUMetricsCollector()
        self._initialized = False
        self._current_device = 0
        self._device_contexts: Dict[int, Any] = {}
        self._cached_memory: Dict[int, Dict[str, Any]] = {}
        self._last_cleanup: Dict[int, float] = {}
        
    def __del__(self) -> None:
        """Cleanup resources."""
        try:
            self.cleanup_all_devices()
            if hasattr(self, '_metrics_collector'):
                self._metrics_collector.shutdown()
        except Exception:
            pass  # Suppress errors during shutdown
    
    def initialize(self) -> None:
        """Initialize GPU resources."""
        if self._initialized:
            return
            
        if not torch.cuda.is_available():
            warnings.warn("No CUDA devices available")
            return
            
        self._initialized = True
        torch.cuda.init()  # Ensure CUDA is properly initialized
        
    def cleanup_all_devices(self) -> None:
        """Cleanup resources on all devices."""
        import time
        
        for device_id in range(self.get_device_count()):
            try:
                # Clear CUDA cache
                with self.device_context(device_id):
                    torch.cuda.empty_cache()
                    
                # Release any cached memory
                if device_id in self._cached_memory:
                    del self._cached_memory[device_id]
                    
                # Clear device contexts
                if device_id in self._device_contexts:
                    del self._device_contexts[device_id]
                    
                self._last_cleanup[device_id] = time.time()
            except Exception as e:
                warnings.warn(f"Error cleaning up device {device_id}: {e}")
                
    def clear_memory(self, device_id: Optional[int] = None) -> None:
        """Clear memory on specified device or current device."""
        import time
        
        if device_id is None:
            device_id = self.get_current_device()
            
        try:
            with self.device_context(device_id):
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Clear cached memory info
                if device_id in self._cached_memory:
                    del self._cached_memory[device_id]
                    
                self._last_cleanup[device_id] = time.time()
        except Exception as e:
            warnings.warn(f"Error clearing memory on device {device_id}: {e}")
            
    def get_memory_info(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get memory information for specified device."""
        if device_id is None:
            device_id = self.get_current_device()
            
        try:
            with self.device_context(device_id):
                info = {
                    'allocated': torch.cuda.memory_allocated(device_id),
                    'reserved': torch.cuda.memory_reserved(device_id),
                    'max_allocated': torch.cuda.max_memory_allocated(device_id),
                    'max_reserved': torch.cuda.max_memory_reserved(device_id)
                }
                
                # Cache the info
                self._cached_memory[device_id] = info
                return info
        except Exception as e:
            warnings.warn(f"Error getting memory info for device {device_id}: {e}")
            return {}
            
    @contextmanager
    def device_context(self, device_id: Optional[int] = None, required_memory: int = 0) -> Generator[int, None, None]:
        """Context manager for device operations with memory check."""
        if device_id is None:
            device_id = self.get_optimal_device(required_memory)
            
        if not self._initialized:
            self.initialize()
            
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")
            
        try:
            # Set device
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device_id)
            
            # Check memory availability
            free_memory = self.get_memory_info(device_id).get('free', 0)
            if required_memory > 0 and free_memory < required_memory:
                # Try to free memory
                self.clear_memory(device_id)
                free_memory = self.get_memory_info(device_id).get('free', 0)
                if free_memory < required_memory:
                    raise GPUMemoryError(f"Insufficient memory on device {device_id}")
                    
            yield device_id
        finally:
            # Restore previous device
            torch.cuda.set_device(prev_device)
            
    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
        
    def is_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        return torch.cuda.is_available()
        
    def get_current_device(self) -> int:
        """Get current device ID."""
        return self._current_device
        
    def set_device(self, device_id: int) -> None:
        """Set current device."""
        if 0 <= device_id < self.get_device_count():
            self._current_device = device_id
            torch.cuda.set_device(device_id)
        else:
            raise ValueError(f"Invalid device ID: {device_id}")
    
    def get_device_profile(self, device_id: int) -> Optional[GPUMetrics]:
        """Get detailed profile for a specific device."""
        return self._metrics_collector.get_metrics(device_id)
    
    def get_optimal_device(self, required_memory: int = 0) -> int:
        """Get the optimal device for a new operation."""
        if not self._initialized:
            self.initialize()
            
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")
            
        best_device = 0
        best_score = float('-inf')
        
        for device_id in range(torch.cuda.device_count()):
            metrics = self._metrics_collector.get_metrics(device_id)
            if metrics is None:
                continue
                
            # Skip if not enough memory
            if required_memory > 0:
                free_memory = metrics.memory_total - metrics.memory_used
                if free_memory < required_memory:
                    continue
            
            # Calculate device score based on multiple factors
            memory_score = (metrics.memory_total - metrics.memory_used) / metrics.memory_total
            util_score = (100 - metrics.utilization) / 100
            temp_score = (100 - min(metrics.temperature, 100)) / 100
            
            # Weighted score (can be tuned based on priorities)
            score = memory_score * 0.5 + util_score * 0.3 + temp_score * 0.2
            
            if score > best_score:
                best_score = score
                best_device = device_id
        
        return best_device

class GPUResourceMonitor:
    """Monitors and manages GPU resources."""
    
    def __init__(self, multi_gpu: MultiGPUManager, batch_sizer_config: Optional[BatchSizerConfig] = None) -> None:
        self.multi_gpu = multi_gpu
        self._current_device: Optional[int] = None
        self.batch_sizer = BatchSizer(batch_sizer_config)
    
    @contextmanager
    def monitor_device(self) -> Generator[Optional[GPUMetrics], None, None]:
        """Context manager for monitoring GPU device usage."""
        if not torch.cuda.is_available():
            yield None
            return
            
        # Get optimal device
        self._current_device = self.multi_gpu.get_optimal_device()
        if self._current_device is None:
            yield None
            return
            
        try:
            # Record initial state
            start_metrics = self.multi_gpu.get_device_profile(self._current_device)
            
            yield start_metrics
            
            # Record final state and update batch sizer
            end_metrics = self.multi_gpu.get_device_profile(self._current_device)
            if end_metrics is not None:
                processing_time = torch.cuda.Event().elapsed_time(torch.cuda.Event())
                self.batch_sizer.update(end_metrics, processing_time)
        except Exception as e:
            warnings.warn(f"Error monitoring GPU device: {e}")
            yield None
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        if self._current_device is None:
            return None
        return self.multi_gpu.get_device_profile(self._current_device)

class GPUManager:
    """Manager for GPU operations."""

    def __init__(self) -> None:
        self._initialized = False
        self.backend: GPUBackend = 'cpu'
        self._metrics_collector = GPUMetricsCollector()

    def initialize(self) -> None:
        """Initialize GPU backend."""
        if self._initialized:
            return

        # Try CUDA first
        try:
            import cupy as cp
            self.backend = 'cuda'
        except ImportError:
            # Try OpenCL next
            try:
                import pyopencl as cl
                self.backend = 'opencl'
            except ImportError:
                # Fall back to CPU
                self.backend = 'cpu'

        self._initialized = True

    def get_backend(self) -> GPUBackend:
        """Get current GPU backend."""
        if not self._initialized:
            self.initialize()
        return self.backend

    def get_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        if not self._initialized or self.backend == 'cpu':
            return None
        metrics_dict = self._metrics_collector.collect_metrics()
        if metrics_dict is None:
            return None
        return GPUMetrics.from_dict(metrics_dict)

    def process_edges(self, gray: ImageArray, params: EdgeDetectionParams) -> EdgeMap:
        """Process edges using available GPU backend."""
        if not self._initialized:
            self.initialize()

        # Apply bilateral filter
        gray_filtered = cv2.bilateralFilter(
            src=gray,
            d=params.get('d', 5),
            sigmaColor=params.get('sigma_color', 50.0),
            sigmaSpace=params.get('sigma_space', 50.0)
        )

        # Compute adaptive thresholds
        mean_intensity = float(np.mean(gray_filtered.astype(np.float64)))
        std_intensity = float(np.std(gray_filtered.astype(np.float64)))
        low_threshold = max(0, mean_intensity - std_intensity)
        high_threshold = min(255, mean_intensity + std_intensity)

        # Multi-scale edge detection
        edges_fine = cv2.Canny(
            image=gray_filtered,
            threshold1=params.get('threshold1', low_threshold),
            threshold2=params.get('threshold2', high_threshold),
            apertureSize=params.get('aperture_size', 3),
            L2gradient=params.get('l2_gradient', False)
        )

        gray_medium = cv2.GaussianBlur(gray_filtered, (5, 5), 1.5)
        edges_medium = cv2.Canny(
            image=gray_medium,
            threshold1=params.get('threshold1', low_threshold) * 0.8,
            threshold2=params.get('threshold2', high_threshold) * 0.8,
            apertureSize=params.get('aperture_size', 3),
            L2gradient=params.get('l2_gradient', False)
        )

        gray_coarse = cv2.GaussianBlur(gray_filtered, (9, 9), 2.5)
        edges_coarse = cv2.Canny(
            image=gray_coarse,
            threshold1=params.get('threshold1', low_threshold) * 0.6,
            threshold2=params.get('threshold2', high_threshold) * 0.6,
            apertureSize=params.get('aperture_size', 3),
            L2gradient=params.get('l2_gradient', False)
        )

        # Combine edges with weighted addition
        edge_map = cv2.addWeighted(
            edges_fine.astype(np.float32),
            0.5,
            cv2.addWeighted(
                edges_medium.astype(np.float32),
                0.3,
                edges_coarse.astype(np.float32),
                0.2,
                0
            ),
            0.5,
            0
        )

        # Compute gradients
        gradient_x = cv2.Sobel(gray_filtered, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_filtered, cv2.CV_32F, 0, 1, ksize=3)

        # Compute edge strength and gradient
        edge_strength = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_gradient = np.arctan2(gradient_y, gradient_x)

        # Normalize edge strength
        edge_min = float(np.min(edge_strength))
        edge_max = float(np.max(edge_strength))
        if edge_max > edge_min:
            edge_strength = ((edge_strength - edge_min) * 255.0 / (edge_max - edge_min))

        return {
            'edge_map': edge_map.astype(np.float32),
            'edge_strength': edge_strength.astype(np.float32),
            'edge_gradient': edge_gradient.astype(np.float32)
        } 