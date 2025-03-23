"""GPU metrics collection module for HelixZone."""

from __future__ import annotations
from typing import Dict, Optional, Union, cast, TypedDict, NotRequired
from typing_extensions import Final
import psutil
import warnings
import threading
import time

# Constants
MEMORY_CRITICAL_THRESHOLD: Final[float] = 95.0  # Percentage
MEMORY_WARNING_THRESHOLD: Final[float] = 85.0  # Percentage
UPDATE_INTERVAL: Final[float] = 1.0  # Seconds
CACHE_DURATION: Final[float] = 0.5  # Seconds

# Type definitions
class MetricsDict(TypedDict):
    """GPU metrics dictionary type."""
    memory_utilization: float
    utilization: float
    temperature: float
    power_usage: float
    memory_total: int
    memory_used: int
    memory_free: int

class GPUMetrics:
    """GPU metrics data class."""
    def __init__(
        self,
        memory_utilization: float = 0.0,
        utilization: float = 0.0,
        temperature: float = 0.0,
        power_usage: float = 0.0,
        memory_total: int = 0,
        memory_used: int = 0,
        memory_free: int = 0
    ) -> None:
        self.memory_utilization = memory_utilization
        self.utilization = utilization
        self.temperature = temperature
        self.power_usage = power_usage
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.memory_free = memory_free

    @classmethod
    def from_dict(cls, data: MetricsDict) -> 'GPUMetrics':
        """Create GPUMetrics from a dictionary."""
        return cls(
            memory_utilization=data['memory_utilization'],
            utilization=data['utilization'],
            temperature=data['temperature'],
            power_usage=data['power_usage'],
            memory_total=data['memory_total'],
            memory_used=data['memory_used'],
            memory_free=data['memory_free']
        )

    def to_dict(self) -> MetricsDict:
        """Convert to dictionary format."""
        return {
            'memory_utilization': self.memory_utilization,
            'utilization': self.utilization,
            'temperature': self.temperature,
            'power_usage': self.power_usage,
            'memory_total': self.memory_total,
            'memory_used': self.memory_used,
            'memory_free': self.memory_free
        }

class GPUMetricsCollector:
    """Collector for GPU metrics."""

    def __init__(self) -> None:
        self._initialized = False
        self._has_cuda = False
        self._has_opencl = False
        self._metrics_cache: Dict[int, MetricsDict] = {}
        self._last_update: Dict[int, float] = {}
        self._stop_event = threading.Event()
        self._collection_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """Initialize metrics collector."""
        if self._initialized:
            return

        try:
            import cupy as cp
            self._has_cuda = True
        except ImportError:
            warnings.warn("CUDA not available, falling back to CPU metrics only")

        try:
            import pyopencl as cl
            self._has_opencl = True
        except ImportError:
            warnings.warn("OpenCL not available, falling back to CPU metrics only")

        # Start background collection thread
        self._collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self._collection_thread.start()
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the metrics collector."""
        self._stop_event.set()
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=1.0)

    def get_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get metrics for a specific GPU device."""
        if not self._initialized:
            self.initialize()

        # Check cache first
        if device_id in self._metrics_cache:
            last_update = self._last_update.get(device_id, 0)
            if time.time() - last_update <= CACHE_DURATION:
                return GPUMetrics.from_dict(self._metrics_cache[device_id])

        # Collect fresh metrics
        metrics_dict = self.collect_metrics()
        if metrics_dict is not None:
            self._metrics_cache[device_id] = metrics_dict
            self._last_update[device_id] = time.time()
            return GPUMetrics.from_dict(metrics_dict)
        return None

    def _collect_metrics_loop(self) -> None:
        """Background metrics collection loop."""
        while not self._stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                if metrics is not None:
                    self._metrics_cache[0] = metrics
                    self._last_update[0] = time.time()
            except Exception as e:
                warnings.warn(f"Error collecting metrics: {e}")
            time.sleep(UPDATE_INTERVAL)

    def collect_metrics(self) -> Optional[MetricsDict]:
        """Collect GPU metrics."""
        if not self._initialized:
            self.initialize()

        if self._has_cuda:
            try:
                import cupy as cp
                device = cp.cuda.Device()
                memory_info = device.mem_info
                memory_total = memory_info[1]
                memory_used = memory_info[1] - memory_info[0]
                memory_free = memory_info[0]
                memory_utilization = (memory_used / memory_total) * 100.0

                return {
                    'memory_utilization': memory_utilization,
                    'utilization': device.attributes['MultiProcessorCount'] * 100.0,
                    'temperature': device.attributes.get('Temperature', 0.0),
                    'power_usage': device.attributes.get('PowerUsage', 0.0),
                    'memory_total': memory_total,
                    'memory_used': memory_used,
                    'memory_free': memory_free
                }
            except Exception as e:
                warnings.warn(f"Failed to collect CUDA metrics: {e}")
                return None

        if self._has_opencl:
            try:
                import pyopencl as cl
                platforms = cl.get_platforms()
                if not platforms:
                    return None

                # Get first available GPU device
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        device = devices[0]
                        memory_total = device.global_mem_size
                        memory_free = device.global_mem_cache_size
                        memory_used = memory_total - memory_free
                        memory_utilization = (memory_used / memory_total) * 100.0

                        return {
                            'memory_utilization': memory_utilization,
                            'utilization': 0.0,  # Not available in OpenCL
                            'temperature': 0.0,  # Not available in OpenCL
                            'power_usage': 0.0,  # Not available in OpenCL
                            'memory_total': memory_total,
                            'memory_used': memory_used,
                            'memory_free': memory_free
                        }
                return None
            except Exception as e:
                warnings.warn(f"Failed to collect OpenCL metrics: {e}")
                return None

        # Fall back to CPU metrics
        try:
            memory = psutil.virtual_memory()
            return {
                'memory_utilization': memory.percent,
                'utilization': psutil.cpu_percent(),
                'temperature': 0.0,  # Not available for CPU
                'power_usage': 0.0,  # Not available for CPU
                'memory_total': memory.total,
                'memory_used': memory.used,
                'memory_free': memory.available
            }
        except Exception as e:
            warnings.warn(f"Failed to collect CPU metrics: {e}")
            return None 