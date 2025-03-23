"""Central type definitions for HelixZone."""

from typing import (
    Protocol,
    TypeVar,
    Dict,
    Optional,
    Final,
    Union,
    runtime_checkable,
    Any,
    Tuple,
)
from dataclasses import dataclass
import torch
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.cuda import Event as CudaEvent
    from torch.cuda import Stream as CudaStream

# Type variables
T = TypeVar('T')
DeviceType = Union[str, torch.device]

class GPUBackend(Enum):
    """Supported GPU backends."""
    CUDA = auto()
    OPENCL = auto()
    CPU = auto()

@runtime_checkable
class GPUDevice(Protocol):
    """Protocol for GPU device interface."""
    @property
    def name(self) -> str: ...
    @property
    def total_memory(self) -> int: ...
    @property
    def compute_capability(self) -> Tuple[int, int]: ...

@runtime_checkable
class NVMLDevice(Protocol):
    """Protocol for NVML device interface."""
    handle: Any

@runtime_checkable
class NVMLUtilizationRates(Protocol):
    """Protocol for NVML utilization rates."""
    gpu: int
    memory: int

@dataclass(frozen=True)
class GPUMetrics:
    """Comprehensive GPU metrics."""
    device_id: int
    utilization: float  # GPU utilization percentage
    memory_used: int    # Memory used in bytes
    memory_total: int   # Total memory in bytes
    temperature: float  # Temperature in Celsius
    power_usage: float  # Power usage in Watts
    timestamp: float    # When these metrics were collected
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        return (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0.0
    
    @property
    def is_memory_critical(self) -> bool:
        """Check if memory usage is in critical state."""
        return self.memory_utilization > 90.0

@dataclass(frozen=True)
class BatchSizerConfig:
    """Configuration for batch size optimization."""
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    target_memory_utilization: float = 80.0
    target_gpu_utilization: float = 90.0
    growth_factor: float = 1.5
    shrink_factor: float = 0.7

# Constants
UPDATE_INTERVAL: Final[float] = 1.0  # Update metrics every second
CACHE_DURATION: Final[float] = 0.1   # Cache metrics for 100ms
MEMORY_CRITICAL_THRESHOLD: Final[float] = 90.0  # Memory usage percentage considered critical
MEMORY_WARNING_THRESHOLD: Final[float] = 80.0   # Memory usage percentage that triggers warnings
TEMPERATURE_CRITICAL_THRESHOLD: Final[float] = 85.0  # GPU temperature (°C) considered critical
TEMPERATURE_WARNING_THRESHOLD: Final[float] = 75.0   # GPU temperature (°C) that triggers warnings

# Custom exceptions
class GPUError(Exception):
    """Base exception for GPU-related errors."""
    pass

class GPUMemoryError(GPUError):
    """Exception raised for GPU memory-related issues."""
    pass

class GPUTemperatureError(GPUError):
    """Exception raised for GPU temperature-related issues."""
    pass

class GPUInitializationError(GPUError):
    """Exception raised when GPU initialization fails."""
    pass

class NVMLError(GPUError):
    """Exception raised for NVML-related issues."""
    pass

# Type aliases
MetricsDict = Dict[int, GPUMetrics]
PerformanceHistory = Dict[int, float]

__all__ = [
    'GPUBackend',
    'GPUDevice',
    'NVMLDevice',
    'NVMLUtilizationRates',
    'GPUMetrics',
    'BatchSizerConfig',
    'UPDATE_INTERVAL',
    'CACHE_DURATION',
    'MEMORY_CRITICAL_THRESHOLD',
    'MEMORY_WARNING_THRESHOLD',
    'TEMPERATURE_CRITICAL_THRESHOLD',
    'TEMPERATURE_WARNING_THRESHOLD',
    'GPUError',
    'GPUMemoryError',
    'GPUTemperatureError',
    'NVMLError',
    'MetricsDict',
    'PerformanceHistory',
] 