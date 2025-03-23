"""Type definitions for NVIDIA GPU monitoring using pynvml."""

from typing import Protocol, TypeVar, Any
from typing_extensions import TypeAlias
from dataclasses import dataclass

# Type definitions
class NVMLError(Exception):
    """Base exception for NVML errors."""
    pass

@dataclass
class nvmlUtilizationRates:
    """GPU utilization rates."""
    gpu: int
    memory: int

# Alias for better type hints
NVMLUtilizationRates: TypeAlias = nvmlUtilizationRates

class nvmlDevice(Protocol):
    """Protocol for NVIDIA GPU device."""
    def __init__(self) -> None: ...

# Alias for better type hints
NVMLDevice: TypeAlias = nvmlDevice

# Constants
NVML_TEMPERATURE_GPU: int = 0

# Function type definitions
def nvmlInit() -> None:
    """Initialize NVML library."""
    raise NotImplementedError("This is a type stub")

def nvmlShutdown() -> None:
    """Shut down NVML library."""
    raise NotImplementedError("This is a type stub")

def nvmlDeviceGetHandleByIndex(index: int) -> nvmlDevice:
    """Get handle to GPU device by index."""
    raise NotImplementedError("This is a type stub")

def nvmlDeviceGetUtilizationRates(device: nvmlDevice) -> nvmlUtilizationRates:
    """Get GPU utilization rates."""
    raise NotImplementedError("This is a type stub")

def nvmlDeviceGetTemperature(device: nvmlDevice, sensor_type: int) -> float:
    """Get GPU temperature."""
    raise NotImplementedError("This is a type stub")

def nvmlDeviceGetPowerUsage(device: nvmlDevice) -> int:
    """Get GPU power usage in milliwatts."""
    raise NotImplementedError("This is a type stub") 