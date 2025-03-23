"""Type stubs for pynvml."""

from typing import Any, Optional, Union, Tuple, Protocol

class NVMLError(Exception): ...

class NVMLUtilizationRates(Protocol):
    """Protocol for NVML utilization rates."""
    gpu: int
    memory: int

class NVMLDevice(Protocol):
    """Protocol for NVML device."""
    handle: Any

# Concrete implementations
class nvmlUtilizationRates:
    """Concrete implementation of utilization rates."""
    gpu: int
    memory: int

class nvmlDevice:
    """Concrete implementation of NVML device."""
    handle: Any

def nvmlInit() -> None: ...
def nvmlShutdown() -> None: ...
def nvmlDeviceGetHandleByIndex(index: int) -> NVMLDevice: ...
def nvmlDeviceGetUtilizationRates(handle: NVMLDevice) -> NVMLUtilizationRates: ...
def nvmlDeviceGetTemperature(handle: NVMLDevice, sensor_type: int) -> float: ...
def nvmlDeviceGetPowerUsage(handle: NVMLDevice) -> int: ...

# Constants
NVML_TEMPERATURE_GPU: int = 0

# Make Protocol types available at the top level
__all__ = [
    'NVMLError',
    'NVMLUtilizationRates',
    'NVMLDevice',
    'nvmlUtilizationRates',
    'nvmlDevice',
    'nvmlInit',
    'nvmlShutdown',
    'nvmlDeviceGetHandleByIndex',
    'nvmlDeviceGetUtilizationRates',
    'nvmlDeviceGetTemperature',
    'nvmlDeviceGetPowerUsage',
    'NVML_TEMPERATURE_GPU',
] 