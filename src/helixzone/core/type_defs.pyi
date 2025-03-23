"""Type stub file for HelixZone type definitions."""

from typing import (
    Union, TypeVar, Any, Literal, Tuple, List, Dict, 
    Protocol, runtime_checkable, Final, TypedDict, Optional,
    overload
)
import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

# Type variables
T = TypeVar('T', bound=np.generic)

# Basic array types
Array = NDArray[Any]
ImageArray = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
EdgeMapArray = NDArray[np.float32]
FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

# Complex types
ImageSource = Union[str, NDArray[np.uint8]]
TensorOrArray = Union[NDArray[Any], 'GpuArray']

class EdgeDetectionParams(TypedDict):
    sigma: float
    low_threshold: float
    high_threshold: float

class ProcessingResults(TypedDict):
    original: NDArray[np.uint8]
    processed: NDArray[np.uint8]
    params: EdgeDetectionParams

# GPU device protocol
class GpuDevice(Protocol):
    def __init__(self) -> None: ...
    def allocate(self, shape: tuple[int, ...], dtype: Any) -> 'GpuArray': ...
    def to_cpu(self, array: 'GpuArray') -> NDArray[Any]: ...
    def from_cpu(self, array: NDArray[Any]) -> 'GpuArray': ...

class GpuArray(Protocol):
    shape: tuple[int, ...]
    dtype: Any

# GPU types
GPUBackend = Literal['cuda', 'opencl', 'cpu']

@runtime_checkable
class GPUDevice(Protocol):
    """Protocol for GPU devices."""
    @property
    def name(self) -> str: ...
    
    @property
    def total_memory(self) -> int: ...
    
    @property
    def compute_capability(self) -> Tuple[int, int]: ...
    
    def reset(self) -> None: ...
    
    def synchronize(self) -> None: ...

# Type guards
def is_image_array(arr: Any) -> bool: ...
def is_float_array(arr: Any) -> bool: ...
def is_gray_image(arr: Any) -> bool: ...
def is_color_image(arr: Any) -> bool: ...
def is_edge_map(arr: Any) -> bool: ... 