"""Dynamic batch size management for GPU processing."""

from __future__ import annotations
from typing import Dict, List, Optional, Union, cast
from typing_extensions import Final
import warnings

# Constants
INITIAL_BATCH_SIZE: Final[int] = 32
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 256
MEMORY_BUFFER: Final[float] = 0.2  # 20% buffer for GPU memory

# Type aliases
BatchConfig = Dict[str, Union[int, float]]

class BatchSizer:
    """Dynamic batch size manager for GPU processing."""

    def __init__(self, initial_batch_size: int = INITIAL_BATCH_SIZE) -> None:
        self._batch_size = initial_batch_size
        self._initialized = False
        self._has_cuda = False
        self._has_opencl = False
        self._memory_history: List[float] = []

    def initialize(self) -> None:
        """Initialize batch sizer."""
        if self._initialized:
            return

        try:
            import cupy as cp
            self._has_cuda = True
        except ImportError:
            warnings.warn("CUDA not available, falling back to CPU processing")

        try:
            import pyopencl as cl
            self._has_opencl = True
        except ImportError:
            warnings.warn("OpenCL not available, falling back to CPU processing")

        self._initialized = True

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self._batch_size

    def update_batch_size(self, memory_utilization: float) -> None:
        """Update batch size based on memory utilization."""
        self._memory_history.append(memory_utilization)
        if len(self._memory_history) > 10:
            self._memory_history.pop(0)

        avg_utilization = sum(self._memory_history) / len(self._memory_history)
        
        # Adjust batch size based on memory utilization
        if avg_utilization > 90.0:  # High memory pressure
            self._batch_size = max(MIN_BATCH_SIZE, self._batch_size // 2)
        elif avg_utilization < 50.0:  # Low memory pressure
            self._batch_size = min(MAX_BATCH_SIZE, self._batch_size * 2)

    def get_config(self) -> BatchConfig:
        """Get current batch configuration."""
        return {
            'batch_size': self._batch_size,
            'min_batch_size': MIN_BATCH_SIZE,
            'max_batch_size': MAX_BATCH_SIZE,
            'memory_buffer': MEMORY_BUFFER,
            'avg_memory_utilization': (
                sum(self._memory_history) / len(self._memory_history)
                if self._memory_history else 0.0
            )
        }

    def reset(self) -> None:
        """Reset batch sizer to initial state."""
        self._batch_size = INITIAL_BATCH_SIZE
        self._memory_history.clear() 