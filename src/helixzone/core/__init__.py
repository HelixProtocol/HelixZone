"""Core module for HelixZone."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import all modules to make them available
from . import gpu_manager
from . import logging_manager
from . import model_training
from . import feature_extraction
from . import image_processing
from . import tools

# Import and re-export all type definitions
from .type_defs import (
    ImageArray,
    FloatArray,
    GPUBackend,
    ColorMode,
    EdgeDetectionParams,
    GPUMetrics,
    MetricsDict,
    BatchSizerConfig,
    ProcessingResults,
    GPUDevice,
    GPUMemoryError,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    Mat,
    GaussianBlurInput,
    CVMat,
    CVMatLike,
    MatLike,
    to_mat,
    to_float_img,
    to_uint8_img,
    ensure_mat,
    ensure_float32,
    ensure_uint8,
    ImageFloat,
    ImageUInt8,
    ImageBool,
    ArrayLike,
    ensure_array,
)

if TYPE_CHECKING:
    from .pynvml import (
        NVMLError,
        NVMLUtilizationRates,
        NVMLDevice,
        nvmlUtilizationRates,
        nvmlDevice,
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetPowerUsage,
        NVML_TEMPERATURE_GPU,
    )

from .tools import (
    Tool,
    BrushTool,
    EraserTool,
    SelectionTool,
    RectangleSelection,
    EllipseSelection,
    LassoSelection,
    MagneticLassoSelection,
)

# Export module names and types
__all__ = [
    # Modules
    'gpu_manager',
    'logging_manager',
    'model_training',
    'feature_extraction',
    'image_processing',
    'tools',
    # Tools
    'Tool',
    'BrushTool',
    'EraserTool',
    'SelectionTool',
    'RectangleSelection',
    'EllipseSelection',
    'LassoSelection',
    'MagneticLassoSelection',
    # Type definitions
    'ImageArray',
    'FloatArray',
    'GPUBackend',
    'ColorMode',
    'EdgeDetectionParams',
    'GPUDevice',
    'GPUMetrics',
    'MetricsDict',
    'BatchSizerConfig',
    'ProcessingResults',
    'GPUMemoryError',
    'MEMORY_CRITICAL_THRESHOLD',
    'MEMORY_WARNING_THRESHOLD',
    'Mat',
    'GaussianBlurInput',
    'CVMat',
    'CVMatLike',
    'MatLike',
    'to_mat',
    'to_float_img',
    'to_uint8_img',
    'ensure_mat',
    'ensure_float32',
    'ensure_uint8',
    'ImageFloat',
    'ImageUInt8',
    'ImageBool',
    'ArrayLike',
    'ensure_array',
] 