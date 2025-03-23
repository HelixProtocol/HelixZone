"""Dynamic kernel parameter tuning module."""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
import pyopencl as cl
from dataclasses import dataclass
from enum import Enum, auto
import time
import json
import os
from pathlib import Path
from .opencl_vendor import VendorType, VendorOptimizer

@dataclass
class TuningResult:
    """Result of a kernel parameter tuning run."""
    work_group_size: Tuple[int, int]
    vector_width: int
    local_memory_size: int
    execution_time: float
    throughput: float
    memory_usage: float
    success: bool
    error: Optional[str] = None

class TuningMetric(Enum):
    """Metrics for kernel tuning."""
    EXECUTION_TIME = auto()
    THROUGHPUT = auto()
    MEMORY_USAGE = auto()
    COMBINED = auto()

class KernelTuner:
    """Dynamic kernel parameter tuner."""
    def __init__(self, device: cl.Device, cache_dir: Optional[str] = None):
        self.device = device
        self.optimizer = VendorOptimizer(device)
        self.cache_dir = cache_dir or str(Path.home() / '.helixzone' / 'kernel_tuning')
        self._ensure_cache_dir()
        self._load_cache()
        
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, kernel_name: str) -> str:
        """Get cache file path for kernel."""
        vendor_name = self.optimizer.vendor.name.lower()
        return os.path.join(self.cache_dir, f"{kernel_name}_{vendor_name}.json")
        
    def _load_cache(self) -> None:
        """Load tuning cache."""
        self.cache: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(self.cache_dir):
            return
            
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.cache_dir, file), 'r') as f:
                    kernel_name = file.split('_')[0]
                    self.cache[kernel_name] = json.load(f)
                    
    def _save_cache(self, kernel_name: str) -> None:
        """Save tuning cache."""
        if kernel_name not in self.cache:
            return
            
        cache_path = self._get_cache_path(kernel_name)
        with open(cache_path, 'w') as f:
            json.dump(self.cache[kernel_name], f, indent=2)
            
    def _get_work_group_sizes(self) -> List[Tuple[int, int]]:
        """Get list of work group sizes to try."""
        base_sizes = []
        
        if self.optimizer.vendor == VendorType.AMD:
            base_sizes = [(16, 16), (8, 32), (32, 8), (8, 8)]
        elif self.optimizer.vendor == VendorType.NVIDIA:
            base_sizes = [(32, 32), (16, 64), (64, 16), (16, 16)]
        elif self.optimizer.vendor == VendorType.INTEL:
            base_sizes = [(8, 8), (16, 4), (4, 16), (16, 16)]
        else:
            base_sizes = [(16, 16), (8, 8), (32, 32), (8, 32)]
            
        return base_sizes
        
    def _get_vector_widths(self) -> List[int]:
        """Get list of vector widths to try."""
        if self.optimizer.vendor == VendorType.AMD:
            return [4, 2, 1]
        elif self.optimizer.vendor == VendorType.NVIDIA:
            return [2, 4, 1]
        elif self.optimizer.vendor == VendorType.INTEL:
            return [8, 4, 2, 1]
        else:
            return [4, 2, 1]
            
    def _get_local_memory_sizes(self) -> List[int]:
        """Get list of local memory sizes to try."""
        max_size = self.optimizer.get_local_memory_size()
        sizes = []
        
        size = 1024  # Start with 1KB
        while size <= max_size:
            sizes.append(size)
            size *= 2
            
        return sizes
        
    def _evaluate_parameters(
        self,
        kernel_name: str,
        work_group_size: Tuple[int, int],
        vector_width: int,
        local_memory_size: int,
        test_data: NDArray[np.uint8],
        warmup_runs: int = 3,
        timing_runs: int = 10
    ) -> TuningResult:
        """Evaluate a set of kernel parameters."""
        try:
            # Create context and queue
            ctx = cl.Context([self.device])
            queue = cl.CommandQueue(ctx)
            
            # Prepare kernel with parameters
            defines = [
                f"#define WORK_GROUP_SIZE_X {work_group_size[0]}",
                f"#define WORK_GROUP_SIZE_Y {work_group_size[1]}",
                f"#define VECTOR_WIDTH {vector_width}",
                f"#define LOCAL_MEM_SIZE {local_memory_size}"
            ]
            
            # Get kernel source
            kernel_source = self.optimizer.optimize_kernel_source(
                self._get_kernel_source(kernel_name)
            )
            
            # Build program
            program = cl.Program(ctx, "\n".join(defines + [kernel_source])).build(
                options=list(self.optimizer.get_kernel_options().keys())
            )
            
            # Prepare buffers
            height, width = test_data.shape
            input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=test_data)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, test_data.nbytes)
            
            # Warm up runs
            for _ in range(warmup_runs):
                getattr(program, kernel_name)(
                    queue,
                    (width, height),
                    work_group_size,
                    input_buf,
                    output_buf,
                    np.int32(width),
                    np.int32(height)
                )
                queue.finish()
                
            # Timing runs
            times = []
            for _ in range(timing_runs):
                start = time.perf_counter()
                
                getattr(program, kernel_name)(
                    queue,
                    (width, height),
                    work_group_size,
                    input_buf,
                    output_buf,
                    np.int32(width),
                    np.int32(height)
                )
                queue.finish()
                
                end = time.perf_counter()
                times.append(end - start)
                
            # Calculate metrics
            execution_time = float(np.mean(times))
            throughput = float((width * height) / execution_time)
            memory_usage = float(input_buf.size + output_buf.size)
            
            return TuningResult(
                work_group_size=work_group_size,
                vector_width=vector_width,
                local_memory_size=local_memory_size,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage=memory_usage,
                success=True
            )
            
        except Exception as e:
            return TuningResult(
                work_group_size=work_group_size,
                vector_width=vector_width,
                local_memory_size=local_memory_size,
                execution_time=float('inf'),
                throughput=0.0,
                memory_usage=float('inf'),
                success=False,
                error=str(e)
            )
            
    def _get_kernel_source(self, kernel_name: str) -> str:
        """Get kernel source code."""
        if kernel_name == 'bilateral_filter':
            from .opencl_kernels import BILATERAL_FILTER_KERNEL
            return BILATERAL_FILTER_KERNEL
        elif kernel_name == 'non_maximum_suppression':
            from .opencl_kernels import NMS_KERNEL
            return NMS_KERNEL
        elif kernel_name == 'sobel_edge_detection':
            from .opencl_kernels import EDGE_DETECTION_KERNEL
            return EDGE_DETECTION_KERNEL
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
            
    def tune_kernel(
        self,
        kernel_name: str,
        test_data: NDArray[np.uint8],
        metric: TuningMetric = TuningMetric.EXECUTION_TIME,
        max_trials: int = 100,
        cache: bool = True
    ) -> Dict[str, Any]:
        """Tune kernel parameters."""
        # Check cache first
        if cache and kernel_name in self.cache:
            return self.cache[kernel_name]
            
        best_result: Optional[TuningResult] = None
        results: List[TuningResult] = []
        
        # Generate parameter combinations
        work_group_sizes = self._get_work_group_sizes()
        vector_widths = self._get_vector_widths()
        local_memory_sizes = self._get_local_memory_sizes()
        
        # Try different combinations
        trials = 0
        for work_group_size in work_group_sizes:
            for vector_width in vector_widths:
                for local_memory_size in local_memory_sizes:
                    if trials >= max_trials:
                        break
                        
                    result = self._evaluate_parameters(
                        kernel_name,
                        work_group_size,
                        vector_width,
                        local_memory_size,
                        test_data
                    )
                    
                    if result.success:
                        results.append(result)
                        
                        # Update best result
                        if best_result is None:
                            best_result = result
                        else:
                            if metric == TuningMetric.EXECUTION_TIME:
                                if result.execution_time < best_result.execution_time:
                                    best_result = result
                            elif metric == TuningMetric.THROUGHPUT:
                                if result.throughput > best_result.throughput:
                                    best_result = result
                            elif metric == TuningMetric.MEMORY_USAGE:
                                if result.memory_usage < best_result.memory_usage:
                                    best_result = result
                            else:  # COMBINED
                                score = (
                                    result.throughput / result.execution_time *
                                    (1.0 / (1.0 + result.memory_usage / 1e6))
                                )
                                best_score = (
                                    best_result.throughput / best_result.execution_time *
                                    (1.0 / (1.0 + best_result.memory_usage / 1e6))
                                )
                                if score > best_score:
                                    best_result = result
                                    
                    trials += 1
                    
        if best_result is None:
            raise RuntimeError("No successful parameter combination found")
            
        # Save results to cache
        tuning_data = {
            'best_parameters': {
                'work_group_size': best_result.work_group_size,
                'vector_width': best_result.vector_width,
                'local_memory_size': best_result.local_memory_size
            },
            'performance': {
                'execution_time': best_result.execution_time,
                'throughput': best_result.throughput,
                'memory_usage': best_result.memory_usage
            },
            'device': {
                'name': self.device.name,
                'vendor': self.device.vendor,
                'version': self.device.version
            }
        }
        
        if cache:
            self.cache[kernel_name] = tuning_data
            self._save_cache(kernel_name)
            
        return tuning_data
        
    def get_optimal_parameters(self, kernel_name: str) -> Dict[str, Any]:
        """Get optimal parameters for a kernel."""
        if kernel_name not in self.cache:
            raise ValueError(f"No tuning data available for kernel: {kernel_name}")
            
        return self.cache[kernel_name]['best_parameters'] 