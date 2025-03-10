"""OpenCL support for GPU acceleration."""

from typing import (
    Any, Callable, Dict, Optional, Protocol, Tuple, TypeVar, Generic,
    runtime_checkable, Generator, Iterator, Type, Sequence, Union, List, Set
)
from typing_extensions import TypeGuard
import numpy as np
import numpy.typing as npt
import pyopencl as cl  # type: ignore
import threading
import time
from contextlib import contextmanager

# Define type variables
T_co = TypeVar("T_co", covariant=True)  # Covariant for return values

class OpenCLError(Exception):
    """Custom exception for OpenCL-specific errors."""
    pass

class PerformanceMonitor:
    """Monitor OpenCL operation performance."""
    
    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start_operation(self, name: str) -> None:
        """Start timing an operation."""
        self._start_times[name] = time.perf_counter()
        
    def end_operation(self, name: str) -> None:
        """End timing an operation and record results."""
        if name in self._start_times:
            duration = time.perf_counter() - self._start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            del self._start_times[name]
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation."""
        if name in self.timings and self.timings[name]:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0.0
    
    def print_statistics(self) -> None:
        """Print performance statistics."""
        print("\nPerformance Statistics:")
        print("-" * 50)
        for name, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"{name}:")
                print(f"  Average: {avg_time*1000:.2f}ms")
                print(f"  Min: {min_time*1000:.2f}ms")
                print(f"  Max: {max_time*1000:.2f}ms")
                print(f"  Calls: {len(times)}")
        print("-" * 50)

class MemoryPool:
    """Pool for reusing OpenCL buffers."""
    
    def __init__(self, context: cl.Context) -> None:
        self._context = context
        self._available_buffers: Dict[int, List[cl.Buffer]] = {}
        self._active_buffers: Set[cl.Buffer] = set()

    def get_buffer(self, size: int, flags: int, hostbuf: Any = None) -> cl.Buffer:
        """Get a buffer from the pool or create a new one."""
        if size in self._available_buffers and self._available_buffers[size]:
            buffer = self._available_buffers[size].pop()
            self._active_buffers.add(buffer)
            return buffer
        buffer = cl.Buffer(self._context, flags, size, hostbuf=hostbuf)
        self._active_buffers.add(buffer)
        return buffer

    def return_buffer(self, buffer: cl.Buffer) -> None:
        """Return a buffer to the pool."""
        if buffer in self._active_buffers:
            size = buffer.size
            if size not in self._available_buffers:
                self._available_buffers[size] = []
            self._available_buffers[size].append(buffer)
            self._active_buffers.remove(buffer)

    def cleanup(self) -> None:
        """Release all buffers in the pool."""
        for buffers in self._available_buffers.values():
            for buffer in buffers:
                buffer.release()
        self._available_buffers.clear()
        for buffer in self._active_buffers:
            buffer.release()
        self._active_buffers.clear()

@runtime_checkable
class OpenCLTask(Protocol[T_co]):
    """Protocol defining the interface for tasks that can be executed on OpenCL devices."""
    
    @property
    def result(self) -> T_co:
        """Get the task result."""
        ...
    
    def wait(self) -> None:
        """Wait for task completion."""
        ...
    
    def get_kernel_source(self) -> str:
        """Get the OpenCL kernel source code for this task."""
        ...
    
    def get_kernel_name(self) -> str:
        """Get the name of the kernel function."""
        ...
    
    def prepare_data(self) -> Tuple[npt.NDArray[np.float32], ...]:
        """Prepare input data arrays for the kernel."""
        ...
    
    def validate_result(self, result: Any) -> bool:
        """Validate the result type."""
        ...
    
    def process_result(self, result: npt.NDArray[np.float32]) -> T_co:
        """Process the kernel output to produce the final result."""
        ...

class OpenCLTaskImpl(Generic[T_co]):
    """Base implementation for OpenCL tasks."""
    
    def __init__(self) -> None:
        self._result: Optional[Union[T_co, Exception]] = None
        self._completed = threading.Event()
        self._success = False
    
    @property
    def result(self) -> T_co:
        """Get the task result."""
        self.wait()
        if self._result is None:
            raise RuntimeError("Task has no result")
        if not self._success:
            if isinstance(self._result, Exception):
                raise RuntimeError(f"Task failed: {self._result}")
            raise RuntimeError("Task failed with unknown error")
        if isinstance(self._result, Exception):
            raise RuntimeError("Invalid state: success flag with exception result")
        return self._result
    
    def wait(self) -> None:
        """Wait for task completion."""
        self._completed.wait()
    
    def _complete(self, result: Union[T_co, Exception], success: bool) -> None:
        """Complete the task with a result."""
        self._result = result
        self._success = success
        self._completed.set()

class OpenCLContext:
    """Manages OpenCL context and command queue."""
    
    def __init__(self, device_type: cl.device_type = cl.device_type.GPU) -> None:
        # Try to get GPU platform first
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=device_type)
                    if devices:
                        self.device = devices[0]
                        self.context = cl.Context(devices=[self.device])
                        self.queue = cl.CommandQueue(
                            self.context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE
                        )
                        break
                except cl.RuntimeError as e:
                    print(f"Warning: Failed to initialize device on platform {platform.name}: {e}")
            else:
                raise OpenCLError(f"No OpenCL device of type {device_type} found")
        except Exception as e:
            raise OpenCLError(f"Failed to initialize OpenCL: {e}")
        
        # Initialize support components
        self._program_cache: Dict[str, cl.Program] = {}
        self._memory_pool = MemoryPool(self.context)
        self._performance_monitor = PerformanceMonitor()
        
        # Get device capabilities
        self._max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self._max_work_item_sizes = self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
        
        print(f"Initialized OpenCL device: {self.device.name}")
        print(f"Max work group size: {self._max_work_group_size}")
        print(f"Max work item sizes: {self._max_work_item_sizes}")
    
    def get_optimal_work_group_size(self, global_size: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate optimal work group size based on device capabilities and input size."""
        optimal_size = []
        for i, size in enumerate(global_size):
            max_size = min(self._max_work_item_sizes[i], self._max_work_group_size)
            # Find largest power of 2 that divides the global size and is <= max_size
            wg_size = 1
            while wg_size * 2 <= max_size and size % (wg_size * 2) == 0:
                wg_size *= 2
            optimal_size.append(wg_size)
        return tuple(optimal_size)
    
    def execute_task(self, task: OpenCLTask[T_co]) -> OpenCLTaskImpl[T_co]:
        """Execute a task on the OpenCL device."""
        opencl_task = OpenCLTaskImpl[T_co]()
        
        def run_task() -> None:
            self._performance_monitor.start_operation("total_execution")
            buffers: List[cl.Buffer] = []
            output_buffer: Optional[cl.Buffer] = None
            try:
                # Get or compile kernel
                self._performance_monitor.start_operation("kernel_compilation")
                kernel_source = task.get_kernel_source()
                if kernel_source not in self._program_cache:
                    program = cl.Program(self.context, kernel_source).build()
                    self._program_cache[kernel_source] = program
                else:
                    program = self._program_cache[kernel_source]
                self._performance_monitor.end_operation("kernel_compilation")
                
                # Get kernel function
                kernel_func = getattr(program, task.get_kernel_name())
                
                # Prepare input data
                self._performance_monitor.start_operation("data_preparation")
                input_arrays = task.prepare_data()
                
                # Create OpenCL buffers using memory pool
                for arr in input_arrays:
                    buf = self._memory_pool.get_buffer(
                        arr.nbytes,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=arr
                    )
                    buffers.append(buf)
                
                # Create output buffer
                output_shape = input_arrays[0].shape
                output = np.empty(output_shape, dtype=np.float32)
                output_buffer = self._memory_pool.get_buffer(
                    output.nbytes,
                    cl.mem_flags.WRITE_ONLY
                )
                self._performance_monitor.end_operation("data_preparation")
                
                # Execute kernel with optimal work group size
                self._performance_monitor.start_operation("kernel_execution")
                global_size = output_shape[::-1]  # Reverse shape for OpenCL
                local_size = self.get_optimal_work_group_size(global_size)
                
                kernel_func(
                    self.queue,
                    global_size,
                    local_size,
                    *buffers,
                    output_buffer
                )
                
                # Read result
                cl.enqueue_copy(self.queue, output, output_buffer)
                self.queue.finish()
                self._performance_monitor.end_operation("kernel_execution")
                
                # Process and set result
                self._performance_monitor.start_operation("result_processing")
                result = task.process_result(output)
                if task.validate_result(result):
                    opencl_task._complete(result, True)
                else:
                    error = TypeError(f"Invalid result type: {type(result)}")
                    opencl_task._complete(error, False)
                self._performance_monitor.end_operation("result_processing")
                
            except cl.RuntimeError as e:
                error = OpenCLError(f"OpenCL runtime error: {e}")
                opencl_task._complete(error, False)
                raise error
            except Exception as e:
                error = OpenCLError(f"OpenCL operation failed: {e}")
                opencl_task._complete(error, False)
                raise error
            finally:
                # Cleanup buffers
                for buf in buffers:
                    self._memory_pool.return_buffer(buf)
                if output_buffer is not None:
                    self._memory_pool.return_buffer(output_buffer)
                self._performance_monitor.end_operation("total_execution")
        
        # Run task in thread pool
        threading.Thread(target=run_task, daemon=True).start()
        return opencl_task
    
    def release(self) -> None:
        """Release OpenCL resources."""
        try:
            # Print performance statistics
            self._performance_monitor.print_statistics()
            
            # Cleanup resources
            for program in self._program_cache.values():
                program.release()
            self._program_cache.clear()
            
            self._memory_pool.cleanup()
            self.queue.finish()
            self.queue.release()
            self.context.release()
        except Exception as e:
            raise OpenCLError(f"Failed to release OpenCL resources: {e}")

class EdgeDetectionTask(OpenCLTask[npt.NDArray[np.float32]]):
    """OpenCL task for edge detection."""
    
    def __init__(self, image: npt.NDArray[np.float32], threshold: float = 0.1) -> None:
        self.image = image
        self.threshold = np.float32(threshold)
        self._result: Optional[npt.NDArray[np.float32]] = None
        self._completed = threading.Event()
    
    @property
    def result(self) -> npt.NDArray[np.float32]:
        self.wait()
        if self._result is None:
            raise RuntimeError("Task has no result")
        return self._result
    
    def wait(self) -> None:
        self._completed.wait()
    
    def get_kernel_name(self) -> str:
        return "edge_detection"
    
    def get_kernel_source(self) -> str:
        return """
        __kernel void edge_detection(
            __global const float *input,
            __global const float *threshold,
            __global float *output
        ) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int width = get_global_size(0);
            int height = get_global_size(1);
            
            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
                return;
            
            int idx = y * width + x;
            
            // Sobel operators
            float gx = input[idx - 1 - width]  + 2 * input[idx - 1] + input[idx - 1 + width]
                    - input[idx + 1 - width] - 2 * input[idx + 1] - input[idx + 1 + width];
            
            float gy = input[idx - width - 1] + 2 * input[idx - width] + input[idx - width + 1]
                    - input[idx + width - 1] - 2 * input[idx + width] - input[idx + width + 1];
            
            float magnitude = sqrt(gx * gx + gy * gy);
            output[idx] = magnitude > threshold[0] ? 1.0f : 0.0f;
        }
        """
    
    def prepare_data(self) -> Tuple[npt.NDArray[np.float32], ...]:
        return (self.image, np.array([self.threshold], dtype=np.float32))
    
    def validate_result(self, result: Any) -> bool:
        return isinstance(result, np.ndarray) and result.dtype == np.float32
    
    def process_result(self, result: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self._result = result
        self._completed.set()
        return result

def process_edges_opencl(image: npt.NDArray[np.float32], threshold: float = 0.1) -> Iterator[npt.NDArray[np.float32]]:
    """Process edges using OpenCL acceleration."""
    ctx = OpenCLContext()
    try:
        task = EdgeDetectionTask(image, threshold)
        result = ctx.execute_task(task)
        yield result.result
    finally:
        ctx.release() 