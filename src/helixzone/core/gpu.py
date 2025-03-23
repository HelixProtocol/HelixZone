"""GPU acceleration module for image processing."""
from typing import Dict, Optional, Any, Literal
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import cupy as cp
from numpy.typing import NDArray
from .opencl import process_edges_opencl, EdgeDetectionTask, OpenCLContext
import pyopencl as cl
import logging

# Global processing queue and thread
_processing_queue: Queue = Queue()
_processing_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# GPU backend type
GPUBackend = Literal['cuda', 'opencl', 'cpu']

class GPUManager:
    """Manager for GPU operations."""
    def __init__(self):
        self._backend: GPUBackend = 'cpu'
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize GPU backend."""
        if self._initialized:
            return
            
        try:
            # Try CUDA first
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            self._backend = 'cuda'
        except Exception:
            try:
                # Try OpenCL next
                import pyopencl as cl
                platforms = cl.get_platforms()
                if platforms and any(p.get_devices(device_type=cl.device_type.GPU) for p in platforms):
                    self._backend = 'opencl'
                else:
                    self._backend = 'cpu'
            except Exception:
                self._backend = 'cpu'
                
        self._initialized = True
        
    def get_backend(self) -> GPUBackend:
        """Get current GPU backend."""
        if not self._initialized:
            self.initialize()
        return self._backend
        
    def process_edges(self, gray: NDArray[np.uint8], params: Dict[str, Any]) -> Dict[str, NDArray]:
        """Process edges using available GPU backend."""
        if not self._initialized:
            self.initialize()
            
        if self._backend == 'cuda':
            return process_edges_gpu(gray, params)
        elif self._backend == 'opencl':
            # Convert uint8 to float32 and extract threshold from params
            float_image = np.asarray(gray, dtype=np.float32) / 255.0
            threshold = float(params.get('threshold', 0.1))
            
            # Process edges using OpenCL
            task = EdgeDetectionTask(float_image, threshold)
            ctx = OpenCLContext()
            try:
                result = ctx.execute_task(task)
                edge_result = result.result  # Get the result from the Future
                return {
                    'edge_map': (edge_result * 255).astype(np.uint8),
                    'edge_strength': edge_result.astype(np.uint8),
                    'edge_gradient': np.zeros_like(edge_result, dtype=np.float32)  # Placeholder for compatibility
                }
            finally:
                ctx.release()
        else:
            return process_edges_cpu(gray, params)

# Global GPU manager instance
_gpu_manager = GPUManager()

def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    return _gpu_manager

def start_processing_thread() -> None:
    """Start the background processing thread."""
    global _processing_thread
    if _processing_thread is None:
        _stop_event.clear()
        _processing_thread = threading.Thread(target=_process_queue, daemon=True)
        _processing_thread.start()

def stop_processing_thread() -> None:
    """Stop the background processing thread."""
    global _processing_thread
    if _processing_thread is not None:
        _stop_event.set()
        _processing_thread.join()
        _processing_thread = None

def _process_queue() -> None:
    """Process tasks from the queue."""
    while not _stop_event.is_set():
        try:
            # Get task with timeout to allow checking stop event
            task = _processing_queue.get(timeout=0.1)
            if task is None:
                continue
                
            func, args, callback = task
            try:
                result = func(*args)
                if callback is not None:
                    callback(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                
            _processing_queue.task_done()
            
        except Empty:
            continue

class GPUContext:
    """Context manager for GPU memory management."""
    def __init__(self):
        self.arrays_to_free = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for arr in self.arrays_to_free:
            if isinstance(arr, cp.ndarray):
                arr.device.synchronize()
                del arr
        cp.get_default_memory_pool().free_all_blocks()

def process_edges_gpu(gray: NDArray[np.uint8], params: Dict[str, Any]) -> Dict[str, NDArray]:
    """Process edges using GPU acceleration."""
    try:
        # Input validation
        if gray is None:
            raise ValueError("Input image cannot be None")
        
        if not isinstance(gray, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(gray)}")
            
        if gray.size == 0 or gray.ndim != 2:
            raise ValueError(f"Invalid image dimensions: {gray.shape}")
            
        # Parameter validation with defaults
        d = params.get('d', 5)
        sigmaColor = params.get('sigmaColor', 50.0)
        sigmaSpace = params.get('sigmaSpace', 50.0)
        
        # Validate parameters
        if d < 1 or d % 2 == 0:
            logging.warning(f"Invalid bilateral filter diameter: {d}, using 5 instead")
            d = 5
            
        with GPUContext() as ctx:
            if params.get('operation') == 'nms':
                # Validate NMS specific parameters
                if 'edge_strength' not in params or 'angle' not in params:
                    raise ValueError("Missing required parameters for NMS operation")
                    
                width = params.get('width', gray.shape[1])
                height = params.get('height', gray.shape[0])
                return _apply_nms_gpu(params['edge_strength'], params['angle'], width, height)
                
            # Transfer data to GPU
            d_gray = cp.asarray(gray)
            ctx.arrays_to_free.append(d_gray)

            # Apply bilateral filter on GPU
            try:
                d_gray_filtered = cp.asarray(cv2.bilateralFilter(
                    cp.asnumpy(d_gray),
                    d=d,
                    sigmaColor=sigmaColor,
                    sigmaSpace=sigmaSpace
                ))
                ctx.arrays_to_free.append(d_gray_filtered)
            except cv2.error as e:
                logging.error(f"Bilateral filter failed: {e}")
                # Fallback to Gaussian blur
                d_gray_filtered = cp.asarray(cv2.GaussianBlur(
                    cp.asnumpy(d_gray), (5, 5), 1.5
                ))
                ctx.arrays_to_free.append(d_gray_filtered)

            # Convert to numpy array for OpenCV operations
            gray_filtered_np = cp.asnumpy(d_gray_filtered)
            gray_filtered_arr = np.asarray(gray_filtered_np, dtype=np.float64)
            
            # Compute adaptive thresholds using numpy operations
            if gray_filtered_arr.size > 0:
                mean_intensity = float(np.mean(gray_filtered_arr.astype(np.float64)))
                std_intensity = float(np.std(gray_filtered_arr.astype(np.float64)))
                low_threshold = max(0, mean_intensity - std_intensity)
                high_threshold = min(255, mean_intensity + std_intensity)
            else:
                # Default values if array is empty
                logging.warning("Empty filtered array, using default thresholds")
                low_threshold = 50
                high_threshold = 150

            # Multi-scale edge detection with error handling
            try:
                edges_fine = cv2.Canny(
                    gray_filtered_np,
                    low_threshold,
                    high_threshold,
                    apertureSize=3,
                    L2gradient=True
                )
            except Exception as e:
                logging.error(f"Fine edge detection failed: {e}")
                edges_fine = np.zeros_like(gray_filtered_np, dtype=np.uint8)

            try:
                gray_medium = cv2.GaussianBlur(gray_filtered_np, (5, 5), 1.5)
                edges_medium = cv2.Canny(
                    gray_medium,
                    low_threshold * 0.8,
                    high_threshold * 0.8,
                    apertureSize=3,
                    L2gradient=True
                )
            except Exception as e:
                logging.error(f"Medium edge detection failed: {e}")
                edges_medium = np.zeros_like(gray_filtered_np, dtype=np.uint8)

            try:
                gray_coarse = cv2.GaussianBlur(gray_filtered_np, (9, 9), 2.5)
                edges_coarse = cv2.Canny(
                    gray_coarse,
                    low_threshold * 0.6,
                    high_threshold * 0.6,
                    apertureSize=5,
                    L2gradient=True
                )
            except Exception as e:
                logging.error(f"Coarse edge detection failed: {e}")
                edges_coarse = np.zeros_like(gray_filtered_np, dtype=np.uint8)

            # Combine edges with weighted addition
            try:
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
            except Exception as e:
                logging.error(f"Edge combination failed: {e}")
                # Fallback to using fine edges only
                edge_map = edges_fine.astype(np.float32)

            # Compute gradients with validation
            try:
                gradient_x = cv2.Sobel(gray_filtered_np, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(gray_filtered_np, cv2.CV_32F, 0, 1, ksize=3)
                
                # Compute edge strength and gradient
                edge_strength = np.sqrt(gradient_x**2 + gradient_y**2)
                edge_gradient = np.arctan2(gradient_y, gradient_x)
                
                # Normalize edge strength with validation
                edge_min = float(np.min(edge_strength)) if edge_strength.size > 0 else 0
                edge_max = float(np.max(edge_strength)) if edge_strength.size > 0 else 255
                
                if edge_max > edge_min and edge_strength.size > 0:
                    edge_strength = ((edge_strength - edge_min) * 255.0 / (edge_max - edge_min))
            except Exception as e:
                logging.error(f"Gradient computation failed: {e}")
                # Create fallback values
                edge_strength = np.zeros_like(gray_filtered_np, dtype=np.float32)
                edge_gradient = np.zeros_like(gray_filtered_np, dtype=np.float32)

            # Transfer results back to CPU
            result = {
                'edge_map': edge_map.astype(np.uint8),
                'edge_strength': edge_strength.astype(np.uint8),
                'edge_gradient': edge_gradient
            }
            
            # Final validation of results
            for key, value in result.items():
                if value is None or value.size == 0:
                    logging.warning(f"Empty result for {key}, creating fallback")
                    result[key] = np.zeros_like(gray if key != 'edge_gradient' else gray.astype(np.float32))
                    
            return result

    except Exception as e:
        logging.error(f"GPU processing error: {e}", exc_info=True)
        try:
            return process_edges_cpu(gray, params)
        except Exception as cpu_error:
            logging.error(f"CPU fallback also failed: {cpu_error}", exc_info=True)
            # Last resort fallback
            return {
                'edge_map': np.zeros_like(gray, dtype=np.uint8),
                'edge_strength': np.zeros_like(gray, dtype=np.uint8),
                'edge_gradient': np.zeros_like(gray, dtype=np.float32)
            }

def _apply_nms_gpu(edge_strength: NDArray[np.uint8], angle: NDArray[np.float32], width: int, height: int) -> Dict[str, NDArray]:
    """Apply non-maximum suppression using GPU."""
    try:
        with GPUContext() as ctx:
            d_strength = cp.asarray(edge_strength)
            d_angle = cp.asarray(angle)
            d_suppressed = cp.zeros_like(d_strength)
            ctx.arrays_to_free.extend([d_strength, d_angle, d_suppressed])
            
            # Custom CUDA kernel for non-maximum suppression
            kernel = cp.ElementwiseKernel(
                'T strength, T angle, raw T strength_map',
                'T output',
                '''
                if (i < 1 || i >= strength_map.size() - 1) {
                    output = 0;
                    return;
                }
                
                int w = (int)sqrt(strength_map.size() / sizeof(T));
                int x = i % w;
                int y = i / w;
                
                if (x < 1 || x >= w - 1) {
                    output = 0;
                    return;
                }
                
                T val = strength;
                T n1, n2;
                
                switch ((int)angle) {
                    case 0:  // -45 degrees
                        n1 = strength_map[(y-1)*w + (x-1)];
                        n2 = strength_map[(y+1)*w + (x+1)];
                        break;
                    case 1:  // vertical
                        n1 = strength_map[(y-1)*w + x];
                        n2 = strength_map[(y+1)*w + x];
                        break;
                    case 2:  // 45 degrees
                        n1 = strength_map[(y-1)*w + (x+1)];
                        n2 = strength_map[(y+1)*w + (x-1)];
                        break;
                    default:  // horizontal
                        n1 = strength_map[y*w + (x-1)];
                        n2 = strength_map[y*w + (x+1)];
                        break;
                }
                
                output = (val >= n1 && val >= n2) ? val : 0;
                ''',
                'non_maximum_suppression'
            )
            
            # Apply the kernel
            d_suppressed = kernel(d_strength, d_angle, d_strength)
            suppressed = cp.asnumpy(d_suppressed)
            
            return {
                'edge_strength': suppressed
            }
            
    except Exception as e:
        print(f"GPU NMS error: {e}")
        return {
            'edge_strength': edge_strength
        }

def process_edges_cpu(gray: NDArray[np.uint8], params: Dict[str, Any]) -> Dict[str, NDArray]:
    """Process edges using CPU as fallback."""
    try:
        if params.get('operation') == 'nms':
            return _apply_nms_cpu(params['edge_strength'], params['angle'], params['width'], params['height'])
            
        # Apply bilateral filter
        gray_filtered = cv2.bilateralFilter(
            gray,
            d=params['d'],
            sigmaColor=params['sigmaColor'],
            sigmaSpace=params['sigmaSpace']
        )

        # Compute adaptive thresholds
        mean_intensity = float(np.mean(np.asarray(gray_filtered, dtype=np.float64)))
        std_intensity = float(np.std(np.asarray(gray_filtered, dtype=np.float64)))
        low_threshold = max(0, mean_intensity - std_intensity)
        high_threshold = min(255, mean_intensity + std_intensity)

        # Multi-scale edge detection
        edges_fine = cv2.Canny(
            gray_filtered,
            low_threshold,
            high_threshold,
            apertureSize=3,
            L2gradient=True
        )

        gray_medium = cv2.GaussianBlur(gray_filtered, (5, 5), 1.5)
        edges_medium = cv2.Canny(
            gray_medium,
            low_threshold * 0.8,
            high_threshold * 0.8,
            apertureSize=3,
            L2gradient=True
        )

        gray_coarse = cv2.GaussianBlur(gray_filtered, (9, 9), 2.5)
        edges_coarse = cv2.Canny(
            gray_coarse,
            low_threshold * 0.6,
            high_threshold * 0.6,
            apertureSize=5,
            L2gradient=True
        )

        # Combine edges
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
            'edge_map': edge_map.astype(np.uint8),
            'edge_strength': edge_strength.astype(np.uint8),
            'edge_gradient': edge_gradient
        }

    except Exception as e:
        print(f"CPU processing error: {e}")
        return {
            'edge_map': np.zeros_like(gray, dtype=np.uint8),
            'edge_strength': np.zeros_like(gray, dtype=np.uint8),
            'edge_gradient': np.zeros_like(gray, dtype=np.float32)
        }

def _apply_nms_cpu(edge_strength: NDArray[np.uint8], angle: NDArray[np.float32], width: int, height: int) -> Dict[str, NDArray]:
    """Apply non-maximum suppression using CPU."""
    try:
        suppressed = np.zeros_like(edge_strength)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                angle_val = angle[i, j]
                strength = edge_strength[i, j]
                
                if angle_val == 0:  # -45 degrees
                    neighbors = [edge_strength[i-1, j-1], edge_strength[i+1, j+1]]
                elif angle_val == 1:  # vertical
                    neighbors = [edge_strength[i-1, j], edge_strength[i+1, j]]
                elif angle_val == 2:  # 45 degrees
                    neighbors = [edge_strength[i-1, j+1], edge_strength[i+1, j-1]]
                else:  # horizontal
                    neighbors = [edge_strength[i, j-1], edge_strength[i, j+1]]
                
                if strength >= max(neighbors):
                    suppressed[i, j] = strength
                    
        return {
            'edge_strength': suppressed
        }
        
    except Exception as e:
        print(f"CPU NMS error: {e}")
        return {
            'edge_strength': edge_strength
        } 