"""Batch processing module for edge detection."""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, Future
from .gpu import get_gpu_manager, GPUManager

class BatchProcessor:
    """Batch processor for edge detection."""
    def __init__(self, max_workers: int = 4):
        self._gpu_manager: GPUManager = get_gpu_manager()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: List[Future] = []
        self._batch_size = 4  # Process 4 images at a time
        
    def process_batch(self, images: List[NDArray[np.uint8]], params: Dict[str, Any]) -> List[Dict[str, NDArray]]:
        """Process a batch of images."""
        results = []
        for i in range(0, len(images), self._batch_size):
            batch = images[i:i + self._batch_size]
            futures = [
                self._executor.submit(self._gpu_manager.process_edges, img, params)
                for img in batch
            ]
            self._futures.extend(futures)
            
            # Wait for this batch to complete
            batch_results = [f.result() for f in futures]
            results.extend(batch_results)
            
            # Remove completed futures
            self._futures = [f for f in self._futures if not f.done()]
        
        return results

    def process_edges_async(self, image: NDArray[np.uint8], params: Dict[str, Any]) -> Future:
        """Process edges asynchronously."""
        future = self._executor.submit(self._gpu_manager.process_edges, image, params)
        self._futures.append(future)
        return future
        
    def wait_all(self) -> None:
        """Wait for all pending operations to complete."""
        for future in self._futures:
            future.result()
        self._futures.clear()
        
    def shutdown(self) -> None:
        """Shutdown the batch processor."""
        self.wait_all()
        self._executor.shutdown()

class BatchContext:
    """Context manager for batch processing."""
    def __init__(self, max_workers: int = 4):
        self.processor: Optional[BatchProcessor] = None
        self.max_workers = max_workers
        
    def __enter__(self) -> BatchProcessor:
        self.processor = BatchProcessor(max_workers=self.max_workers)
        return self.processor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processor:
            self.processor.shutdown()
            self.processor = None

def process_edges_batch(images: List[NDArray[np.uint8]], params: Dict[str, Any], max_workers: int = 4) -> List[Dict[str, NDArray]]:
    """Process multiple images in batches."""
    with BatchContext(max_workers=max_workers) as processor:
        return processor.process_batch(images, params)

def process_edges_async(image: NDArray[np.uint8], params: Dict[str, Any], processor: BatchProcessor) -> Future:
    """Process edges asynchronously using the provided batch processor."""
    return processor.process_edges_async(image, params)

def wait_for_results(futures: List[Future]) -> List[Dict[str, NDArray]]:
    """Wait for a list of futures to complete and return their results."""
    return [f.result() for f in futures] 