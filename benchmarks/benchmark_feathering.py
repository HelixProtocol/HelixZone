"""
Comprehensive benchmarking suite for HelixZone feathering operations.
"""

import cv2
import numpy as np
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, TypedDict, cast, Any

try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from helixzone.core.ml_utils import EnhancedLassoFeathering

class ColorProcessingResults(TypedDict):
    standard_time: float
    color_aware_time: float
    overhead: float

class BatchProcessingResult(TypedDict):
    batch_size: int
    sequential_time: float
    parallel_time: float
    speedup: float

class SizeScalingResult(TypedDict):
    size: str
    megapixels: float
    cpu_time: float
    gpu_time: float
    ram_usage: float
    gpu_usage: float

BenchmarkResults = Dict[str, Union[List[Union[SizeScalingResult, BatchProcessingResult]], ColorProcessingResults]]

class FeatheringBenchmark:
    def __init__(self):
        self.feathering = EnhancedLassoFeathering()
        self.results: BenchmarkResults = {}
        
    def _create_test_image(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create test image and mask of specified size."""
        image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 4
        cv2.circle(mask, center, radius, (255,), -1)
        return image, mask
        
    def _measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        gpu_mem = 0.0
        if HAS_GPU and GPUtil.getGPUs():
            gpu = GPUtil.getGPUs()[0]
            gpu_mem = float(gpu.memoryUsed)
        return {
            "ram_mb": float(mem_info.rss) / (1024 * 1024),
            "gpu_mb": gpu_mem
        }
        
    def benchmark_size_scaling(self) -> None:
        """Benchmark performance across different image sizes."""
        sizes = [
            (512, 512),    # 0.25MP
            (1024, 1024),  # 1MP
            (2048, 2048),  # 4MP
            (4096, 4096)   # 16MP
        ]
        
        results: List[SizeScalingResult] = []
        for size in sizes:
            image, mask = self._create_test_image(size)
            
            # Measure CPU performance
            start_mem = self._measure_memory()
            start_time = time.time()
            _ = self.feathering.apply_lasso_feathering(image, mask)
            cpu_time = time.time() - start_time
            end_mem = self._measure_memory()
            
            # Measure GPU performance if available
            gpu_time = 0.0
            try:
                if hasattr(self.feathering, 'enable_gpu'):
                    self.feathering.enable_gpu()
                    start_time = time.time()
                    _ = self.feathering.apply_lasso_feathering(image, mask)
                    gpu_time = time.time() - start_time
            except Exception:
                pass  # GPU acceleration not available
            
            results.append(SizeScalingResult(
                size=f"{size[0]}x{size[1]}",
                megapixels=float(size[0] * size[1]) / (1024 * 1024),
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                ram_usage=end_mem["ram_mb"] - start_mem["ram_mb"],
                gpu_usage=end_mem["gpu_mb"] - start_mem["gpu_mb"]
            ))
            
        self.results["size_scaling"] = cast(Any, results)
        
    def benchmark_color_processing(self) -> None:
        """Benchmark color-aware vs standard processing."""
        image, mask = self._create_test_image((2048, 2048))
        
        # Standard processing
        start_time = time.time()
        _ = self.feathering.apply_lasso_feathering(image, mask)
        standard_time = time.time() - start_time
        
        # Color-aware processing
        start_time = time.time()
        _ = self.feathering.apply_color_aware_feathering(image, mask)
        color_time = time.time() - start_time
        
        self.results["color_processing"] = ColorProcessingResults(
            standard_time=standard_time,
            color_aware_time=color_time,
            overhead=(color_time / standard_time) - 1
        )
        
    def benchmark_batch_processing(self) -> None:
        """Benchmark batch processing performance."""
        batch_sizes = [1, 2, 4, 8]
        image, mask = self._create_test_image((1024, 1024))
        
        results: List[BatchProcessingResult] = []
        for batch_size in batch_sizes:
            images = [image] * batch_size
            masks = [mask] * batch_size
            
            # Sequential processing
            start_time = time.time()
            for img, msk in zip(images, masks):
                _ = self.feathering.apply_lasso_feathering(img, msk)
            seq_time = time.time() - start_time
            
            # Parallel processing
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                _ = list(executor.map(
                    lambda x: self.feathering.apply_lasso_feathering(*x),
                    zip(images, masks)
                ))
            parallel_time = time.time() - start_time
            
            results.append(BatchProcessingResult(
                batch_size=batch_size,
                sequential_time=seq_time,
                parallel_time=parallel_time,
                speedup=seq_time / parallel_time
            ))
            
        self.results["batch_processing"] = cast(Any, results)
        
    def run_all_benchmarks(self) -> None:
        """Run all benchmarks and save results."""
        print("Running size scaling benchmarks...")
        self.benchmark_size_scaling()
        
        print("Running color processing benchmarks...")
        self.benchmark_color_processing()
        
        print("Running batch processing benchmarks...")
        self.benchmark_batch_processing()
        
        self._save_results()
        
    def _save_results(self) -> None:
        """Save benchmark results to markdown file."""
        output = "# HelixZone Benchmark Results\n\n"
        
        # Size scaling results
        output += "## Image Size Scaling\n\n"
        output += "| Size | CPU Time (s) | GPU Time (s) | RAM Usage (MB) | GPU Usage (MB) |\n"
        output += "|------|-------------|--------------|----------------|----------------|\n"
        size_scaling = cast(List[SizeScalingResult], self.results["size_scaling"])
        for result in size_scaling:
            output += f"| {result['size']} | {result['cpu_time']:.3f} | {result['gpu_time']:.3f} "
            output += f"| {result['ram_usage']:.1f} | {result['gpu_usage']:.1f} |\n"
            
        # Color processing results
        output += "\n## Color Processing Overhead\n\n"
        color_results = cast(ColorProcessingResults, self.results["color_processing"])
        output += f"- Standard processing: {color_results['standard_time']:.3f}s\n"
        output += f"- Color-aware processing: {color_results['color_aware_time']:.3f}s\n"
        output += f"- Overhead: {color_results['overhead']*100:.1f}%\n"
        
        # Batch processing results
        output += "\n## Batch Processing Performance\n\n"
        output += "| Batch Size | Sequential (s) | Parallel (s) | Speedup |\n"
        output += "|------------|----------------|--------------|----------|\n"
        batch_results = cast(List[BatchProcessingResult], self.results["batch_processing"])
        for result in batch_results:
            output += f"| {result['batch_size']} | {result['sequential_time']:.3f} "
            output += f"| {result['parallel_time']:.3f} | {result['speedup']:.2f}x |\n"
            
        # Save to file
        Path("benchmarks/results.md").write_text(output)
        print("Results saved to benchmarks/results.md")

if __name__ == "__main__":
    benchmark = FeatheringBenchmark()
    benchmark.run_all_benchmarks() 