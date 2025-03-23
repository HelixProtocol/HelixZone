"""Performance benchmarks for format support operations."""

import os
import pytest
import numpy as np
import time
from PyQt6.QtGui import QImage

from src.helixzone.core.format_support import (
    get_format_support,
    RawFormatHandler,
    HdrFormatHandler,
    RawProcessingParams,
    RawProcessingMode
)

# Setup test data paths (assuming there's a test_data directory)
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
SAMPLE_RAW_FILE = os.path.join(TEST_DATA_DIR, "sample.cr2")
SAMPLE_HDR_FILE = os.path.join(TEST_DATA_DIR, "sample.hdr")

# Mark these tests as slow
pytestmark = pytest.mark.slow

# Define test fixtures and parameters
image_sizes = [
    (1920, 1080),  # Full HD
    (3840, 2160),  # 4K
    (7680, 4320),  # 8K
]

kernel_sizes = [3, 5, 9, 15, 31]


# Utility functions for the benchmarks
def create_test_image(size):
    """Create a test image of the specified size."""
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)


def get_memory_usage():
    """Get the current memory usage of the process."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


class TestFormatSupportPerformance:
    """Performance benchmarks for format support operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        self.format_support = get_format_support()
        
        # Create directories if they don't exist
        self.output_dir = os.path.join(TEST_DATA_DIR, "benchmark_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store performance results
        self.results = []
        yield
        # Save results to file after tests
        self._save_results()
    
    def _save_results(self):
        """Save benchmark results to file."""
        if not self.results:
            return
        
        result_path = os.path.join(self.output_dir, "benchmark_results.csv")
        with open(result_path, "w") as f:
            f.write("test,parameters,duration_ms,memory_mb\n")
            for result in self.results:
                f.write(f"{result['test']},{result['parameters']},{result['duration_ms']:.2f},{result['memory_mb']:.2f}\n")

    @pytest.mark.skipif(not os.path.exists(SAMPLE_RAW_FILE), reason="Sample RAW file not found")
    @pytest.mark.skipif(not RawFormatHandler.is_raw_file(SAMPLE_RAW_FILE) if os.path.exists(SAMPLE_RAW_FILE) else True,
                       reason="RAW support not available")
    @pytest.mark.parametrize("processing_mode", [
        RawProcessingMode.LINEAR,
        RawProcessingMode.SRGB,
        RawProcessingMode.CUSTOM
    ])
    def test_raw_image_loading_performance(self, benchmark, processing_mode):
        """Benchmark RAW image loading performance with different processing modes."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_RAW_FILE):
            pytest.skip(f"Sample RAW file not found: {SAMPLE_RAW_FILE}")
        
        params = RawProcessingParams(mode=processing_mode)
        options = {
            'raw_processing_mode': params.mode,
            'brightness': params.brightness,
            'contrast': params.contrast,
            'saturation': params.saturation,
            'auto_white_balance': params.auto_white_balance
        }
        
        # Define the operation to benchmark
        def load_raw_image():
            mem_before = get_memory_usage()
            start_time = time.time()
            image = self.format_support.load_image(SAMPLE_RAW_FILE, options)
            duration = (time.time() - start_time) * 1000  # ms
            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before
            
            # Store results
            self.results.append({
                'test': 'raw_loading',
                'parameters': f"mode={processing_mode.name}",
                'duration_ms': duration,
                'memory_mb': mem_usage
            })
            
            return image
        
        # Run the benchmark
        image = benchmark(load_raw_image)
        
        # Verify the image was loaded successfully
        assert image is not None
        assert isinstance(image, QImage)
        assert not image.isNull()

    @pytest.mark.skipif(not os.path.exists(SAMPLE_HDR_FILE), reason="Sample HDR file not found")
    @pytest.mark.skipif(not HdrFormatHandler.is_hdr_file(SAMPLE_HDR_FILE) if os.path.exists(SAMPLE_HDR_FILE) else True,
                       reason="HDR support not available")
    @pytest.mark.parametrize("tone_map", [True, False])
    @pytest.mark.parametrize("exposure", [-1.0, 0.0, 1.0])
    def test_hdr_image_loading_performance(self, benchmark, tone_map, exposure):
        """Benchmark HDR image loading performance with different settings."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_HDR_FILE):
            pytest.skip(f"Sample HDR file not found: {SAMPLE_HDR_FILE}")
        
        options = {
            'tone_map': tone_map,
            'exposure': exposure
        }
        
        # Define the operation to benchmark
        def load_hdr_image():
            mem_before = get_memory_usage()
            start_time = time.time()
            image = self.format_support.load_image(SAMPLE_HDR_FILE, options)
            duration = (time.time() - start_time) * 1000  # ms
            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before
            
            # Store results
            self.results.append({
                'test': 'hdr_loading',
                'parameters': f"tone_map={tone_map},exposure={exposure}",
                'duration_ms': duration,
                'memory_mb': mem_usage
            })
            
            return image
        
        # Run the benchmark
        image = benchmark(load_hdr_image)
        
        # Verify the image was loaded successfully
        assert image is not None
        assert isinstance(image, QImage)
        assert not image.isNull()

    @pytest.mark.parametrize("image_size", image_sizes)
    def test_image_processing_scalability(self, benchmark, image_size):
        """Benchmark image processing scalability with different image sizes."""
        # Create a test image
        image_data = create_test_image(image_size)
        
        # Convert to QImage
        height, width, channels = image_data.shape
        bytes_per_line = channels * width
        qimage = QImage(image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Define the operation to benchmark
        def process_image():
            mem_before = get_memory_usage()
            start_time = time.time()
            
            # Simulate some processing (e.g., color conversion)
            # In a real benchmark, this would call actual processing functions
            processed_image = QImage(qimage)
            for y in range(processed_image.height()):
                for x in range(processed_image.width()):
                    color = processed_image.pixelColor(x, y)
                    # Convert to grayscale
                    gray = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
                    processed_image.setPixelColor(x, y, QImage.fromRgb(gray, gray, gray))
            
            duration = (time.time() - start_time) * 1000  # ms
            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before
            
            # Calculate pixels per second
            pixel_count = width * height
            pixels_per_ms = pixel_count / duration
            
            # Store results
            self.results.append({
                'test': 'image_processing',
                'parameters': f"size={width}x{height}",
                'duration_ms': duration,
                'memory_mb': mem_usage,
                'pixels_per_ms': pixels_per_ms
            })
            
            return processed_image
        
        # Run the benchmark
        result = benchmark(process_image)
        
        # Verify the result
        assert result is not None
        assert isinstance(result, QImage)
        assert not result.isNull()

    @pytest.mark.parametrize("threads", [1, 2, 4, 8])
    def test_threading_performance(self, benchmark, threads):
        """Benchmark performance with different thread counts."""
        # This is a placeholder for threading performance tests
        # In a real implementation, this would use the actual threading mechanism
        
        # For now, we'll just simulate the work
        def threaded_work():
            mem_before = get_memory_usage()
            start_time = time.time()
            
            # Simulate work that would be distributed across threads
            total_work = 10_000_000
            work_per_thread = total_work // threads
            
            # Simulate the work (just some busy work)
            result = 0
            for i in range(work_per_thread):
                result += i % 10
            
            duration = (time.time() - start_time) * 1000  # ms
            mem_after = get_memory_usage()
            mem_usage = mem_after - mem_before
            
            # Store results
            self.results.append({
                'test': 'threading',
                'parameters': f"threads={threads}",
                'duration_ms': duration,
                'memory_mb': mem_usage,
                'work_units': total_work,
                'units_per_ms': total_work / duration
            })
            
            return result
        
        # Run the benchmark
        result = benchmark(threaded_work)
        
        # This is just a placeholder verification
        assert result is not None 