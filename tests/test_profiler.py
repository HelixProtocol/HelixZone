"""Tests for the performance profiling system."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from unittest.mock import patch, MagicMock
import psutil

from benchmarks.profile_performance import (
    PerformanceProfiler,
    MemoryMetrics,
    OperationMetrics,
    ThresholdConfig,
    ThresholdManager
)

class TestThresholdManager(unittest.TestCase):
    """Test suite for ThresholdManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ThresholdConfig(
            memory_ratio=2.0,
            memory_release=0.9,
            memory_retention=30.0,
            duration_outlier_std=1.5,
            gpu_utilization=0.6,
            cpu_threshold=70.0
        )
        self.manager = ThresholdManager(self.config)
        
    def test_threshold_initialization(self):
        """Test threshold manager initialization."""
        self.assertEqual(self.manager.config.memory_ratio, 2.0)
        self.assertEqual(self.manager.config.memory_release, 0.9)
        self.assertEqual(len(self.manager.history), 0)
        
    def test_update_history(self):
        """Test metric history updates."""
        self.manager.update_history('memory_ratio', 1.5)
        self.manager.update_history('memory_ratio', 2.5)
        
        self.assertEqual(len(self.manager.history['memory_ratio']), 2)
        self.assertEqual(self.manager.history['memory_ratio'][0], 1.5)
        
    def test_dynamic_threshold(self):
        """Test dynamic threshold calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            self.manager.update_history('test_metric', v)
            
        threshold = self.manager.get_dynamic_threshold('test_metric')
        mean = np.mean(values)
        std = np.std(values)
        expected = mean + 2 * std
        
        self.assertAlmostEqual(threshold, expected, places=6)
        
    def test_optimization_suggestions(self):
        """Test optimization suggestions generation."""
        metrics = OperationMetrics(
            operation="test_op",
            duration=1.0,
            cpu_percent=90.0,  # High CPU usage
            memory_used=300.0,  # 3x image size
            gpu_used=5.0  # Low GPU utilization
        )
        
        suggestions = self.manager.get_optimization_suggestions(metrics, 100.0)
        
        self.assertTrue(any("CPU" in s for s in suggestions))
        self.assertTrue(any("memory" in s.lower() for s in suggestions))
        self.assertTrue(any("GPU" in s for s in suggestions))

class TestPerformanceProfiler(unittest.TestCase):
    """Test suite for PerformanceProfiler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_create_test_data(self):
        """Test test data generation."""
        images, masks = self.profiler._create_test_data()
        
        self.assertIn('simple', images)
        self.assertIn('complex', images)
        self.assertIn('high_res', images)
        
        self.assertEqual(images['simple'].shape[:2], (1024, 1024))
        self.assertEqual(images['high_res'].shape[:2], (4096, 4096))
        
    def test_memory_metrics_recording(self):
        """Test memory metrics recording."""
        metric = MemoryMetrics(
            operation="test_op",
            image_name="test_image",
            pre_memory=100.0,
            post_memory=150.0,
            peak_memory=200.0,
            total_allocated=300.0,
            total_freed=250.0,
            timestamp=datetime.now()
        )
        
        self.profiler.memory_metrics.append(metric)
        self.assertEqual(len(self.profiler.memory_metrics), 1)
        self.assertEqual(self.profiler.memory_metrics[0].operation, "test_op")
        
    def test_operation_metrics_recording(self):
        """Test operation metrics recording."""
        metric = OperationMetrics(
            operation="test_op",
            duration=1.5,
            cpu_percent=60.0,
            memory_used=100.0,
            gpu_used=50.0
        )
        
        self.profiler.operation_metrics.append(metric)
        self.assertEqual(len(self.profiler.operation_metrics), 1)
        self.assertEqual(self.profiler.operation_metrics[0].duration, 1.5)
        
    def test_profile_output_generation(self):
        """Test profile output file generation."""
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_cpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Check if profile file was created
        profile_files = list(self.profiler.output_dir.glob("cpu_profile_*.txt"))
        self.assertTrue(len(profile_files) > 0)
        
        # Check file content
        content = profile_files[0].read_text()
        self.assertIn("Profile", content)

class TestGPUProfiling(unittest.TestCase):
    """Test suite for GPU profiling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_gpu_memory_tracking(self, mock_memory_allocated, mock_is_available):
        """Test GPU memory tracking functionality."""
        mock_is_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify GPU metrics were recorded
        gpu_metrics = [m for m in self.profiler.operation_metrics if m.gpu_used is not None]
        self.assertTrue(len(gpu_metrics) > 0)
        
        # Verify GPU memory value
        gpu_memory = gpu_metrics[0].gpu_used
        self.assertIsNotNone(gpu_memory)
        if gpu_memory is not None:  # Type guard for mypy
            self.assertGreater(gpu_memory, 0.0)
        
    @patch('torch.profiler.profile')
    def test_gpu_profiler_configuration(self, mock_profiler):
        """Test GPU profiler setup and configuration."""
        mock_profiler.return_value = MagicMock()
        
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify profiler was configured correctly
        mock_profiler.assert_called_once()
        args, kwargs = mock_profiler.call_args
        self.assertIn('record_shapes', kwargs)
        self.assertIn('profile_memory', kwargs)
        self.assertTrue(kwargs['record_shapes'])
        self.assertTrue(kwargs['profile_memory'])
        
    def test_gpu_visualization_generation(self):
        """Test GPU memory visualization generation."""
        # Add test GPU metrics
        self.profiler.operation_metrics.append(
            OperationMetrics(
                operation="GPU_test",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=200.0
            )
        )
        
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Check if visualization was generated
        timeline_files = list(self.profiler.output_dir.glob("gpu_memory_timeline_*.png"))
        self.assertTrue(len(timeline_files) > 0)

class TestCorrelationAnalysis(unittest.TestCase):
    """Test suite for correlation analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
        # Add test metrics
        self.profiler.operation_metrics.extend([
            OperationMetrics(
                operation="op1",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=150.0
            ),
            OperationMetrics(
                operation="op2",
                duration=2.0,
                cpu_percent=75.0,
                memory_used=200.0,
                gpu_used=250.0
            )
        ])
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_correlation_calculation(self):
        """Test correlation calculation between metrics."""
        self.profiler.analyze_correlations()
        
        # Check if correlation analysis file was generated
        analysis_files = list(self.profiler.output_dir.glob("correlation_analysis_*.md"))
        self.assertTrue(len(analysis_files) > 0)
        
        # Verify correlation content
        content = analysis_files[0].read_text()
        self.assertIn("Metric Correlations", content)
        self.assertIn("Duration vs CPU Usage", content)
        self.assertIn("Duration vs Memory Usage", content)
        
    def test_correlation_visualization(self):
        """Test correlation visualization generation."""
        self.profiler.analyze_correlations()
        
        # Check if heatmap was generated
        heatmap_files = list(self.profiler.output_dir.glob("correlation_heatmap_*.png"))
        self.assertTrue(len(heatmap_files) > 0)
        
    def test_insights_generation(self):
        """Test performance insights generation."""
        self.profiler.analyze_correlations()
        
        analysis_files = list(self.profiler.output_dir.glob("correlation_analysis_*.md"))
        content = analysis_files[0].read_text()
        
        # Verify insights sections
        self.assertIn("Performance Insights", content)
        self.assertIn("Recommendations", content)

class TestMemoryLeakDetection(unittest.TestCase):
    """Test suite for memory leak detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_leak_detection(self):
        """Test memory leak detection in allocation profiling."""
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_memory_allocation({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Check if allocation profile was generated
        profile_files = list(self.profiler.output_dir.glob("memory_allocation_*.txt"))
        self.assertTrue(len(profile_files) > 0)
        
        # Verify leak detection content
        content = profile_files[0].read_text()
        self.assertIn("Memory Leak", content)
        
    def test_leak_threshold_validation(self):
        """Test memory leak threshold validation."""
        config = ThresholdConfig(memory_retention=50.0)  # 50MB threshold
        self.profiler.threshold_manager = ThresholdManager(config)
        
        metrics = OperationMetrics(
            operation="test_op",
            duration=1.0,
            cpu_percent=50.0,
            memory_used=100.0  # Exceeds threshold
        )
        
        suggestions = self.profiler.threshold_manager.get_optimization_suggestions(
            metrics,
            image_size=10.0
        )
        
        self.assertTrue(any("memory" in s.lower() for s in suggestions))
        
    def test_leak_reporting(self):
        """Test memory leak reporting in detailed profiling."""
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_memory_detailed({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Check if detailed profile was generated
        profile_files = list(self.profiler.output_dir.glob("memory_profile_detailed_*.txt"))
        self.assertTrue(len(profile_files) > 0)
        
        # Verify memory change reporting
        content = profile_files[0].read_text()
        self.assertIn("Memory change:", content)

class TestVisualizationGeneration(unittest.TestCase):
    """Test suite for visualization generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
        # Add test memory metrics
        self.profiler.memory_metrics.extend([
            MemoryMetrics(
                operation="op1",
                image_name="test",
                pre_memory=100.0,
                post_memory=150.0,
                peak_memory=200.0,
                total_allocated=300.0,
                total_freed=250.0,
                timestamp=datetime.now()
            ),
            MemoryMetrics(
                operation="op2",
                image_name="test",
                pre_memory=150.0,
                post_memory=200.0,
                peak_memory=250.0,
                total_allocated=400.0,
                total_freed=350.0,
                timestamp=datetime.now()
            )
        ])
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_memory_timeline_generation(self):
        """Test memory usage timeline visualization."""
        self.profiler.visualize_memory_usage()
        
        # Check if timeline was generated
        timeline_files = list(self.profiler.output_dir.glob("memory_timeline_*.png"))
        self.assertTrue(len(timeline_files) > 0)
        
    def test_peak_memory_visualization(self):
        """Test peak memory visualization."""
        self.profiler.visualize_memory_usage()
        
        # Check if peak memory plot was generated
        peak_files = list(self.profiler.output_dir.glob("peak_memory_*.png"))
        self.assertTrue(len(peak_files) > 0)
        
    def test_allocation_pattern_visualization(self):
        """Test memory allocation pattern visualization."""
        self.profiler.visualize_memory_usage()
        
        # Check if allocation pattern plot was generated
        pattern_files = list(self.profiler.output_dir.glob("allocation_pattern_*.png"))
        self.assertTrue(len(pattern_files) > 0)
        
    def test_html_report_generation(self):
        """Test HTML report generation."""
        self.profiler.visualize_memory_usage()
        
        # Check if HTML report was generated
        report_files = list(self.profiler.output_dir.glob("memory_report_*.html"))
        self.assertTrue(len(report_files) > 0)
        
        # Verify report content
        content = report_files[0].read_text()
        self.assertIn("Memory Profile Visualization Report", content)
        self.assertIn("Key Findings", content)

class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_empty_metrics_handling(self):
        """Test handling of various empty metric scenarios."""
        # Test with completely empty metrics
        self.profiler.memory_metrics = []
        self.profiler.operation_metrics = []
        self.profiler.visualize_memory_usage()
        self.profiler.analyze_correlations()
        
        # Test with single empty metric
        self.profiler.memory_metrics = [
            MemoryMetrics(
                operation="",  # Empty operation name
                image_name="",  # Empty image name
                pre_memory=0.0,
                post_memory=0.0,
                peak_memory=0.0,
                total_allocated=0.0,
                total_freed=0.0,
                timestamp=datetime.now()
            )
        ]
        self.profiler.visualize_memory_usage()
        
        # Test with None values
        self.profiler.operation_metrics = [
            OperationMetrics(
                operation=None,  # type: ignore
                duration=None,  # type: ignore
                cpu_percent=None,  # type: ignore
                memory_used=None,  # type: ignore
                gpu_used=None
            )
        ]
        self.profiler.analyze_correlations()
        
        # Verify reports are still generated
        report_files = list(self.profiler.output_dir.glob("memory_report_*.html"))
        self.assertTrue(len(report_files) > 0)
        
    def test_invalid_image_data_scenarios(self):
        """Test handling of various invalid image data scenarios."""
        # Test with empty image
        empty_image = np.array([], dtype=np.uint8)
        empty_mask = np.array([], dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.profiler.profile_cpu(
                {"test": empty_image}, 
                {"test": empty_mask}
            )
            
        # Test with wrong number of channels
        invalid_channels = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)  # 4 channels
        valid_mask = np.zeros((100, 100), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.profiler.profile_cpu(
                {"test": invalid_channels}, 
                {"test": valid_mask}
            )
            
        # Test with wrong data type
        invalid_type = np.random.rand(100, 100, 3)  # float instead of uint8
        
        with self.assertRaises(ValueError):
            self.profiler.profile_cpu(
                {"test": invalid_type}, 
                {"test": valid_mask}
            )
            
        # Test with zero-sized dimension
        zero_dim = np.zeros((0, 100, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.profiler.profile_cpu(
                {"test": zero_dim}, 
                {"test": valid_mask}
            )
            
    def test_mismatched_dimensions_scenarios(self):
        """Test handling of various mismatched dimension scenarios."""
        base_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test with different height
        mask_wrong_height = np.zeros((50, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.profiler.profile_memory_detailed(
                {"test": base_image}, 
                {"test": mask_wrong_height}
            )
            
        # Test with different width
        mask_wrong_width = np.zeros((100, 50), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.profiler.profile_memory_detailed(
                {"test": base_image}, 
                {"test": mask_wrong_width}
            )
            
        # Test with 3D mask
        mask_wrong_dims = np.zeros((100, 100, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.profiler.profile_memory_detailed(
                {"test": base_image}, 
                {"test": mask_wrong_dims}
            )
            
        # Test with mismatched image/mask pairs
        with self.assertRaises(ValueError):
            self.profiler.profile_memory_detailed(
                {"test1": base_image}, 
                {"test2": np.zeros((100, 100), dtype=np.uint8)}  # Different keys
            )
            
    @patch('psutil.Process')
    def test_memory_access_error_scenarios(self, mock_process):
        """Test handling of various memory access error scenarios."""
        # Test AccessDenied error
        mock_process.side_effect = psutil.AccessDenied()
        images, masks = self.profiler._create_test_data()
        self.profiler.profile_memory_detailed(
            {"test": images["simple"]}, 
            {"test": masks["simple"]}
        )
        
        # Test NoSuchProcess error
        mock_process.side_effect = psutil.NoSuchProcess(0)
        self.profiler.profile_memory_detailed(
            {"test": images["simple"]}, 
            {"test": masks["simple"]}
        )
        
        # Test TimeoutError
        mock_process.side_effect = TimeoutError()
        self.profiler.profile_memory_detailed(
            {"test": images["simple"]}, 
            {"test": masks["simple"]}
        )
        
        # Verify all errors were logged
        profile_files = list(self.profiler.output_dir.glob("memory_profile_detailed_*.txt"))
        content = profile_files[0].read_text()
        self.assertIn("Error", content)
        
    def test_corrupted_metrics_scenarios(self):
        """Test handling of various corrupted metrics scenarios."""
        # Test with invalid memory values
        self.profiler.memory_metrics.extend([
            MemoryMetrics(
                operation="corrupted1",
                image_name="test",
                pre_memory=-1.0,  # Invalid negative
                post_memory=float('inf'),  # Invalid infinite
                peak_memory=float('nan'),  # Invalid NaN
                total_allocated=0.0,
                total_freed=0.0,
                timestamp=datetime.now()
            ),
            MemoryMetrics(
                operation="corrupted2",
                image_name="test",
                pre_memory=100.0,
                post_memory=50.0,  # Invalid: post < pre
                peak_memory=25.0,  # Invalid: peak < post
                total_allocated=200.0,
                total_freed=300.0,  # Invalid: freed > allocated
                timestamp=datetime.now()
            )
        ])
        
        # Test with invalid operation metrics
        self.profiler.operation_metrics.extend([
            OperationMetrics(
                operation="invalid1",
                duration=-1.0,  # Invalid negative duration
                cpu_percent=150.0,  # Invalid CPU percentage
                memory_used=float('inf'),  # Invalid infinite memory
                gpu_used=-100.0  # Invalid negative GPU usage
            ),
            OperationMetrics(
                operation="invalid2",
                duration=float('nan'),  # Invalid NaN duration
                cpu_percent=float('inf'),  # Invalid infinite CPU
                memory_used=-1.0,  # Invalid negative memory
                gpu_used=float('nan')  # Invalid NaN GPU usage
            )
        ])
        
        # Should handle all corrupted data gracefully
        self.profiler.visualize_memory_usage()
        self.profiler.analyze_correlations()
        
        # Verify visualizations and reports were generated
        timeline_files = list(self.profiler.output_dir.glob("memory_timeline_*.png"))
        self.assertTrue(len(timeline_files) > 0)
        
        # Verify corrupted data warnings were logged
        report_files = list(self.profiler.output_dir.glob("memory_report_*.html"))
        content = report_files[0].read_text()
        self.assertIn("invalid", content.lower())
        
    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    def test_visualization_error_scenarios(self, mock_heatmap, mock_savefig):
        """Test handling of various visualization error scenarios."""
        # Test permission error
        mock_savefig.side_effect = PermissionError()
        self.profiler.visualize_memory_usage()
        
        # Test memory error
        mock_savefig.side_effect = MemoryError()
        self.profiler.visualize_memory_usage()
        
        # Test plotting error
        mock_heatmap.side_effect = ValueError("Invalid data for heatmap")
        self.profiler.analyze_correlations()
        
        # Test file system error
        mock_savefig.side_effect = OSError("No space left on device")
        self.profiler.visualize_memory_usage()
        
        # Verify error reports were generated
        error_files = list(self.profiler.output_dir.glob("visualization_error_*.txt"))
        self.assertTrue(len(error_files) > 0)
        
    def test_threshold_edge_cases_scenarios(self):
        """Test various threshold edge cases scenarios."""
        # Test extreme threshold values
        configs = [
            ThresholdConfig(
                memory_ratio=0.0,
                memory_release=0.0,
                memory_retention=0.0,
                duration_outlier_std=0.0,
                gpu_utilization=0.0,
                cpu_threshold=0.0
            ),
            ThresholdConfig(
                memory_ratio=float('inf'),
                memory_release=1.0,
                memory_retention=float('inf'),
                duration_outlier_std=float('inf'),
                gpu_utilization=1.0,
                cpu_threshold=100.0
            ),
            ThresholdConfig(
                memory_ratio=-1.0,  # Invalid negative
                memory_release=2.0,  # Invalid > 1
                memory_retention=float('nan'),  # Invalid NaN
                duration_outlier_std=-1.0,  # Invalid negative
                gpu_utilization=1.5,  # Invalid > 1
                cpu_threshold=150.0  # Invalid > 100
            )
        ]
        
        # Test with various metrics
        metrics = [
            OperationMetrics(
                operation="edge1",
                duration=0.0,
                cpu_percent=0.0,
                memory_used=0.0,
                gpu_used=0.0
            ),
            OperationMetrics(
                operation="edge2",
                duration=float('inf'),
                cpu_percent=100.0,
                memory_used=float('inf'),
                gpu_used=float('inf')
            ),
            OperationMetrics(
                operation="edge3",
                duration=-1.0,
                cpu_percent=150.0,
                memory_used=-100.0,
                gpu_used=-50.0
            )
        ]
        
        for config in configs:
            self.profiler.threshold_manager = ThresholdManager(config)
            for metric in metrics:
                # Should handle edge cases without errors
                suggestions = self.profiler.threshold_manager.get_optimization_suggestions(
                    metric,
                    image_size=1.0
                )
                self.assertIsInstance(suggestions, list)
                
        # Verify threshold violations are properly reported
        self.profiler.visualize_memory_usage()
        report_files = list(self.profiler.output_dir.glob("memory_report_*.html"))
        content = report_files[0].read_text()
        self.assertIn("threshold", content.lower())
        self.assertIn("warning", content.lower())

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    def test_gpu_error_scenarios(self, mock_max_mem, mock_mem_allocated, mock_is_available):
        """Test handling of various GPU error scenarios."""
        images, masks = self.profiler._create_test_data()
        
        # Test GPU not available
        mock_is_available.return_value = False
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify graceful fallback
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        self.assertTrue(len(profile_files) > 0)
        content = profile_files[0].read_text()
        self.assertIn("GPU not available", content)
        
        # Test GPU memory allocation error
        mock_is_available.return_value = True
        mock_mem_allocated.side_effect = RuntimeError("CUDA out of memory")
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify OOM error handling
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("CUDA out of memory", content)
        
        # Test GPU driver error
        mock_mem_allocated.side_effect = RuntimeError("CUDA driver error")
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify driver error handling
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("CUDA driver error", content)
        
        # Test GPU memory inconsistency
        mock_mem_allocated.side_effect = None
        mock_mem_allocated.return_value = 1024 * 1024 * 100  # 100MB
        mock_max_mem.return_value = 1024 * 1024 * 50  # 50MB (invalid: max < current)
        
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify inconsistency warning
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("memory inconsistency", content.lower())
        
    @patch('torch.cuda.is_available')
    @patch('torch.profiler.profile')
    def test_gpu_profiler_error_scenarios(self, mock_profiler, mock_is_available):
        """Test handling of GPU profiler error scenarios."""
        mock_is_available.return_value = True
        images, masks = self.profiler._create_test_data()
        
        # Test profiler initialization error
        mock_profiler.side_effect = RuntimeError("CUDA initialization error")
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify initialization error handling
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("initialization error", content.lower())
        
        # Test profiler memory error
        mock_profiler.side_effect = MemoryError("Insufficient system memory")
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify memory error handling
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("memory error", content.lower())
        
        # Test profiler timeout
        mock_profiler.side_effect = TimeoutError("Profiler operation timeout")
        self.profiler.profile_gpu({"test": images["simple"]}, {"test": masks["simple"]})
        
        # Verify timeout handling
        profile_files = list(self.profiler.output_dir.glob("gpu_profile_*.txt"))
        content = profile_files[-1].read_text()
        self.assertIn("timeout", content.lower())
        
    @patch('torch.cuda.is_available')
    def test_gpu_metric_corruption_scenarios(self, mock_is_available):
        """Test handling of corrupted GPU metrics."""
        mock_is_available.return_value = True
        
        # Test with invalid GPU metrics
        self.profiler.operation_metrics.extend([
            OperationMetrics(
                operation="gpu_invalid1",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=-1.0  # Invalid negative GPU usage
            ),
            OperationMetrics(
                operation="gpu_invalid2",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=float('inf')  # Invalid infinite GPU usage
            ),
            OperationMetrics(
                operation="gpu_invalid3",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=float('nan')  # Invalid NaN GPU usage
            )
        ])
        
        # Should handle corrupted GPU metrics gracefully
        self.profiler.analyze_correlations()
        
        # Verify correlation analysis handles invalid GPU metrics
        analysis_files = list(self.profiler.output_dir.glob("correlation_analysis_*.md"))
        content = analysis_files[0].read_text()
        self.assertIn("invalid GPU metrics", content.lower())
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_gpu_threshold_error_scenarios(self, mock_mem_allocated, mock_is_available):
        """Test handling of GPU threshold error scenarios."""
        mock_is_available.return_value = True
        mock_mem_allocated.return_value = 1024 * 1024 * 1000  # 1GB
        
        # Test with invalid GPU thresholds
        config = ThresholdConfig(
            memory_ratio=2.0,
            memory_release=0.9,
            memory_retention=30.0,
            gpu_utilization=-0.5  # Invalid negative utilization
        )
        self.profiler.threshold_manager = ThresholdManager(config)
        
        metrics = OperationMetrics(
            operation="gpu_threshold_test",
            duration=1.0,
            cpu_percent=50.0,
            memory_used=100.0,
            gpu_used=2000.0  # Exceeds typical GPU memory
        )
        
        # Should handle invalid thresholds gracefully
        suggestions = self.profiler.threshold_manager.get_optimization_suggestions(
            metrics,
            image_size=100.0
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertTrue(any("GPU" in s for s in suggestions))

class TestIntegration(unittest.TestCase):
    """Test suite for integration between profiling components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler()
        self.profiler.output_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_full_profiling_workflow(self):
        """Test complete profiling workflow integration."""
        # Create test data
        images, masks = self.profiler._create_test_data()
        
        # Run all profiling operations
        self.profiler.run_all_profiles()
        
        # Verify all expected outputs were generated
        expected_files = [
            "cpu_profile_*.txt",
            "memory_profile_detailed_*.txt",
            "memory_allocation_*.txt",
            "correlation_analysis_*.md",
            "memory_report_*.html",
            "profile_summary.md"
        ]
        
        for pattern in expected_files:
            files = list(self.profiler.output_dir.glob(pattern))
            self.assertTrue(
                len(files) > 0,
                f"Missing expected output: {pattern}"
            )
            
    def test_memory_correlation_integration(self):
        """Test integration between memory profiling and correlation analysis."""
        images, masks = self.profiler._create_test_data()
        
        # Run memory profiling
        self.profiler.profile_memory_detailed(
            {"test": images["simple"]}, 
            {"test": masks["simple"]}
        )
        
        # Run correlation analysis
        self.profiler.analyze_correlations()
        
        # Verify correlation analysis includes memory metrics
        analysis_files = list(self.profiler.output_dir.glob("correlation_analysis_*.md"))
        content = analysis_files[0].read_text()
        self.assertIn("Memory Usage", content)
        
    def test_threshold_visualization_integration(self):
        """Test integration between threshold management and visualization."""
        # Configure custom thresholds
        config = ThresholdConfig(
            memory_ratio=2.0,
            memory_release=0.9,
            memory_retention=30.0
        )
        self.profiler.threshold_manager = ThresholdManager(config)
        
        # Add test metrics
        self.profiler.memory_metrics.append(
            MemoryMetrics(
                operation="test",
                image_name="test",
                pre_memory=100.0,
                post_memory=300.0,  # Exceeds memory ratio threshold
                peak_memory=400.0,
                total_allocated=500.0,
                total_freed=200.0,  # Below memory release threshold
                timestamp=datetime.now()
            )
        )
        
        # Generate visualizations
        self.profiler.visualize_memory_usage()
        
        # Verify threshold violations are highlighted in report
        report_files = list(self.profiler.output_dir.glob("memory_report_*.html"))
        content = report_files[0].read_text()
        self.assertIn("threshold", content.lower())
        self.assertIn("warning", content.lower())
        
    @patch('torch.cuda.is_available')
    def test_gpu_memory_correlation_integration(self, mock_cuda_available):
        """Test integration between GPU profiling and correlation analysis."""
        mock_cuda_available.return_value = True
        
        # Add test metrics with GPU data
        self.profiler.operation_metrics.extend([
            OperationMetrics(
                operation="gpu_op1",
                duration=1.0,
                cpu_percent=50.0,
                memory_used=100.0,
                gpu_used=200.0
            ),
            OperationMetrics(
                operation="gpu_op2",
                duration=2.0,
                cpu_percent=75.0,
                memory_used=150.0,
                gpu_used=300.0
            )
        ])
        
        # Run correlation analysis
        self.profiler.analyze_correlations()
        
        # Verify GPU correlations are included
        analysis_files = list(self.profiler.output_dir.glob("correlation_analysis_*.md"))
        content = analysis_files[0].read_text()
        self.assertIn("GPU Usage", content)
        
    def test_profiling_summary_integration(self):
        """Test integration of all profiling data in summary generation."""
        images, masks = self.profiler._create_test_data()
        
        # Run individual profiling components
        self.profiler.profile_cpu({"test": images["simple"]}, {"test": masks["simple"]})
        self.profiler.profile_memory_detailed({"test": images["simple"]}, {"test": masks["simple"]})
        self.profiler.profile_memory_allocation({"test": images["simple"]}, {"test": masks["simple"]})
        self.profiler.analyze_correlations()
        
        # Generate summary
        self.profiler.generate_summary()
        
        # Verify summary includes all components
        summary_file = self.profiler.output_dir / "profile_summary.md"
        content = summary_file.read_text()
        
        expected_sections = [
            "CPU Performance",
            "Memory Usage",
            "Memory Allocation",
            "Correlation Analysis"
        ]
        
        for section in expected_sections:
            self.assertIn(section, content)

if __name__ == '__main__':
    unittest.main() 