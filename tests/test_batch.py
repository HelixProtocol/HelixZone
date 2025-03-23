"""Tests for the batch processing module."""

import os
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pytest
from PyQt6.QtGui import QImage
from pathlib import Path
from numpy.typing import NDArray
from helixzone.core.batch import (
    BatchProcessor,
    BatchContext,
    process_edges_batch,
    process_edges_async,
    wait_for_results
)
from helixzone.core.type_checker import ProcessingParams
from concurrent.futures import Future

@pytest.fixture
def test_image_array() -> NDArray[np.uint8]:
    """Create a test image array."""
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[25:75, 25:75] = 255  # White square in the middle
    return img_data

@pytest.fixture
def test_gray_array() -> NDArray[np.uint8]:
    """Create a test grayscale image array."""
    img_data = np.zeros((100, 100), dtype=np.uint8)
    img_data[25:75, 25:75] = 255  # White square in the middle
    return img_data

@pytest.fixture
def test_processing_params() -> ProcessingParams:
    """Create test processing parameters."""
    return ProcessingParams(
        sigma=1.0,
        threshold=0.5,
        kernel_size=3,
        iterations=1,
        normalize=True,
        threshold1=100.0,
        threshold2=200.0,
        aperture_size=3,
        l2_gradient=False
    )

@pytest.fixture
def test_batch_processor() -> BatchProcessor:
    """Create test batch processor."""
    return BatchProcessor(max_workers=2)

def test_batch_processor_init(test_batch_processor: BatchProcessor) -> None:
    """Test batch processor initialization."""
    assert test_batch_processor is not None
    assert test_batch_processor._executor is not None
    assert test_batch_processor._batch_size == 4
    assert test_batch_processor._futures == []

def test_process_batch(test_batch_processor: BatchProcessor, test_image_array: NDArray[np.uint8], 
                       test_processing_params: ProcessingParams) -> None:
    """Test batch processing."""
    # Create a list of test images
    images = [test_image_array] * 3
    params = vars(test_processing_params)
    
    results = test_batch_processor.process_batch(images, params)
    assert isinstance(results, list)
    assert len(results) == len(images)
    
    for result in results:
        assert isinstance(result, dict)
        assert "edge_map" in result
        assert "edge_gradient" in result
        assert "edge_strength" in result

def test_process_edges_async(test_batch_processor: BatchProcessor, test_image_array: NDArray[np.uint8], 
                           test_processing_params: ProcessingParams) -> None:
    """Test asynchronous edge processing."""
    params = vars(test_processing_params)
    future = test_batch_processor.process_edges_async(test_image_array, params)
    
    assert isinstance(future, Future)
    result = future.result()
    assert isinstance(result, dict)
    assert "edge_map" in result
    assert "edge_gradient" in result
    assert "edge_strength" in result

def test_batch_context() -> None:
    """Test batch context manager."""
    with BatchContext(max_workers=2) as processor:
        assert isinstance(processor, BatchProcessor)
        assert processor._executor is not None
        assert processor._batch_size == 4

def test_process_edges_batch(test_image_array: NDArray[np.uint8], test_processing_params: ProcessingParams) -> None:
    """Test batch processing of edges."""
    # Create a list of test images
    images = [test_image_array] * 3
    params = vars(test_processing_params)
    
    results = process_edges_batch(images, params, max_workers=2)
    assert isinstance(results, list)
    assert len(results) == len(images)
    
    for result in results:
        assert isinstance(result, dict)
        assert "edge_map" in result
        assert "edge_gradient" in result
        assert "edge_strength" in result

def test_wait_for_results(test_batch_processor: BatchProcessor, test_image_array: NDArray[np.uint8], 
                        test_processing_params: ProcessingParams) -> None:
    """Test waiting for results from futures."""
    params = vars(test_processing_params)
    futures = [
        test_batch_processor.process_edges_async(test_image_array, params),
        test_batch_processor.process_edges_async(test_image_array, params)
    ]
    
    results = wait_for_results(futures)
    assert isinstance(results, list)
    assert len(results) == len(futures)
    
    for result in results:
        assert isinstance(result, dict)
        assert "edge_map" in result
        assert "edge_gradient" in result
        assert "edge_strength" in result 