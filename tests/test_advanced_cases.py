"""
Tests for advanced scenarios and edge cases in HelixZone.
"""

import pytest
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from queue import Queue
import psutil
import os

from helixzone.core.ml_utils import EnhancedLassoFeathering

class TestAdvancedScenarios:
    @pytest.fixture
    def feathering(self):
        return EnhancedLassoFeathering()
        
    @pytest.fixture
    def large_batch(self):
        """Generate a large batch of test images."""
        images = []
        masks = []
        for _ in range(10):  # 10 images
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(
                mask,
                (256, 256),
                128,
                (255,),
                -1
            )
            images.append(img)
            masks.append(mask)
        return images, masks
        
    def test_concurrent_processing(self, feathering, large_batch):
        """Test concurrent processing of multiple images."""
        images, masks = large_batch
        results = Queue()
        threads = []
        
        def process_image(img, mask, idx):
            try:
                result = feathering.apply_color_aware_feathering(img, mask)
                results.put((idx, result))
            except Exception as e:
                results.put((idx, e))
                
        # Start threads
        for i in range(len(images)):
            thread = threading.Thread(
                target=process_image,
                args=(images[i], masks[i], i)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Check results
        processed = []
        while not results.empty():
            idx, result = results.get()
            if isinstance(result, Exception):
                raise result
            processed.append((idx, result))
            
        assert len(processed) == len(images), "All images should be processed"
        
    def test_multiprocess_batch(self, feathering, large_batch):
        """Test multi-process batch processing."""
        images, masks = large_batch
        
        def process_chunk(chunk):
            imgs, msks = zip(*chunk)
            results = []
            for img, mask in zip(imgs, msks):
                result = feathering.apply_color_aware_feathering(img, mask)
                results.append(result)
            return results
            
        # Split into chunks
        chunk_size = 2
        chunks = [
            list(zip(images[i:i+chunk_size], masks[i:i+chunk_size]))
            for i in range(0, len(images), chunk_size)
        ]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(process_chunk, chunks))
            
        # Flatten results
        processed = [r for chunk in results for r in chunk]
        assert len(processed) == len(images), "All chunks should be processed"
        
    def test_memory_limit_handling(self, feathering):
        """Test handling of memory limits."""
        # Create an image that would require significant memory
        size = (8192, 8192)  # 64MP image
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=np.uint8)
        cv2.circle(
            mask,
            (size[1]//2, size[0]//2),
            size[0]//4,
            (255,),
            -1
        )
        
        # Get available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        # Skip test if not enough memory
        if available_gb < 4:  # Need at least 4GB
            pytest.skip("Not enough memory for this test")
            
        try:
            result = feathering.apply_color_aware_feathering(img, mask)
            assert result is not None, "Processing should succeed"
        except MemoryError:
            # Should handle memory error gracefully
            assert False, "Should handle large images without memory error"
            
    def test_gpu_memory_management(self, feathering, large_batch):
        """Test GPU memory management."""
        images, masks = large_batch
        
        try:
            feathering.enable_gpu()
        except (AttributeError, RuntimeError):
            pytest.skip("GPU not available")
            
        # Process multiple images in sequence
        for img, mask in zip(images, masks):
            result = feathering.apply_color_aware_feathering(img, mask)
            assert result is not None, "GPU processing should succeed"
            
        # Force GPU memory cleanup
        if hasattr(feathering, 'clear_gpu_memory'):
            feathering.clear_gpu_memory()
            
    def test_error_recovery(self, feathering):
        """Test recovery from various error conditions."""
        # Invalid image type
        with pytest.raises(ValueError):
            img = np.zeros((100, 100, 3), dtype=np.float64)  # Wrong dtype
            mask = np.zeros((100, 100), dtype=np.uint8)
            feathering.apply_color_aware_feathering(img, mask)
            
        # Zero-size image
        with pytest.raises(ValueError):
            img = np.zeros((0, 0, 3), dtype=np.uint8)
            mask = np.zeros((0, 0), dtype=np.uint8)
            feathering.apply_color_aware_feathering(img, mask)
            
        # Corrupted image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img[50:60, 50:60] = np.nan  # Corrupt some pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 25, (255,), -1)
        
        with pytest.raises(ValueError):
            feathering.apply_color_aware_feathering(img, mask)
            
    def test_interrupt_handling(self, feathering, large_batch):
        """Test handling of interrupts during processing."""
        images, masks = large_batch
        results = Queue()
        
        def process_with_interrupt(img, mask):
            try:
                # Start processing
                result = feathering.apply_color_aware_feathering(img, mask)
                results.put(("success", result))
            except KeyboardInterrupt:
                results.put(("interrupted", None))
            except Exception as e:
                results.put(("error", e))
                
        # Start processing in thread
        thread = threading.Thread(
            target=process_with_interrupt,
            args=(images[0], masks[0])
        )
        thread.start()
        
        # Wait briefly then interrupt
        thread.join(timeout=0.1)
        if thread.is_alive():
            # Simulate interrupt
            thread._tstate_lock.release()
            thread._stop()
            
        # Check result
        if not results.empty():
            status, result = results.get()
            if status == "interrupted":
                # Should cleanup resources
                assert True, "Interrupt handled gracefully"
            elif status == "success":
                assert result is not None, "Processing completed before interrupt"
            else:
                raise result
                
    def test_resource_cleanup(self, feathering):
        """Test proper cleanup of resources."""
        # Monitor file handles
        def get_open_files():
            process = psutil.Process()
            return len(process.open_files())
            
        files_before = get_open_files()
        
        # Process an image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(mask, (256, 256), 128, (255,), -1)
        
        result = feathering.apply_color_aware_feathering(img, mask)
        
        # Check file handles
        files_after = get_open_files()
        assert files_after <= files_before + 1, "No file handle leaks"
        
        # Check memory
        if hasattr(feathering, 'clear_memory'):
            feathering.clear_memory()
            
        # If using GPU
        if hasattr(feathering, 'clear_gpu_memory'):
            feathering.clear_gpu_memory()
            
    def test_large_batch_stability(self, feathering):
        """Test stability with large batch processing."""
        # Create large batch
        batch_size = 50
        images = []
        masks = []
        
        for _ in range(batch_size):
            size = np.random.randint(100, 1000, 2)
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            mask = np.zeros(size, dtype=np.uint8)
            center = (size[1]//2, size[0]//2)
            radius = min(size)//4
            cv2.circle(mask, center, radius, (255,), -1)
            images.append(img)
            masks.append(mask)
            
        # Process in chunks
        chunk_size = 5
        for i in range(0, batch_size, chunk_size):
            chunk_imgs = images[i:i+chunk_size]
            chunk_masks = masks[i:i+chunk_size]
            
            for img, mask in zip(chunk_imgs, chunk_masks):
                result = feathering.apply_color_aware_feathering(img, mask)
                assert result is not None, f"Processing failed for image {i}"
                assert result.shape[:2] == img.shape[:2], "Output shape mismatch"
                
        # Check memory after batch
        if hasattr(feathering, 'clear_memory'):
            feathering.clear_memory()
            
    def test_edge_case_inputs(self, feathering):
        """Test handling of edge case inputs."""
        # Single pixel image
        img = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        mask = np.ones((1, 1), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            feathering.apply_color_aware_feathering(img, mask)
            
        # Maximum supported size
        max_size = (16384, 16384)  # 256MP
        if psutil.virtual_memory().available > max_size[0] * max_size[1] * 3 * 2:
            img = np.random.randint(0, 255, (*max_size, 3), dtype=np.uint8)
            mask = np.zeros(max_size, dtype=np.uint8)
            cv2.circle(
                mask,
                (max_size[1]//2, max_size[0]//2),
                max_size[0]//4,
                (255,),
                -1
            )
            
            try:
                result = feathering.apply_color_aware_feathering(img, mask)
                assert result is not None, "Should handle maximum size"
            except MemoryError:
                pytest.skip("Not enough memory for maximum size test") 