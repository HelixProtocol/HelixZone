"""Memory management utilities for HelixZone.

This module provides tools for tracking memory usage, detecting leaks,
and ensuring proper cleanup of resources.
"""

import os
import gc
import sys
import time
import weakref
import threading
import logging
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
import numpy as np
import torch
import psutil
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    system_total: int
    system_used: int
    system_available: int
    process_rss: int
    gpu_total: Optional[List[int]] = None
    gpu_used: Optional[List[int]] = None
    gpu_free: Optional[List[int]] = None
    allocated_tensors: Optional[int] = None
    reserved_memory: Optional[int] = None

class MemoryManager:
    """Manager for tracking and optimizing memory usage.
    
    Features:
    - System and GPU memory monitoring
    - Memory leak detection
    - Resource cleanup utilities
    - Memory-efficient processing for large images
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'MemoryManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = MemoryManager()
        return cls._instance
    
    def __init__(self):
        """Initialize memory manager."""
        self._snapshots: List[MemorySnapshot] = []
        self._snapshot_lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._leak_detection_enabled = False
        self._tracked_objects: Dict[int, Tuple[weakref.ref, str, int]] = {}
        self._allocation_sites: Dict[int, str] = {}
        
        # Configure GPU memory tracking if available
        self._has_cuda = torch.cuda.is_available()
        
        # Set threshold for warnings (percentages)
        self.system_memory_threshold = 90  # 90% usage triggers warning
        self.gpu_memory_threshold = 85     # 85% usage triggers warning
        
    def start_monitoring(self, interval: float = 5.0):
        """Start periodic memory monitoring.
        
        Args:
            interval: Time between measurements in seconds
        """
        if self._monitoring:
            return
            
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory_usage,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self._monitoring:
            return
            
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._monitoring = False
        logger.info("Memory monitoring stopped")
        
    def _monitor_memory_usage(self, interval: float):
        """Background thread for monitoring memory usage."""
        while not self._stop_monitoring.is_set():
            try:
                self.take_snapshot()
                self.check_memory_pressure()
                self.detect_potential_leaks()
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                
            self._stop_monitoring.wait(interval)
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        process = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            system_total=vm.total,
            system_used=vm.used,
            system_available=vm.available,
            process_rss=process.memory_info().rss
        )
        
        # Add GPU info if available
        if self._has_cuda:
            devices = torch.cuda.device_count()
            snapshot.gpu_total = []
            snapshot.gpu_used = []
            snapshot.gpu_free = []
            
            for i in range(devices):
                info = torch.cuda.get_device_properties(i)
                total = info.total_memory
                
                # Get current memory usage
                reserved = torch.cuda.memory_reserved(i)
                allocated = torch.cuda.memory_allocated(i)
                free = total - reserved
                
                snapshot.gpu_total.append(total)
                snapshot.gpu_used.append(allocated)
                snapshot.gpu_free.append(free)
                
            # Track PyTorch's global allocator stats
            snapshot.allocated_tensors = torch.cuda.memory_allocated()
            snapshot.reserved_memory = torch.cuda.memory_reserved()
        
        # Store the snapshot
        with self._snapshot_lock:
            self._snapshots.append(snapshot)
            # Keep only the last 100 snapshots
            if len(self._snapshots) > 100:
                self._snapshots.pop(0)
                
        return snapshot
    
    def check_memory_pressure(self) -> Tuple[bool, str]:
        """Check for memory pressure and return status.
        
        Returns:
            Tuple of (is_critical, message)
        """
        if not self._snapshots:
            return False, "No memory data available"
            
        snapshot = self._snapshots[-1]
        
        # Check system memory
        system_usage_percent = (snapshot.system_used / snapshot.system_total) * 100
        is_system_critical = system_usage_percent > self.system_memory_threshold
        
        # Check GPU memory if available
        is_gpu_critical = False
        gpu_message = ""
        
        if self._has_cuda and snapshot.gpu_total:
            for i, (total, used) in enumerate(zip(snapshot.gpu_total, snapshot.gpu_used)):
                usage_percent = (used / total) * 100
                if usage_percent > self.gpu_memory_threshold:
                    is_gpu_critical = True
                    gpu_message = f"GPU {i} memory usage critical: {usage_percent:.1f}%"
        
        is_critical = is_system_critical or is_gpu_critical
        
        message = ""
        if is_system_critical:
            message = f"System memory usage critical: {system_usage_percent:.1f}%"
        if gpu_message:
            message = f"{message} {gpu_message}" if message else gpu_message
            
        if is_critical:
            logger.warning(f"Memory pressure detected: {message}")
            
        return is_critical, message
    
    def enable_leak_detection(self):
        """Enable tracking of allocations for leak detection."""
        self._leak_detection_enabled = True
        logger.info("Memory leak detection enabled")
    
    def disable_leak_detection(self):
        """Disable tracking of allocations."""
        self._leak_detection_enabled = False
        self._tracked_objects.clear()
        self._allocation_sites.clear()
        logger.info("Memory leak detection disabled")
    
    def track_object(self, obj: Any, description: str = ""):
        """Track an object for potential leak detection.
        
        Args:
            obj: Object to track
            description: Description of the object
        """
        if not self._leak_detection_enabled:
            return
            
        obj_id = id(obj)
        import traceback
        stack = traceback.extract_stack()
        allocation_site = "".join(traceback.format_list(stack[-4:-1]))
        self._allocation_sites[obj_id] = allocation_site
        
        # Store weak reference to avoid creating a reference cycle
        self._tracked_objects[obj_id] = (weakref.ref(obj), description, time.time())
    
    def detect_potential_leaks(self) -> List[Tuple[str, str, float]]:
        """Check for potential memory leaks.
        
        Returns:
            List of (description, allocation_site, age_in_seconds) for potential leaks
        """
        if not self._leak_detection_enabled:
            return []
            
        # Run garbage collection to ensure dead objects are removed
        gc.collect()
        
        now = time.time()
        potential_leaks = []
        to_remove = []
        
        for obj_id, (weak_ref, description, timestamp) in self._tracked_objects.items():
            # Check if object still exists
            obj = weak_ref()
            if obj is None:
                to_remove.append(obj_id)
                continue
                
            # Check age - objects older than 60 seconds might be leaks
            age = now - timestamp
            if age > 60:
                allocation_site = self._allocation_sites.get(obj_id, "Unknown")
                potential_leaks.append((description, allocation_site, age))
                
                # Log potential leak
                logger.warning(f"Potential memory leak detected: {description}, age: {age:.1f}s")
        
        # Remove tracked objects that no longer exist
        for obj_id in to_remove:
            del self._tracked_objects[obj_id]
            if obj_id in self._allocation_sites:
                del self._allocation_sites[obj_id]
                
        return potential_leaks
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report.
        
        Returns:
            Dictionary with memory statistics
        """
        snapshot = self.take_snapshot()
        
        # Calculate system memory usage
        system_percent = (snapshot.system_used / snapshot.system_total) * 100
        
        report = {
            "timestamp": snapshot.timestamp,
            "system": {
                "total_mb": snapshot.system_total / (1024 * 1024),
                "used_mb": snapshot.system_used / (1024 * 1024),
                "available_mb": snapshot.system_available / (1024 * 1024),
                "usage_percent": system_percent
            },
            "process": {
                "rss_mb": snapshot.process_rss / (1024 * 1024)
            }
        }
        
        # Add GPU info if available
        if self._has_cuda and snapshot.gpu_total:
            report["gpu"] = []
            
            for i, (total, used, free) in enumerate(zip(
                snapshot.gpu_total, snapshot.gpu_used, snapshot.gpu_free
            )):
                gpu_report = {
                    "device": i,
                    "total_mb": total / (1024 * 1024),
                    "used_mb": used / (1024 * 1024),
                    "free_mb": free / (1024 * 1024),
                    "usage_percent": (used / total) * 100
                }
                report["gpu"].append(gpu_report)
                
            report["pytorch"] = {
                "allocated_mb": (snapshot.allocated_tensors or 0) / (1024 * 1024),
                "reserved_mb": (snapshot.reserved_memory or 0) / (1024 * 1024)
            }
        
        return report
    
    def cleanup_unused_memory(self) -> Tuple[int, int]:
        """Cleanup unused memory and return amount freed.
        
        Returns:
            Tuple of (system_bytes_freed, gpu_bytes_freed)
        """
        # Force garbage collection
        gc.collect()
        
        # Get memory before
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss
        gpu_before = 0
        
        if self._has_cuda:
            gpu_before = torch.cuda.memory_reserved()
            
            # Empty PyTorch CUDA caches
            torch.cuda.empty_cache()
        
        # Run GC again
        gc.collect()
        
        # Get memory after
        rss_after = process.memory_info().rss
        gpu_after = 0
        
        if self._has_cuda:
            gpu_after = torch.cuda.memory_reserved()
        
        # Calculate freed memory
        system_freed = max(0, rss_before - rss_after)
        gpu_freed = max(0, gpu_before - gpu_after)
        
        logger.info(f"Memory cleanup: freed {system_freed/(1024*1024):.1f} MB system, "
                   f"{gpu_freed/(1024*1024):.1f} MB GPU")
        
        return system_freed, gpu_freed
    
    @contextmanager
    def monitor_allocation(self, description: str):
        """Context manager to monitor memory allocation during a block.
        
        Args:
            description: Description of the operation being monitored
        """
        if not self._has_cuda:
            yield
            return
            
        # Record starting memory
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        start_time = time.time()
        
        try:
            # Execute the block
            yield
        finally:
            # Calculate memory changes
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            duration = time.time() - start_time
            
            allocated_diff = end_allocated - start_allocated
            reserved_diff = end_reserved - start_reserved
            
            logger.debug(
                f"Memory for {description}: "
                f"allocated {allocated_diff/(1024*1024):.2f} MB, "
                f"reserved {reserved_diff/(1024*1024):.2f} MB, "
                f"duration {duration:.2f}s"
            )

# Create a global instance
memory_manager = MemoryManager.get_instance()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance.
    
    Returns:
        The memory manager instance
    """
    return memory_manager 