"""Tests for thread pool implementation."""
import time
import pytest
import numpy as np
from helixzone.core.thread_pool import ThreadPool, ScalingPolicy, Task, TaskStats
import threading

def test_thread_pool_init():
    """Test thread pool initialization."""
    pool = ThreadPool(min_workers=2, max_workers=4)
    assert pool.min_workers == 2
    assert pool.max_workers == 4
    assert pool.current_workers == 2
    assert len(pool.workers) == 2
    assert pool.scaling_policy == ScalingPolicy.ADAPTIVE
    pool.shutdown()

def test_task_execution():
    """Test basic task execution."""
    pool = ThreadPool(min_workers=1)
    
    def square(x: int) -> int:
        return x * x
    
    task = pool.submit(square, args=(5,))
    time.sleep(0.1)  # Give time for task to complete
    assert task.result == 25
    assert task.stats is not None
    assert task.stats.success
    pool.shutdown()

def test_callback_execution():
    """Test task callback execution."""
    pool = ThreadPool(min_workers=1)
    result = []
    
    def task_func(x: int) -> int:
        return x * 2
        
    def callback(res: int) -> None:
        result.append(res)
    
    task = pool.submit(task_func, args=(3,), callback=callback)
    time.sleep(0.1)  # Give time for callback to execute
    assert result == [6]
    pool.shutdown()

def test_error_handling():
    """Test error handling in tasks."""
    pool = ThreadPool(min_workers=1)
    
    def failing_task() -> None:
        raise ValueError("Test error")
    
    task = pool.submit(failing_task)
    time.sleep(0.1)  # Give time for task to complete
    assert isinstance(task.error, ValueError)
    assert task.stats is not None
    assert not task.stats.success
    assert task.stats.error is not None and "Test error" in task.stats.error
    pool.shutdown()

def test_scaling_up():
    """Test thread pool scaling up."""
    pool = ThreadPool(
        min_workers=1,
        max_workers=4,
        scaling_policy=ScalingPolicy.DYNAMIC,
        scale_up_threshold=0.5,
        monitor_interval=0.1
    )
    
    def cpu_task() -> None:
        # Simulate CPU-intensive task
        start = time.time()
        while time.time() - start < 0.2:
            _ = [i * i for i in range(10000)]  # Increased workload
    
    # Submit multiple tasks to trigger scaling
    tasks = [pool.submit(cpu_task) for _ in range(10)]
    
    # Wait for tasks to start and scaling to occur
    time.sleep(0.5)  # Wait for initial tasks
    for _ in range(5):  # Check multiple times
        if pool.current_workers > 1:
            break
        time.sleep(0.2)
    
    assert pool.current_workers > 1
    pool.shutdown()

def test_scaling_down():
    """Test thread pool scaling down."""
    pool = ThreadPool(
        min_workers=2,
        max_workers=4,
        scaling_policy=ScalingPolicy.DYNAMIC,
        scale_down_threshold=0.2,
        monitor_interval=0.1
    )
    
    # Force pool to scale up
    pool._start_workers(2)  # Add 2 more workers
    time.sleep(0.2)  # Give time for workers to start
    
    assert pool.current_workers == 4
    
    # Wait for scaling down with periodic checks
    for _ in range(10):  # Check multiple times
        if pool.current_workers == 2:
            break
        time.sleep(0.3)
    
    assert pool.current_workers == 2
    pool.shutdown()

def test_wait_all():
    """Test waiting for all tasks to complete."""
    pool = ThreadPool(min_workers=2)
    results = []
    
    def delayed_append(x: int) -> None:
        time.sleep(0.1)
        results.append(x)
    
    tasks = [pool.submit(delayed_append, args=(i,)) for i in range(5)]
    pool.wait_all()
    
    assert sorted(results) == [0, 1, 2, 3, 4]
    pool.shutdown()

def test_stats_collection():
    """Test task statistics collection."""
    pool = ThreadPool(min_workers=2, stats_window=5)
    
    def quick_task(x: int) -> int:
        return x + 1
    
    tasks = [pool.submit(quick_task, args=(i,)) for i in range(10)]
    time.sleep(0.1)  # Give time for tasks to complete
    
    stats = pool.get_stats()
    assert stats['workers']['current'] == 2
    assert stats['tasks']['completed'] == 5
    assert stats['tasks']['success_rate'] == 1.0
    pool.shutdown()

def test_adaptive_scaling():
    """Test adaptive scaling policy."""
    pool = ThreadPool(
        min_workers=1,
        max_workers=4,
        scaling_policy=ScalingPolicy.ADAPTIVE,
        scale_up_threshold=0.5,
        monitor_interval=0.1
    )
    
    def cpu_task() -> None:
        # Simulate CPU-intensive task with varying load
        start = time.time()
        while time.time() - start < 0.2:
            _ = [i * i for i in range(1000)]
    
    # Submit tasks in waves to test adaptive scaling
    for _ in range(3):
        tasks = [pool.submit(cpu_task) for _ in range(5)]
        time.sleep(0.5)
    
    stats = pool.get_stats()
    assert 1 < stats['workers']['current'] <= 4
    pool.shutdown()

def test_shutdown():
    """Test thread pool shutdown."""
    pool = ThreadPool(min_workers=2)
    
    def long_task() -> None:
        time.sleep(0.5)
    
    tasks = [pool.submit(long_task) for _ in range(5)]
    pool.shutdown(wait=True)
    
    assert pool.stop_event.is_set()
    assert all(not worker.is_alive() for worker in pool.workers)
    assert not pool.monitor_thread or not pool.monitor_thread.is_alive() 

def test_task_priority():
    """Test that tasks are executed according to their priority."""
    pool = ThreadPool(min_workers=1)
    results = []
    
    def task(x: int) -> None:
        time.sleep(0.1)  # Small delay to ensure ordering
        results.append(x)
    
    # Submit tasks with different priorities
    pool.submit(task, args=(3,), priority=0)  # Low priority
    pool.submit(task, args=(1,), priority=2)  # High priority
    pool.submit(task, args=(2,), priority=1)  # Medium priority
    
    pool.wait_all()
    assert results == [1, 2, 3]  # Should execute in priority order
    pool.shutdown()

def test_stress_scaling():
    """Test thread pool under heavy load with rapid task submission."""
    pool = ThreadPool(
        min_workers=2,
        max_workers=8,
        scaling_policy=ScalingPolicy.ADAPTIVE,
        monitor_interval=0.1
    )
    results = []
    
    def cpu_task(x: int) -> int:
        # Simulate varying CPU load
        start = time.time()
        while time.time() - start < 0.1:
            _ = [i * i for i in range(1000)]
        return x
    
    # Submit tasks in waves to test scaling under varying load
    for wave in range(3):
        tasks = [pool.submit(cpu_task, args=(i,)) for i in range(20)]
        time.sleep(0.3)  # Give time between waves
        
    pool.wait_all()
    stats = pool.get_stats()
    
    # Verify scaling behavior
    assert stats['workers']['current'] > 2  # Should have scaled up
    assert stats['tasks']['completed'] > 50  # Should have completed most tasks
    assert stats['tasks']['success_rate'] > 0.95  # High success rate
    pool.shutdown()

def test_error_recovery():
    """Test pool continues functioning after task errors."""
    pool = ThreadPool(min_workers=2)
    results = []
    
    def failing_task() -> None:
        raise ValueError("Simulated error")
        
    def good_task(x: int) -> None:
        results.append(x)
    
    # Mix of failing and successful tasks
    tasks = []
    for i in range(10):
        if i % 2 == 0:
            tasks.append(pool.submit(failing_task))
        else:
            tasks.append(pool.submit(good_task, args=(i,)))
    
    pool.wait_all()
    
    # Verify error handling and continued operation
    assert len([t for t in tasks if t.error is not None]) == 5  # Half should fail
    assert sorted(results) == [1, 3, 5, 7, 9]  # Odd numbers should succeed
    assert pool.current_workers == 2  # Worker count should remain stable
    pool.shutdown()

def test_memory_management():
    """Test memory usage patterns during task execution."""
    pool = ThreadPool(min_workers=2, stats_window=5)
    
    def memory_intensive_task() -> bytes:
        # Allocate and release memory
        data = b'x' * (1024 * 1024)  # 1MB
        time.sleep(0.1)
        return data
    
    # Execute several memory-intensive tasks
    tasks = [pool.submit(memory_intensive_task) for _ in range(10)]
    pool.wait_all()
    
    stats = pool.get_stats()
    # Verify memory stats are being tracked
    assert stats['tasks']['avg_memory_usage'] > 0
    assert len(pool.task_stats) <= 5  # Should respect stats_window
    pool.shutdown()

def test_graceful_shutdown():
    """Test graceful shutdown with pending tasks."""
    pool = ThreadPool(min_workers=2)
    results = []
    
    def slow_task(x: int) -> None:
        time.sleep(0.2)
        results.append(x)
    
    # Submit more tasks than workers
    for i in range(6):
        pool.submit(slow_task, args=(i,))
    
    # Immediate shutdown without wait
    pool.shutdown(wait=False)
    time.sleep(0.1)  # Give a moment for any in-progress tasks
    initial_results = len(results)
    assert initial_results < 6  # Not all tasks should complete
    
    # Test with wait=True
    pool = ThreadPool(min_workers=2)
    results.clear()
    
    for i in range(6):
        pool.submit(slow_task, args=(i,))
    
    pool.shutdown(wait=True)
    assert len(results) == 6  # All tasks should complete
    assert sorted(results) == list(range(6))  # All tasks completed in some order

def test_adaptive_scaling_under_load():
    """Test adaptive scaling behavior under varying load patterns."""
    pool = ThreadPool(
        min_workers=2,
        max_workers=6,
        scaling_policy=ScalingPolicy.ADAPTIVE,
        monitor_interval=0.1,
        scale_up_threshold=0.6,
        scale_down_threshold=0.2
    )
    
    def variable_load_task(work_time: float) -> None:
        start = time.time()
        while time.time() - start < work_time:
            _ = [i * i for i in range(1000)]
    
    # Phase 1: Light load
    for _ in range(4):
        pool.submit(variable_load_task, args=(0.1,))
    time.sleep(0.5)
    stats1 = pool.get_stats()
    
    # Phase 2: Heavy load
    for _ in range(12):
        pool.submit(variable_load_task, args=(0.2,))
    time.sleep(1.0)
    stats2 = pool.get_stats()
    
    # Phase 3: Back to light load
    time.sleep(1.0)  # Let heavy load tasks complete
    for _ in range(4):
        pool.submit(variable_load_task, args=(0.1,))
    time.sleep(0.5)
    stats3 = pool.get_stats()
    
    # Verify scaling behavior through phases
    assert stats1['workers']['current'] <= 4  # Should not scale up much under light load
    assert stats2['workers']['current'] > stats1['workers']['current']  # Should scale up under heavy load
    assert stats3['workers']['current'] < stats2['workers']['current']  # Should scale down after heavy load
    pool.shutdown()

def test_concurrent_task_submission():
    """Test thread pool behavior with concurrent task submission."""
    pool = ThreadPool(min_workers=2, max_workers=4)
    results = set()
    submit_lock = threading.Lock()
    
    def submit_tasks() -> None:
        for i in range(50):
            with submit_lock:
                pool.submit(lambda x: results.add(x), args=(i,))
            time.sleep(0.01)  # Small delay between submissions
    
    # Create multiple threads to submit tasks concurrently
    submitters = [
        threading.Thread(target=submit_tasks)
        for _ in range(3)
    ]
    
    # Start all submitter threads
    for submitter in submitters:
        submitter.start()
    
    # Wait for all submitter threads to complete
    for submitter in submitters:
        submitter.join()
    
    # Wait for all tasks to complete
    pool.wait_all()
    
    # Verify results
    assert len(results) == 150  # All tasks should complete
    assert min(results) == 0
    assert max(results) == 49
    pool.shutdown()

def test_task_cancellation():
    """Test task cancellation and cleanup."""
    pool = ThreadPool(min_workers=2)
    results = []
    
    def long_task(x: int) -> None:
        time.sleep(0.5)
        results.append(x)
    
    # Submit several long-running tasks
    tasks = [pool.submit(long_task, args=(i,)) for i in range(10)]
    
    # Shutdown immediately without waiting
    pool.shutdown(wait=False)
    
    # Verify that not all tasks completed
    time.sleep(0.1)  # Brief wait to allow some tasks to complete
    assert len(results) < 10
    
    # Verify thread cleanup
    assert all(not worker.is_alive() for worker in pool.workers)
    assert not pool.monitor_thread or not pool.monitor_thread.is_alive() 