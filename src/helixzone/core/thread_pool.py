"""Thread pool management module."""

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty, PriorityQueue
from typing import (Any, Callable, Dict, Generic, List, Optional, Tuple,
                    TypeVar)

import numpy as np
import psutil

R = TypeVar("R")


@dataclass
class TaskStats:
    """Task execution statistics."""

    execution_time: float
    queue_time: float
    cpu_usage: float
    memory_usage: float
    success: bool
    error: Optional[str] = None


class ScalingPolicy(Enum):
    """Thread pool scaling policies."""

    FIXED = auto()  # Fixed number of threads
    DYNAMIC = auto()  # Scale based on load
    ADAPTIVE = auto()  # Scale based on performance metrics


class Task(Generic[R]):
    """Task to be executed by the thread pool."""

    _counter = 0
    _counter_lock = threading.Lock()

    def __init__(
        self,
        func: Callable[..., R],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        callback: Optional[Callable[[R], None]] = None,
        priority: int = 0,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.priority = priority
        self.submit_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[R] = None
        self.error: Optional[Exception] = None
        self.stats: Optional[TaskStats] = None
        
        # Ensure unique ordering within same priority
        with Task._counter_lock:
            self.sequence = Task._counter
            Task._counter += 1
    
    def __lt__(self, other: 'Task[Any]') -> bool:
        """Compare tasks for priority queue ordering."""
        if not isinstance(other, Task):
            return NotImplemented
        # Higher priority numbers come first
        return (-self.priority, self.sequence) < (-other.priority, other.sequence)


class ThreadPool:
    """Dynamic thread pool with performance monitoring."""

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 8,
        scaling_policy: ScalingPolicy = ScalingPolicy.ADAPTIVE,
        queue_size: int = 100,
        monitor_interval: float = 1.0,
        scale_up_threshold: float = 0.75,  # CPU usage threshold for scaling up
        scale_down_threshold: float = 0.25,  # CPU usage threshold for scaling down
        scale_up_factor: float = 1.5,  # Multiply current workers by this when scaling up
        scale_down_factor: float = 0.75,  # Multiply current workers by this when scaling down
        stats_window: int = 100,  # Number of tasks to keep stats for
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_policy = scaling_policy
        self.monitor_interval = monitor_interval
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
        self.stats_window = stats_window

        # Task queue with priority
        self.task_queue: PriorityQueue[Task[Any]] = PriorityQueue(maxsize=queue_size)
        self.pending_tasks = threading.Event()

        # Thread management
        self.workers: List[threading.Thread] = []
        self.current_workers = 0  # Initialize to 0 before starting workers
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        # Performance monitoring
        self.task_stats: List[TaskStats] = []
        self.monitor_thread: Optional[threading.Thread] = None

        # Start initial workers and monitor
        self._start_workers(min_workers)
        self._start_monitor()

    def _worker(self) -> None:
        """Worker thread function."""
        while not self.stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop event
                task = self.task_queue.get(timeout=0.1)
                self.pending_tasks.set()

                if self.stop_event.is_set():
                    # Put the task back if we're stopping
                    self.task_queue.put(task)
                    break

                # Execute task and collect metrics
                task.start_time = time.time()
                try:
                    process = psutil.Process()
                    cpu_start = process.cpu_percent()
                    mem_start = process.memory_info().rss

                    task.result = task.func(*task.args, **task.kwargs)

                    cpu_end = process.cpu_percent()
                    mem_end = process.memory_info().rss
                    task.end_time = time.time()

                    # Calculate statistics
                    stats = TaskStats(
                        execution_time=task.end_time - task.start_time,
                        queue_time=task.start_time - task.submit_time,
                        cpu_usage=(cpu_end - cpu_start) / 100.0,
                        memory_usage=mem_end - mem_start,
                        success=True,
                    )

                    # Call callback if provided
                    if task.callback is not None and not self.stop_event.is_set():
                        task.callback(task.result)

                except Exception as e:
                    task.error = e
                    stats = TaskStats(
                        execution_time=time.time() - task.start_time,
                        queue_time=task.start_time - task.submit_time,
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        success=False,
                        error=str(e),
                    )

                task.stats = stats
                with self.lock:
                    self.task_stats.append(stats)
                    if len(self.task_stats) > self.stats_window:
                        self.task_stats.pop(0)

                self.task_queue.task_done()
                if self.task_queue.empty():
                    self.pending_tasks.clear()

            except Empty:
                continue

    def _monitor(self) -> None:
        """Monitor thread function."""
        last_scale_time = time.time()
        scale_cooldown = 0.5  # Minimum time between scaling operations

        while not self.stop_event.is_set():
            time.sleep(self.monitor_interval)

            if self.scaling_policy == ScalingPolicy.FIXED:
                continue

            current_time = time.time()
            if current_time - last_scale_time < scale_cooldown:
                continue

            with self.lock:
                queue_size = self.task_queue.qsize()
                
                # Scale down if queue is empty and we have excess workers
                if queue_size == 0 and self.current_workers > self.min_workers:
                    if not self.task_stats:
                        self._scale_down()
                        last_scale_time = current_time
                        continue
                    
                    # Calculate metrics
                    recent_stats = self.task_stats[-min(10, len(self.task_stats)):]
                    avg_cpu = np.mean([s.cpu_usage for s in recent_stats])
                    avg_queue_time = np.mean([s.queue_time for s in recent_stats])

                    # Determine if scaling is needed
                    if self.scaling_policy == ScalingPolicy.DYNAMIC:
                        if avg_cpu < self.scale_down_threshold:
                            self._scale_down()
                            last_scale_time = current_time
                    else:  # ADAPTIVE
                        if (avg_cpu < self.scale_down_threshold and 
                            avg_queue_time < 0.05):
                            self._scale_down()
                            last_scale_time = current_time
                    continue

                # Check for scaling up
                if not self.task_stats:
                    if queue_size > self.current_workers and self.current_workers < self.max_workers:
                        self._scale_up()
                        last_scale_time = current_time
                    continue

                # Calculate metrics for scaling up
                recent_stats = self.task_stats[-min(10, len(self.task_stats)):]
                avg_cpu = np.mean([s.cpu_usage for s in recent_stats])
                avg_queue_time = np.mean([s.queue_time for s in recent_stats])

                # Determine if scaling up is needed
                if self.scaling_policy == ScalingPolicy.DYNAMIC:
                    if (avg_cpu > self.scale_up_threshold or queue_size > self.current_workers) and self.current_workers < self.max_workers:
                        self._scale_up()
                        last_scale_time = current_time
                else:  # ADAPTIVE
                    if (avg_cpu > self.scale_up_threshold or 
                        avg_queue_time > 0.1 or 
                        queue_size > self.current_workers) and self.current_workers < self.max_workers:
                        self._scale_up()
                        last_scale_time = current_time

    def _scale_up(self) -> None:
        """Scale up the number of workers."""
        with self.lock:
            target_workers = min(
                self.max_workers,
                max(
                    self.current_workers + 1,
                    int(self.current_workers * self.scale_up_factor)
                )
            )
            if target_workers > self.current_workers:
                self._start_workers(target_workers - self.current_workers)

    def _scale_down(self) -> None:
        """Scale down the number of workers."""
        with self.lock:
            target_workers = max(
                self.min_workers,
                min(
                    self.current_workers - 1,
                    int(self.current_workers * self.scale_down_factor)
                )
            )
            if target_workers < self.current_workers:
                excess = self.current_workers - target_workers
                # Mark workers for removal
                for worker in self.workers[-excess:]:
                    worker.daemon = False  # Allow clean shutdown
                # Update counts first
                self.workers = self.workers[:-excess]
                self.current_workers = target_workers

    def _start_workers(self, count: int) -> None:
        """Start new worker threads."""
        with self.lock:
            for _ in range(count):
                if self.current_workers >= self.max_workers:
                    break
                worker = threading.Thread(target=self._worker)
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
                self.current_workers += 1

    def _start_monitor(self) -> None:
        """Start the monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def submit(
        self,
        func: Callable[..., R],
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] = {},
        callback: Optional[Callable[[R], None]] = None,
        priority: int = 0,
    ) -> Task[R]:
        """Submit a task to the thread pool."""
        if self.stop_event.is_set():
            raise RuntimeError("Cannot submit tasks to a stopped thread pool")
            
        # Wrap lambda functions to ensure proper serialization
        if isinstance(func, type(lambda: None)) and func.__name__ == '<lambda>':
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)
            task_func = wrapped_func
        else:
            task_func = func
            
        task = Task(task_func, args, kwargs or {}, callback, priority)
        self.task_queue.put(task)
        self.pending_tasks.set()
        return task

    def wait_all(self) -> None:
        """Wait for all tasks to complete."""
        try:
            while self.pending_tasks.is_set():
                self.task_queue.join()
                time.sleep(0.01)  # Small delay to prevent busy waiting
        except:  # Handle interruption
            pass

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self.stop_event.set()
        
        if wait:
            try:
                # Wait for pending tasks to complete
                self.wait_all()
            except:
                pass  # Handle interruption
            
            # Stop all workers
            for worker in self.workers:
                try:
                    worker.join(timeout=0.5)
                except:
                    pass  # Handle join failures
                    
            if self.monitor_thread:
                try:
                    self.monitor_thread.join(timeout=0.5)
                except:
                    pass
                    
        # Ensure cleanup
        self.workers.clear()
        self.current_workers = 0
        
        # Clear any remaining tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except Empty:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            stats = {
                "workers": {
                    "current": self.current_workers,
                    "min": self.min_workers,
                    "max": self.max_workers,
                },
                "queue": {
                    "size": self.task_queue.qsize(),
                    "capacity": self.task_queue.maxsize,
                },
                "tasks": {
                    "completed": len(self.task_stats),
                    "success_rate": (
                        np.mean([s.success for s in self.task_stats])
                        if self.task_stats
                        else 0.0
                    ),
                    "avg_execution_time": (
                        np.mean([s.execution_time for s in self.task_stats])
                        if self.task_stats
                        else 0.0
                    ),
                    "avg_queue_time": (
                        np.mean([s.queue_time for s in self.task_stats])
                        if self.task_stats
                        else 0.0
                    ),
                    "avg_cpu_usage": (
                        np.mean([s.cpu_usage for s in self.task_stats])
                        if self.task_stats
                        else 0.0
                    ),
                    "avg_memory_usage": (
                        np.mean([s.memory_usage for s in self.task_stats])
                        if self.task_stats
                        else 0.0
                    ),
                },
            }
            return stats
