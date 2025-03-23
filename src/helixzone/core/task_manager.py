"""Task management for background processing with UI feedback.

This module provides a system for running tasks in the background while
keeping the UI responsive and providing progress updates to the user.
"""

import time
import threading
import uuid
import logging
from typing import Dict, List, Callable, Any, Optional, Union, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from .memory_manager import memory_manager

# Configure logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
ProgressCallback = Callable[[float, str], None]
TaskFunction = Callable[..., T]


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Progress information for a task."""
    percent: float = 0.0
    message: str = ""
    current_step: int = 0
    total_steps: int = 0
    
    def __str__(self) -> str:
        """String representation of progress."""
        if self.total_steps > 0:
            return f"{self.percent:.1f}% ({self.current_step}/{self.total_steps}): {self.message}"
        return f"{self.percent:.1f}%: {self.message}"


@dataclass
class Task(Generic[T]):
    """Represents a background task."""
    id: str
    name: str
    function: TaskFunction
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Optional[T] = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    cancellation_event: threading.Event = field(default_factory=threading.Event)
    on_progress: Optional[ProgressCallback] = None
    on_complete: Optional[Callable[[Optional[T]], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if the task is complete (success, failure, or cancellation)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
    
    @property
    def can_cancel(self) -> bool:
        """Check if the task can be cancelled."""
        return not self.is_complete
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the task in seconds."""
        if self.start_time is not None:
            if self.end_time is not None:
                return self.end_time - self.start_time
            return time.time() - self.start_time
        return None


class TaskManager:
    """Manages background tasks with progress reporting and cancellation.
    
    Features:
    - Run tasks in background threads
    - Track task progress
    - Support for task cancellation
    - Progress callbacks for UI updates
    - Memory monitoring during task execution
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'TaskManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = TaskManager()
        return cls._instance
    
    def __init__(self, max_workers: int = 4):
        """Initialize task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}
        self._task_queue = queue.Queue()
        self._running_count = 0
        self._max_concurrent = max_workers
        
        # Start worker thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"Task manager initialized with {max_workers} workers")
    
    def _process_queue(self) -> None:
        """Process tasks from the queue."""
        while not self._stop_event.is_set():
            try:
                # Check if we can run more tasks
                with self._lock:
                    can_run = self._running_count < self._max_concurrent
                
                if can_run:
                    try:
                        # Get task from queue with timeout
                        task_id = self._task_queue.get(timeout=0.5)
                        self._start_task(task_id)
                        self._task_queue.task_done()
                    except queue.Empty:
                        # No tasks in queue
                        pass
                else:
                    # Wait a bit and check again
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in task queue processing: {e}")
                time.sleep(1.0)  # Prevent rapid error loops
    
    def _start_task(self, task_id: str) -> None:
        """Start a task from the queue.
        
        Args:
            task_id: ID of the task to start
        """
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found")
                return
                
            task = self._tasks[task_id]
            
            # Check if already started or cancelled
            if task.status != TaskStatus.PENDING:
                logger.warning(f"Task {task_id} not in pending state: {task.status}")
                return
                
            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            self._running_count += 1
            
            logger.info(f"Starting task {task.name} (ID: {task_id})")
            
            # Submit to executor
            future = self._executor.submit(
                self._execute_task,
                task_id,
                task.function,
                task.args,
                task.kwargs,
                task.on_progress
            )
            self._futures[task_id] = future
            
            # Add done callback
            future.add_done_callback(lambda f: self._task_completed(task_id, f))
    
    def _execute_task(
        self,
        task_id: str,
        fn: TaskFunction,
        args: tuple,
        kwargs: Dict[str, Any],
        progress_callback: Optional[ProgressCallback]
    ) -> Any:
        """Execute a task function.
        
        Args:
            task_id: ID of the task
            fn: Function to call
            args: Positional arguments
            kwargs: Keyword arguments
            progress_callback: Callback for progress updates
            
        Returns:
            Result of the function
        """
        # Get task
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found during execution")
                return None
                
            task = self._tasks[task_id]
            
            # Add cancellation support
            kwargs = kwargs.copy()
            kwargs['cancellation_event'] = task.cancellation_event
            
            # Add progress callback
            if 'progress_callback' not in kwargs and progress_callback is not None:
                kwargs['progress_callback'] = progress_callback
        
        try:
            # Monitor memory during execution
            with memory_manager.monitor_allocation(f"task_{task_id}"):
                # Call the function
                result = fn(*args, **kwargs)
                return result
        except Exception as e:
            # Capture traceback
            trace = traceback.format_exc()
            logger.error(f"Task {task_id} failed: {e}\n{trace}")
            
            # Update task
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.error = e
                    task.error_traceback = trace
            
            # Re-raise to be caught by completion handler
            raise
    
    def _task_completed(self, task_id: str, future: Future) -> None:
        """Handle task completion.
        
        Args:
            task_id: ID of the task
            future: Future object for the task
        """
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found during completion")
                return
                
            task = self._tasks[task_id]
            
            # Update task
            task.end_time = time.time()
            task.progress.percent = 100.0
            
            # Check if cancelled
            if task.cancellation_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.progress.message = "Task cancelled"
                logger.info(f"Task {task.name} (ID: {task_id}) cancelled after "
                           f"{task.duration:.2f} seconds")
            else:
                try:
                    # Get result
                    result = future.result()
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.progress.message = "Task completed successfully"
                    
                    # Call completion callback
                    if task.on_complete is not None:
                        try:
                            task.on_complete(result)
                        except Exception as callback_error:
                            logger.error(f"Error in completion callback for task {task_id}: "
                                        f"{callback_error}")
                    
                    logger.info(f"Task {task.name} (ID: {task_id}) completed in "
                               f"{task.duration:.2f} seconds")
                    
                except Exception as e:
                    # Task failed
                    task.status = TaskStatus.FAILED
                    task.error = e
                    task.error_traceback = traceback.format_exc()
                    task.progress.message = f"Task failed: {e}"
                    
                    # Call error callback
                    if task.on_error is not None:
                        try:
                            task.on_error(e)
                        except Exception as callback_error:
                            logger.error(f"Error in error callback for task {task_id}: "
                                        f"{callback_error}")
                    
                    logger.error(f"Task {task.name} (ID: {task_id}) failed after "
                                f"{task.duration:.2f} seconds: {e}")
            
            # Clean up
            self._running_count = max(0, self._running_count - 1)
            if task_id in self._futures:
                del self._futures[task_id]
    
    def create_task(
        self,
        name: str,
        function: TaskFunction,
        *args,
        **kwargs
    ) -> str:
        """Create a new task.
        
        Args:
            name: Name of the task
            function: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        # Extract callbacks from kwargs
        on_progress = kwargs.pop('on_progress', None)
        on_complete = kwargs.pop('on_complete', None)
        on_error = kwargs.pop('on_error', None)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task
        task = Task(
            id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error
        )
        
        # Store task
        with self._lock:
            self._tasks[task_id] = task
        
        logger.info(f"Created task {name} (ID: {task_id})")
        
        return task_id
    
    def submit_task(
        self,
        name: str,
        function: TaskFunction,
        *args,
        **kwargs
    ) -> str:
        """Create and submit a task for execution.
        
        Args:
            name: Name of the task
            function: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        # Create task
        task_id = self.create_task(name, function, *args, **kwargs)
        
        # Queue task for execution
        self._task_queue.put(task_id)
        
        logger.info(f"Submitted task {name} (ID: {task_id})")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks.
        
        Returns:
            List of tasks
        """
        with self._lock:
            return list(self._tasks.values())
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active (pending or running) tasks.
        
        Returns:
            List of active tasks
        """
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if cancelled, False if already complete or not found
        """
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for cancellation")
                return False
                
            task = self._tasks[task_id]
            
            # Check if can cancel
            if not task.can_cancel:
                logger.warning(f"Task {task_id} cannot be cancelled (already {task.status})")
                return False
                
            # Set cancellation event
            task.cancellation_event.set()
            
            # If pending, update status directly
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.end_time = time.time()
                
                # Remove from queue if possible
                try:
                    # Note: this is not reliable if the queue is being processed
                    with self._task_queue.mutex:
                        self._task_queue.queue.remove(task_id)
                except (ValueError, AttributeError):
                    pass
                
                logger.info(f"Cancelled pending task {task.name} (ID: {task_id})")
            else:
                logger.info(f"Cancelling running task {task.name} (ID: {task_id})")
            
            return True
    
    def cancel_all_tasks(self) -> int:
        """Cancel all active tasks.
        
        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        with self._lock:
            active_tasks = [
                task_id for task_id, task in self._tasks.items()
                if not task.is_complete
            ]
            
            for task_id in active_tasks:
                if self.cancel_task(task_id):
                    cancelled += 1
        
        logger.info(f"Cancelled {cancelled} tasks")
        
        return cancelled
    
    def update_progress(
        self,
        task_id: str,
        percent: float,
        message: str = "",
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> bool:
        """Update progress for a task.
        
        Args:
            task_id: ID of the task
            percent: Progress percentage (0-100)
            message: Progress message
            current_step: Current step number
            total_steps: Total number of steps
            
        Returns:
            True if updated, False if task not found or complete
        """
        with self._lock:
            if task_id not in self._tasks:
                return False
                
            task = self._tasks[task_id]
            
            # Check if can update
            if task.is_complete:
                return False
                
            # Update progress
            task.progress.percent = max(0.0, min(99.9, percent))
            task.progress.message = message
            
            if current_step is not None:
                task.progress.current_step = current_step
            
            if total_steps is not None:
                task.progress.total_steps = total_steps
            
            # Call progress callback
            if task.on_progress is not None:
                try:
                    task.on_progress(task.progress.percent, message)
                except Exception as e:
                    logger.error(f"Error in progress callback for task {task_id}: {e}")
            
            return True
    
    def remove_completed_tasks(self, max_age: float = 3600.0) -> int:
        """Remove completed tasks older than a specified age.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of tasks removed
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.is_complete and task.end_time is not None:
                    age = now - task.end_time
                    if age > max_age:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"Removed {removed} completed tasks")
        
        return removed
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down task manager")
        
        # Cancel all tasks if not waiting
        if not wait:
            self.cancel_all_tasks()
        
        # Stop worker thread
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        
        # Shutdown executor
        self._executor.shutdown(wait=wait)
        
        logger.info("Task manager shutdown complete")


# Create global task manager
task_manager = TaskManager.get_instance()


def get_task_manager() -> TaskManager:
    """Get the global task manager instance.
    
    Returns:
        The task manager instance
    """
    return task_manager


# Utility functions for simpler task submission
def run_in_background(
    name: str,
    function: TaskFunction,
    *args,
    on_progress: Optional[ProgressCallback] = None,
    on_complete: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    **kwargs
) -> str:
    """Run a function in the background.
    
    Args:
        name: Name of the task
        function: Function to run
        *args: Positional arguments for the function
        on_progress: Callback for progress updates
        on_complete: Callback for task completion
        on_error: Callback for task errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Task ID
    """
    return task_manager.submit_task(
        name,
        function,
        *args,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error,
        **kwargs
    )


def report_progress(
    task_id: str,
    percent: float,
    message: str = "",
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None
) -> bool:
    """Report progress for a task.
    
    Args:
        task_id: ID of the task
        percent: Progress percentage (0-100)
        message: Progress message
        current_step: Current step number
        total_steps: Total number of steps
        
    Returns:
        True if updated, False if task not found or complete
    """
    return task_manager.update_progress(
        task_id,
        percent,
        message,
        current_step,
        total_steps
    )


def is_cancelled(cancellation_event: threading.Event) -> bool:
    """Check if a task has been cancelled.
    
    Args:
        cancellation_event: Cancellation event from task kwargs
        
    Returns:
        True if cancelled
    """
    return cancellation_event.is_set()


def check_cancelled(cancellation_event: threading.Event) -> None:
    """Check if a task has been cancelled and raise CancelledError if so.
    
    Args:
        cancellation_event: Cancellation event from task kwargs
        
    Raises:
        concurrent.futures.CancelledError: If the task has been cancelled
    """
    if cancellation_event.is_set():
        from concurrent.futures import CancelledError
        raise CancelledError() 