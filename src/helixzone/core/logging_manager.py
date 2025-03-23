"""Logging and monitoring system for HelixZone."""

import logging
import sys
import os
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, List, Generator
from datetime import datetime
import json
from pathlib import Path
import threading
from queue import Queue
import traceback
from dataclasses import dataclass, asdict
import time
from contextlib import contextmanager
import warnings
import importlib.util
import numpy as np
from logging.handlers import RotatingFileHandler

from .types import (
    NVMLDevice,
    NVMLUtilizationRates,
    GPUMetrics,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
)

# Check if pynvml is available
pynvml_spec = importlib.util.find_spec("pynvml")
pynvml = None
if pynvml_spec is not None:
    try:
        import pynvml
    except ImportError:
        warnings.warn("pynvml is installed but could not be imported. GPU monitoring will be limited.")

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging for the application.
    
    Args:
        log_level: The logging level to use
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Initialize the logging manager which sets up handlers
    manager = LoggingManager.get_instance()
    root_logger = logging.getLogger("helixzone")
    root_logger.setLevel(numeric_level)
    
    # Log the initialization
    root_logger.info(f"Logging system initialized")

@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str]
    memory_usage_mb: float
    gpu_utilization: Optional[float]
    additional_data: Dict[str, Any]
    error_count: int = 0
    warnings_count: int = 0

class AsyncLogHandler:
    """Handles logging asynchronously to prevent I/O bottlenecks."""
    
    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_queue: Queue[Dict[str, Any]] = Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_logs)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        
    def _process_logs(self) -> None:
        """Process logs from queue and write to files."""
        while not self._stop_event.is_set() or not self.log_queue.empty():
            try:
                if not self.log_queue.empty():
                    log_entry = self.log_queue.get_nowait()
                    self._write_log(log_entry)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing log: {e}", file=sys.stderr)
                
    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write a log entry to the appropriate file."""
        try:
            timestamp = datetime.fromtimestamp(log_entry["timestamp"])
            date_str = timestamp.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"helixzone_{date_str}.log"
            
            with log_file.open("a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            print(f"Error writing log: {e}", file=sys.stderr)
            
    def log(self, entry: Dict[str, Any]) -> None:
        """Add a log entry to the queue."""
        if not self._stop_event.is_set():
            self.log_queue.put(entry)
            
    def shutdown(self) -> None:
        """Shutdown the log handler and process remaining logs."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join()

class PerformanceLogger:
    """Logs and tracks performance metrics."""
    
    def __init__(self, log_dir: str = "logs") -> None:
        self.logger = logging.getLogger("helixzone.performance")
        self.async_handler = AsyncLogHandler(log_dir)
        self._metrics: Dict[str, list[OperationMetrics]] = {}
        
    def __del__(self) -> None:
        """Ensure proper cleanup of async handler."""
        if hasattr(self, 'async_handler'):
            self.async_handler.shutdown()
            
    @contextmanager
    def track_operation(
        self,
        operation_name: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Generator[None, None, None]:
        """Context manager for tracking operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        error_message = None
        success = True
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
            success = False
            self.logger.error(f"Operation {operation_name} failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration_ms = (end_time - start_time) * 1000
            
            metrics = OperationMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                memory_usage_mb=(end_memory - start_memory) / (1024 * 1024),
                gpu_utilization=self._get_gpu_utilization(),
                additional_data=additional_data or {}
            )
            
            self._log_metrics(metrics)
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0.0
            
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization if available."""
        try:
            if pynvml:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            else:
                return None
        except (ImportError, Exception):
            return None
            
    def _log_metrics(self, metrics: OperationMetrics) -> None:
        """Log operation metrics."""
        if metrics.operation_name not in self._metrics:
            self._metrics[metrics.operation_name] = []
        self._metrics[metrics.operation_name].append(metrics)
        
        log_entry = {
            "timestamp": time.time(),
            "level": "INFO" if metrics.success else "ERROR",
            "operation": metrics.operation_name,
            **asdict(metrics)
        }
        
        self.async_handler.log(log_entry)
        
    def get_operation_stats(
        self,
        operation_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get statistics for operations."""
        stats = {}
        
        operations = [operation_name] if operation_name else self._metrics.keys()
        
        for op in operations:
            if op not in self._metrics:
                continue
                
            metrics = self._metrics[op]
            durations = [m.duration_ms for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]
            gpu_utils = [m.gpu_utilization for m in metrics if m.gpu_utilization is not None]
            
            stats[op] = {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                "success_rate": len([m for m in metrics if m.success]) / len(metrics) * 100
            }
            
            if gpu_utils:
                stats[op]["avg_gpu_utilization"] = sum(gpu_utils) / len(gpu_utils)
                
        return stats
        
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to a JSON file."""
        data = {
            "metrics": {
                op: [asdict(m) for m in metrics]
                for op, metrics in self._metrics.items()
            },
            "stats": self.get_operation_stats(),
            "export_time": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
class ErrorLogger:
    """Handles error logging and aggregation."""
    
    def __init__(self, log_dir: str = "logs") -> None:
        self.logger = logging.getLogger("helixzone.errors")
        self.async_handler = AsyncLogHandler(log_dir)
        self._error_counts: Dict[str, int] = {}
        
    def __del__(self) -> None:
        """Ensure proper cleanup of async handler."""
        if hasattr(self, 'async_handler'):
            self.async_handler.shutdown()
            
    def log_error(
        self,
        error: Union[Exception, str],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error with context."""
        error_type = type(error).__name__ if isinstance(error, Exception) else "StringError"
        error_message = str(error)
        
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        log_entry = {
            "timestamp": time.time(),
            "level": "ERROR",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "stacktrace": traceback.format_exc() if isinstance(error, Exception) else None
        }
        
        self.async_handler.log(log_entry)
        self.logger.error(f"{error_type}: {error_message}")
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of logged errors."""
        return {
            "total_errors": sum(self._error_counts.values()),
            "error_types": dict(self._error_counts),
            "most_common": max(self._error_counts.items(), key=lambda x: x[1]) if self._error_counts else None
        }

class LoggingManager:
    """Manages application logging.
    
    Features:
    - Console and file logging
    - Error tracking
    - Performance monitoring
    - User action logging
    - Log rotation
    """
    
    _instance: Optional['LoggingManager'] = None
    
    @classmethod
    def get_instance(cls) -> 'LoggingManager':
        """Get singleton instance of LoggingManager."""
        if cls._instance is None:
            cls._instance = LoggingManager()
        return cls._instance
    
    def __init__(self):
        """Initialize logging manager."""
        self.logger = logging.getLogger('helixzone')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        self.formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Error tracking
        self.error_count: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}
        
        # Create log directory if it doesn't exist
        self.log_dir = self._get_log_directory()
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up handlers if not already configured
        if not self.logger.handlers:
            self._setup_console_handler()
            self._setup_file_handler()
            
        self.logger.info("Logging system initialized")
            
    def _get_log_directory(self) -> str:
        """Get the log directory path."""
        # Use appropriate location for logs based on platform
        if sys.platform == 'win32':
            base_dir = os.path.join(os.environ.get('APPDATA', ''), 'HelixZone')
        elif sys.platform == 'darwin':
            base_dir = os.path.expanduser('~/Library/Logs/HelixZone')
        else:  # Linux and others
            base_dir = os.path.expanduser('~/.config/helixzone')
            
        return os.path.join(base_dir, 'logs')
        
    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
    def _setup_file_handler(self) -> None:
        """Set up file logging handler with rotation."""
        log_file = os.path.join(self.log_dir, 'helixzone.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # Create separate error log
        error_log_file = os.path.join(self.log_dir, 'errors.log')
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message and track error frequency."""
        self.logger.error(message, exc_info=exc_info)
        
        # Track error occurrence
        error_type = message.split(':')[0] if ':' in message else message
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        self.last_error_time[error_type] = datetime.now()
        
    def critical(self, message: str, exc_info: bool = True) -> None:
        """Log critical error message."""
        self.logger.critical(message, exc_info=exc_info)
        
    def exception(self, message: str) -> None:
        """Log exception message with traceback."""
        self.logger.exception(message)
        
    def log_user_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """Log user action for auditing and analytics."""
        if details is None:
            details = {}
        self.logger.info(f"USER ACTION: {action} - {details}")
        
    def log_performance(self, operation: str, duration_ms: float) -> None:
        """Log performance metrics."""
        self.logger.info(f"PERFORMANCE: {operation} took {duration_ms:.2f}ms")
        
    def get_error_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of errors logged."""
        summary = {}
        for error_type, count in self.error_count.items():
            summary[error_type] = {
                'count': count,
                'last_occurrence': self.last_error_time.get(error_type)
            }
        return summary
        
    def clear_error_tracking(self) -> None:
        """Clear error tracking data."""
        self.error_count.clear()
        self.last_error_time.clear()

# Create a module-level function to get the logger
def get_logger(name: str = 'helixzone') -> logging.Logger:
    """Get a configured logger instance with the given name."""
    logger = logging.getLogger(name)
    
    # If this is the first call, ensure the root logger is configured
    if not logging.getLogger('helixzone').handlers:
        LoggingManager.get_instance()
        
    return logger

# Configure module-level logger
logger = get_logger()

# Helper functions for common logging patterns
def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None) -> None:
    """Log a function call with arguments."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    arg_str = ', '.join([str(a) for a in args])
    kwarg_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    param_str = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
    logger.debug(f"CALL: {func_name}({param_str})")
    
def log_exception_with_context(e: Exception, context: str) -> None:
    """Log an exception with contextual information."""
    traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    logger.error(f"EXCEPTION in {context}: {str(e)}\n{traceback_str}")
    
def log_operation_result(operation: str, success: bool, details: str = "") -> None:
    """Log the result of an operation."""
    result = "SUCCESS" if success else "FAILURE"
    if details:
        logger.info(f"OPERATION {operation}: {result} - {details}")
    else:
        logger.info(f"OPERATION {operation}: {result}") 