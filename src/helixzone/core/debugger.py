"""Debugging system for HelixZone application."""

import logging
import sys
import os
import time
import traceback
import inspect
import psutil
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, cast
from typing_extensions import ParamSpec
from contextlib import contextmanager
import threading
from queue import Queue
import re

# Type variables for generic function types
T = TypeVar('T')  # For return type
P = ParamSpec('P')  # For function parameters

class ProjectDebugger:
    def __init__(
        self,
        project_name: str = "helixzone_debug",
        log_dir: str = "debug_logs",
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        self.project_name = project_name
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.start_time = datetime.now()
        self.checkpoints: List[Dict[str, Any]] = []
        self.error_count: Dict[str, int] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.debug_queue = Queue()
        self.is_running = True
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        # Start background processing
        self.processing_thread = threading.Thread(target=self._process_debug_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _setup_loggers(self):
        """Setup different loggers for different purposes"""
        # Main debug logger
        self.debug_logger = self._create_logger('debug', 'debug.log')
        
        # Error logger
        self.error_logger = self._create_logger('error', 'error.log')
        
        # Performance logger
        self.perf_logger = self._create_logger('performance', 'performance.log')
        
        # State logger
        self.state_logger = self._create_logger('state', 'state.log')

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a specific logger with both file and console handlers"""
        logger = logging.getLogger(f"{self.project_name}.{name}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, filename),
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
            )
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def start_debug_session(self):
        """Start a new debugging session with system information"""
        self.start_time = datetime.now()
        process = psutil.Process()
        
        system_info = {
            "session_start": self.start_time.isoformat(),
            "python_version": sys.version,
            "process_id": process.pid,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "environment_variables": dict(os.environ)
        }
        
        self.debug_logger.info(f"Debug session started: {json.dumps(system_info, indent=2)}")
        return system_info

    @contextmanager
    def error_boundary(self, section_name: str):
        """Context manager for error handling"""
        try:
            yield
        except Exception as e:
            error_info = {
                "section": section_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            self.error_logger.error(f"Error in {section_name}: {json.dumps(error_info, indent=2)}")
            self.error_count[section_name] = self.error_count.get(section_name, 0) + 1
            raise

    def trace_function(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator for function tracing with performance metrics.
        
        Args:
            func: Function to be decorated
            
        Returns:
            Wrapped function with tracing
            
        Type Parameters:
            P: ParamSpec for the function's parameters
            T: TypeVar for the function's return type
        """
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            func_name = func.__name__
            
            # Log function entry
            self.debug_logger.info(f"Entering {func_name}")
            self.variable_dump(
                prefix=f"{func_name}_args",
                args=args,
                kwargs=kwargs
            )
            
            try:
                with self.error_boundary(func_name):
                    result = func(*args, **kwargs)
                    
                    # Record performance
                    elapsed = time.time() - start_time
                    if func_name not in self.performance_metrics:
                        self.performance_metrics[func_name] = []
                    self.performance_metrics[func_name].append(elapsed)
                    
                    # Log success
                    self.perf_logger.info(
                        f"{func_name} completed in {elapsed:.4f} seconds"
                    )
                    return cast(T, result)
                    
            except Exception as e:
                self.error_logger.error(
                    f"Function {func_name} failed: {str(e)}"
                )
                raise
                
        return cast(Callable[P, T], wrapper)

    def variable_dump(self, prefix: str = "", **variables):
        """Enhanced variable inspection"""
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            location = "unknown:0"
        else:
            code = frame.f_back.f_code
            location = f"{code.co_filename}:{frame.f_back.f_lineno}"
            
        self.debug_queue.put({
            'type': 'variable_dump',
            'data': {
                'prefix': prefix,
                'variables': variables,
                'location': location
            }
        })

    def checkpoint(self, name: str, include_memory: bool = True):
        """Create a debugging checkpoint"""
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        checkpoint_data = {
            'name': name,
            'time': current_time.isoformat(),
            'elapsed': elapsed,
            'location': self._get_caller_info()
        }
        
        if include_memory:
            process = psutil.Process()
            checkpoint_data['memory_mb'] = process.memory_info().rss / 1024 / 1024
            checkpoint_data['cpu_percent'] = process.cpu_percent()
        
        self.checkpoints.append(checkpoint_data)
        self.state_logger.info(f"Checkpoint: {json.dumps(checkpoint_data, indent=2)}")

    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling function"""
        stack = inspect.stack()
        caller = stack[2]  # Get the caller of the caller
        return {
            'file': os.path.basename(caller.filename),
            'line': caller.lineno,
            'function': caller.function
        }

    def _process_debug_queue(self):
        """Background processing of debug information"""
        while self.is_running:
            try:
                item = self.debug_queue.get(timeout=1)
                if item['type'] == 'variable_dump':
                    self._process_variable_dump(item['data'])
                self.debug_queue.task_done()
            except:
                continue

    def _process_variable_dump(self, data: Dict[str, Any]):
        """Process a variable dump entry"""
        prefix = data['prefix']
        location = data['location']
        
        for name, value in data['variables'].items():
            self._log_variable(prefix, name, value, location)

    def _log_variable(self, prefix: str, name: str, value: Any, location: str):
        """Log a single variable with detailed information"""
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(value, (list, tuple)):
            self.debug_logger.debug(
                f"{full_name} at {location}: {type(value).__name__}[{len(value)}]"
            )
            for i, item in enumerate(value):
                self._log_variable(full_name, f"[{i}]", item, location)
        elif isinstance(value, dict):
            self.debug_logger.debug(
                f"{full_name} at {location}: dict{list(value.keys())}"
            )
            for k, v in value.items():
                self._log_variable(full_name, f"[{k}]", v, location)
        else:
            self.debug_logger.debug(
                f"{full_name} at {location}: {type(value).__name__} = {value}"
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report"""
        report = {}
        for func_name, times in self.performance_metrics.items():
            report[func_name] = {
                'calls': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times)
            }
        return report

    def get_error_report(self) -> Dict[str, Any]:
        """Generate an error report"""
        return {
            'total_errors': sum(self.error_count.values()),
            'error_counts': self.error_count,
            'error_locations': self._analyze_error_log()
        }

    def _analyze_error_log(self) -> Dict[str, int]:
        """Analyze error log for common error locations"""
        error_locations = {}
        error_log_path = os.path.join(self.log_dir, 'error.log')
        
        if os.path.exists(error_log_path):
            with open(error_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'ERROR' in line:
                        match = re.search(r'(\w+\.py:\d+)', line)
                        if match:
                            location = match.group(1)
                            error_locations[location] = error_locations.get(location, 0) + 1
        
        return error_locations

    def cleanup(self):
        """Cleanup and generate final reports"""
        self.is_running = False
        self.processing_thread.join()
        
        # Generate final reports
        final_report = {
            'session_duration': (datetime.now() - self.start_time).total_seconds(),
            'performance': self.get_performance_report(),
            'errors': self.get_error_report(),
            'checkpoints': self.checkpoints
        }
        
        # Save final report
        report_path = os.path.join(self.log_dir, 'debug_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, indent=2, default=str, fp=f)
        
        self.debug_logger.info(f"Debug session completed. Report saved to {report_path}") 