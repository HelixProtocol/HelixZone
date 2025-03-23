"""
Detailed performance profiling for HelixZone operations.
"""

import cProfile
import pstats
import io
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TypeVar, Callable, Protocol, Union, TextIO, NamedTuple, cast
import numpy as np
import cv2
import psutil
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from collections import defaultdict
from scipy import stats
from scipy.stats._stats_py import LinregressResult
import json
import csv
import threading
from scipy.signal import find_peaks
from scipy.stats import norm, ks_2samp
from scipy.stats._stats_py import KstestResult
from enum import Enum, auto
import subprocess
import re

# Import statsmodels components with fallbacks
HAS_STATSMODELS = False
ARIMA = None  # type: ignore
adfuller = None  # type: ignore
seasonal_decompose = None  # type: ignore
try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    from statsmodels.tsa.stattools import adfuller  # type: ignore
    from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
    HAS_STATSMODELS = True
except ImportError as e:
    print(f"Warning: statsmodels not fully available: {e}")

# Import prophet with fallback
    HAS_PROPHET = False
Prophet = None  # type: ignore
try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except ImportError as e:
    print(f"Warning: prophet not available: {e}")

import pandas as pd

# Type variable for generic function type
F = TypeVar('F', bound=Callable[..., Any])

# Type hints for optional dependencies
class TorchProfiler(Protocol):
    def schedule(self, wait: int, warmup: int, active: int) -> Any: ...
    def tensorboard_trace_handler(self, dir_name: str) -> Callable[[str], None]: ...
    class ProfilerActivity:
        CPU: Any
        CUDA: Any

class TorchCuda(Protocol):
    def is_available(self) -> bool: ...
    def memory_allocated(self) -> int: ...
    def max_memory_allocated(self) -> int: ...

class TorchModule(Protocol):
    profiler: TorchProfiler
    cuda: TorchCuda

try:
    from memory_profiler import profile as memory_profile  # type: ignore
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    def memory_profile(func: F) -> F:
        """
        Fallback decorator when memory_profiler is not available.
        
        This decorator is used as a no-op replacement when the memory_profiler
        package is not installed. It simply returns the original function unchanged.
        
        Args:
            func: The function to be decorated
            
        Returns:
            The original function unmodified
            
        Note:
            To enable actual memory profiling, install memory_profiler:
            pip install memory_profiler>=0.61.0
        """
        return func

try:
    import torch  # type: ignore
    import torch.profiler  # type: ignore
    HAS_TORCH = True
    TORCH_MODULE: Optional[TorchModule] = torch  # type: ignore
except ImportError:
    HAS_TORCH = False
    TORCH_MODULE = None

from helixzone.core.ml_utils import EnhancedLassoFeathering

@dataclass
class MemoryMetrics:
    """Store memory metrics for visualization."""
    operation: str
    image_name: str
    pre_memory: float  # In MB
    post_memory: float  # In MB
    peak_memory: float  # In MB
    total_allocated: float  # In MB
    total_freed: float  # In MB
    timestamp: datetime

@dataclass
class OperationMetrics:
    """Store operation performance metrics."""
    operation: str
    duration: float  # In seconds
    cpu_percent: float
    memory_used: float  # In MB
    gpu_used: Optional[float] = None  # In MB

@dataclass
class ThresholdConfig:
    """Configuration for performance thresholds."""
    memory_ratio: float = 2.0
    memory_release: float = 0.8
    memory_retention: float = 50.0
    duration_outlier_std: float = 2.0
    gpu_utilization: float = 0.5
    cpu_threshold: float = 80.0
    # IO metrics thresholds
    io_read_latency: float = 100.0  # milliseconds
    io_write_latency: float = 200.0  # milliseconds
    io_bandwidth_threshold: float = 80.0  # percentage of max bandwidth
    # Network metrics thresholds
    network_latency: float = 50.0  # milliseconds
    network_bandwidth_usage: float = 70.0  # percentage of max bandwidth
    packet_loss_threshold: float = 1.0  # percentage
    
    def validate(self) -> List[str]:
        """Validate threshold configuration values."""
        issues = []
        if self.memory_ratio <= 0:
            issues.append("memory_ratio must be positive")
        if not 0 < self.memory_release <= 1:
            issues.append("memory_release must be between 0 and 1")
        if self.memory_retention < 0:
            issues.append("memory_retention must be non-negative")
        if self.duration_outlier_std <= 0:
            issues.append("duration_outlier_std must be positive")
        if not 0 < self.gpu_utilization <= 1:
            issues.append("gpu_utilization must be between 0 and 1")
        if not 0 <= self.cpu_threshold <= 100:
            issues.append("cpu_threshold must be between 0 and 100")
        return issues

@dataclass
class ThresholdViolation:
    """Record of a threshold violation."""
    metric_name: str
    timestamp: datetime
    value: float
    threshold: float
    severity: str  # 'warning' or 'critical'
    trend: str

@dataclass
class ViolationThresholds:
    """Configuration for violation thresholds per metric."""
    warning_multiplier: float = 1.2  # 20% above threshold
    critical_multiplier: float = 1.5  # 50% above threshold
    consecutive_violations: int = 3   # Number of consecutive violations before escalating severity
    violation_window: int = 60        # Time window in seconds for considering consecutive violations

@dataclass
class MetricViolationConfig:
    """Configuration for metric violation detection."""
    warning_threshold: float
    critical_threshold: float
    consecutive_violations: int = 3
    notification_level: str = 'all'  # 'all', 'warning', 'critical'
    adaptive_window: int = 100  # Window size for adaptive thresholds
    min_violations_for_alert: int = 3  # Minimum violations before alerting
    cooldown_period: int = 60  # Cooldown period between alerts in seconds
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.warning_threshold <= 0 or self.critical_threshold <= 0:
            raise ValueError("Thresholds must be positive")
        if self.consecutive_violations < 1:
            raise ValueError("consecutive_violations must be at least 1")
        if self.notification_level not in ['all', 'warning', 'critical']:
            raise ValueError("Invalid notification_level")
        if self.adaptive_window < 10:
            raise ValueError("adaptive_window must be at least 10")
        if self.min_violations_for_alert < 1:
            raise ValueError("min_violations_for_alert must be at least 1")
        if self.cooldown_period < 0:
            raise ValueError("cooldown_period must be non-negative")

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring."""
    enabled: bool = True
    update_interval: float = 1.0  # seconds
    alert_webhook_url: Optional[str] = None
    max_queue_size: int = 1000
    retention_period: int = 3600  # seconds to keep monitoring data

@dataclass
class MetricUpdate:
    """Real-time metric update."""
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)

class TrendAnalysis(NamedTuple):
    """Detailed trend analysis results."""
    trend: str
    confidence: float
    seasonality: Optional[float]
    outliers: List[Tuple[datetime, float]]
    change_points: List[Tuple[datetime, float]]
    forecast: List[Tuple[datetime, float]]

class ForecastMethod(Enum):
    """Available forecasting methods."""
    POLYNOMIAL = auto()
    ARIMA = auto()
    PROPHET = auto()

@dataclass
class ForecastResult:
    """Results from forecasting analysis."""
    forecast: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper) bounds
    model_metrics: Dict[str, float]  # Metrics like RMSE, MAE, AIC
    seasonality_info: Optional[Dict[str, Any]] = None
    method: ForecastMethod = ForecastMethod.POLYNOMIAL
    validation_score: float = 0.0  # Cross-validation score
    
class ModelSelector:
    """Automated model selection based on data characteristics."""
    
    @staticmethod
    def select_best_model(times: List[datetime], values: np.ndarray) -> Tuple[ForecastMethod, Dict[str, float]]:
        """Select the best forecasting model based on data characteristics."""
        metrics: Dict[str, float] = {}
        best_score = float('inf')
        best_method = ForecastMethod.POLYNOMIAL
        
        # Check for minimum data requirements
        if len(values) < 10:
            return ForecastMethod.POLYNOMIAL, {'insufficient_points': 0.0}
            
        # Detect seasonality
        try:
            from scipy.signal import periodogram
            freqs, spectrum = periodogram(values - np.mean(values))
            if max(spectrum[1:]) > 0.1:  # Significant seasonality
                metrics['seasonality_strength'] = float(max(spectrum[1:]))
                if len(values) >= 24:  # Enough data for Prophet
                    best_method = ForecastMethod.PROPHET
                else:
                    best_method = ForecastMethod.ARIMA
        except Exception:
            metrics['seasonality_strength'] = 0.0
            
        # Check stationarity for ARIMA
        try:
            if HAS_STATSMODELS and adfuller is not None:
                adf_result = adfuller(values)
                metrics['adf_pvalue'] = float(adf_result[1])
                if adf_result[1] < 0.05:  # Stationary series
                    best_method = ForecastMethod.ARIMA
        except Exception:
            metrics['adf_pvalue'] = 1.0
            
        # Evaluate each method using cross-validation
        for method in ForecastMethod:
            try:
                score = ModelSelector._evaluate_method(times, values, method)
                metrics[f'{method.name.lower()}_cv_score'] = score
                if score < best_score:
                    best_score = score
                    best_method = method
            except Exception:
                metrics[f'{method.name.lower()}_cv_score'] = float('inf')
                
        metrics['best_score'] = best_score
        return best_method, metrics
        
    @staticmethod
    def _evaluate_method(times: List[datetime], values: np.ndarray, 
                        method: ForecastMethod) -> float:
        """Evaluate a forecasting method using time series cross-validation."""
        if len(values) < 10:
            return float('inf')
            
        # Use time series cross-validation
        n_splits = min(5, len(values) // 10)
        scores = []
        
        for i in range(n_splits):
            train_size = len(values) - (n_splits - i) * 5
            if train_size < 10:  # Need at least 10 points for training
                continue
                
            train_times = times[:train_size]
            train_values = values[:train_size]
            test_values = values[train_size:train_size + 5]
            
            try:
                if method == ForecastMethod.POLYNOMIAL:
                    forecast = ModelSelector._polynomial_forecast(train_times, train_values, 5)
                elif method == ForecastMethod.ARIMA:
                    forecast = ModelSelector._arima_forecast(train_times, train_values, 5)
                elif method == ForecastMethod.PROPHET:
                    forecast = ModelSelector._prophet_forecast(train_times, train_values, 5)
                    
                if forecast:
                    pred_values = [f[1] for f in forecast]
                    rmse = np.sqrt(np.mean((test_values - pred_values) ** 2))
                    scores.append(rmse)
            except Exception:
                continue
                
        return float(np.mean(scores)) if scores else float('inf')
        
    @staticmethod
    def _polynomial_forecast(times: List[datetime], values: np.ndarray, 
                           horizon: int) -> List[Tuple[datetime, float]]:
        """Simple polynomial forecast for cross-validation."""
        numeric_times = np.array([(t - times[0]).total_seconds() for t in times])
        coeffs = np.polyfit(numeric_times, values, min(3, len(values) - 1))
        
        forecast_times = []
        last_time = numeric_times[-1]
        for i in range(horizon):
            forecast_times.append(last_time + (i + 1) * (last_time / len(values)))
            
        forecast_values = np.polyval(coeffs, forecast_times)
        return [(times[0] + timedelta(seconds=float(t)), float(v)) 
                for t, v in zip(forecast_times, forecast_values)]
                
    @staticmethod
    def _arima_forecast(times: List[datetime], values: np.ndarray, 
                       horizon: int) -> List[Tuple[datetime, float]]:
        """Simple ARIMA forecast for cross-validation."""
        if not HAS_STATSMODELS or ARIMA is None:
            return []
            
        try:
            model = ARIMA(values, order=(1, 1, 1))
            results = model.fit()
            forecast = results.forecast(steps=horizon)
            
            return [(times[-1] + timedelta(seconds=(i + 1) * 
                    (times[-1] - times[-2]).total_seconds()), float(f))
                    for i, f in enumerate(forecast)]
        except Exception:
            return []
            
    @staticmethod
    def _prophet_forecast(times: List[datetime], values: np.ndarray, 
                         horizon: int) -> List[Tuple[datetime, float]]:
        """Simple Prophet forecast for cross-validation."""
        if not HAS_PROPHET or Prophet is None:
            return []
            
        try:
            df = pd.DataFrame({'ds': times, 'y': values})
            model = Prophet(
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto'
            )
            model.fit(df)
            
            future_times = [times[-1] + timedelta(seconds=i * 
                          (times[-1] - times[-2]).total_seconds())
                          for i in range(1, horizon + 1)]
            future = pd.DataFrame({'ds': future_times})
            forecast = model.predict(future)
            
            return [(row.ds, float(row.yhat)) 
                    for _, row in forecast.iterrows()]
        except Exception:
            return []

class ThresholdManager:
    """Manages dynamic thresholds and violation detection for metrics."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None,
                 monitoring_config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or ThresholdConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.metric_violation_configs: Dict[str, MetricViolationConfig] = {}
        self.metric_queue: Queue[MetricUpdate] = Queue(
            maxsize=self.monitoring_config.max_queue_size
        )
        self.history: Dict[str, List[float]] = {}
        self.violation_history: Dict[str, List[Tuple[datetime, str, float]]] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.moving_stats: Dict[str, Dict[str, float]] = {}
        
        # Initialize IO and network metrics
        self.io_metrics: Dict[str, List[float]] = {
            'read_latency': [],
            'write_latency': [],
            'bandwidth_usage': []
        }
        self.network_metrics: Dict[str, List[float]] = {
            'latency': [],
            'bandwidth_usage': [],
            'packet_loss': []
        }
        
        # Initialize system counters
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_io_time = time.time()
    
    def configure_metric_thresholds(self, metric_name: str,
                                  config: MetricViolationConfig) -> None:
        """Configure thresholds for a specific metric."""
        self.metric_violation_configs[metric_name] = config
        self.history[metric_name] = []
        self.violation_history[metric_name] = []
        self.moving_stats[metric_name] = {
            'mean': 0.0,
            'std': 0.0,
            'last_update': 0.0
        }
    
    def add_metric_update(self, metric_name: str, value: float) -> None:
        """Add a new metric value and check for violations."""
        if not self.monitoring_config.enabled:
            return
            
        if metric_name not in self.metric_violation_configs:
            return
            
        # Update history
        self.history[metric_name].append(float(value))
        if len(self.history[metric_name]) > self.monitoring_config.retention_period:
            self.history[metric_name].pop(0)
        
        # Update moving statistics
        config = self.metric_violation_configs[metric_name]
        window = self.history[metric_name][-config.adaptive_window:]
        if len(window) >= 30:  # Minimum window for statistics
            self.moving_stats[metric_name]['mean'] = float(np.mean(window))
            self.moving_stats[metric_name]['std'] = float(np.std(window))
        
        # Check for violations
        self._check_violations(metric_name, float(value))
        
        # Add to queue for monitoring
        try:
            self.metric_queue.put_nowait(
                MetricUpdate(metric_name, float(value), datetime.now())
            )
        except Full:
            # Queue is full, remove oldest item
            try:
                self.metric_queue.get_nowait()
                self.metric_queue.put_nowait(
                    MetricUpdate(metric_name, float(value), datetime.now())
                )
            except Empty:
                pass
    
    def _check_violations(self, metric_name: str, value: float) -> None:
        """Check for threshold violations with improved detection."""
        config = self.metric_violation_configs[metric_name]
        stats = self.moving_stats[metric_name]
        now = datetime.now()
        
        # Get base threshold
        threshold = self.get_dynamic_threshold(metric_name)
        
        # Apply adaptive thresholds if enough history
        if stats['std'] > 0:
            z_score = abs(value - stats['mean']) / stats['std']
            if z_score > 3:  # Potential outlier
                # Check cooldown period
                last_alert = self.last_alert_time.get(metric_name)
                if last_alert and (now - last_alert).total_seconds() < config.cooldown_period:
                    return  # Skip if in cooldown period
                
                # Check violation counts
                recent_violations = [
                    v for v in self.violation_history[metric_name]
                    if (now - v[0]).total_seconds() <= config.cooldown_period
                ]
                
                if len(recent_violations) >= config.min_violations_for_alert:
                    severity = (
                        "critical" if value > threshold * config.critical_threshold
                        else "warning" if value > threshold * config.warning_threshold
                        else None
                    )
                    
                    if severity:
                        self.violation_history[metric_name].append((now, severity, value))
                        self.last_alert_time[metric_name] = now
                        
                        # Cleanup old violations
                        cutoff = now - timedelta(seconds=config.cooldown_period)
                        self.violation_history[metric_name] = [
                            v for v in self.violation_history[metric_name]
                            if v[0] > cutoff
                        ]
    
    def get_dynamic_threshold(self, metric_name: str) -> float:
        """Get dynamic threshold with adaptive components."""
        if metric_name not in self.metric_violation_configs:
            return 0.0
            
        stats = self.moving_stats[metric_name]
        if stats['std'] > 0:
            # Use mean + 2*std as base threshold
            base_threshold = stats['mean'] + 2 * stats['std']
        else:
            # Fallback to config threshold
            config = self.metric_violation_configs[metric_name]
            base_threshold = config.warning_threshold
        
        return float(base_threshold)
    
    def analyze_violation_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze violation trends with improved metrics."""
        now = datetime.now()
        trends: Dict[str, Dict[str, Any]] = {}
        
        for metric_name in self.metric_violation_configs:
            violations = self.violation_history.get(metric_name, [])
            if not violations:
                continue

            # Calculate hourly frequency
            hour_ago = now - timedelta(hours=1)
            recent_violations = [v for v in violations if v[0] > hour_ago]
            frequency = len(recent_violations) / 1.0  # per hour
            
            # Count by severity
            severity_counts = {
                'warning': len([v for v in recent_violations if v[1] == 'warning']),
                'critical': len([v for v in recent_violations if v[1] == 'critical'])
            }
            
            # Determine trend
            trend = self._calculate_trend(violations, now)
            
            trends[metric_name] = {
                'frequency': frequency,
                'severity_counts': severity_counts,
                'trend': trend
            }
        
        return trends
        
    def _calculate_trend(self, violations: List[Tuple[datetime, str, float]], now: datetime) -> str:
        """Calculate trend from violations data."""
        if len(violations) < 2:
            return "insufficient_data"
            
        first_half = violations[:len(violations)//2]
        second_half = violations[len(violations)//2:]
        
        first_rate = len(first_half) / max(1, (second_half[0][0] - first_half[0][0]).total_seconds() / 3600)
        second_rate = len(second_half) / max(1, (now - second_half[0][0]).total_seconds() / 3600)
        
        if second_rate > first_rate * 1.2:
            return "increasing"
        elif second_rate < first_rate * 0.8:
            return "decreasing"
        return "stable"

    def update_history(self, metric_name: str, value: float) -> None:
        """Update metric history."""
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)
        if len(self.history[metric_name]) > self.monitoring_config.retention_period:
            self.history[metric_name].pop(0)
            
    def get_optimization_suggestions(self, metrics: OperationMetrics, image_size: float) -> List[str]:
        """Get optimization suggestions based on metrics."""
        suggestions = []
        
        # Memory optimization suggestions
        memory_ratio = metrics.memory_used / image_size
        if memory_ratio > self.config.memory_ratio:
            suggestions.append(
                f"High memory usage detected ({memory_ratio:.1f}x input size). "
                "Consider implementing batch processing or reducing intermediate allocations."
            )
            
        # CPU optimization suggestions
        if metrics.cpu_percent > self.config.cpu_threshold:
            suggestions.append(
                f"High CPU usage detected ({metrics.cpu_percent:.1f}%). "
                "Consider parallelization or algorithmic optimizations."
            )
            
        # GPU optimization suggestions if available
        if metrics.gpu_used is not None:
            gpu_ratio = metrics.gpu_used / image_size
            if gpu_ratio < self.config.gpu_utilization:
                suggestions.append(
                    f"Low GPU utilization detected ({gpu_ratio:.1f}x input size). "
                    "Consider increasing batch size or moving more operations to GPU."
                )
                
        return suggestions

    def visualize_thresholds(self, output_dir: Path) -> None:
        """Visualize threshold evolution over time."""
        if not self.history:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Create threshold evolution plot
        plt.figure(figsize=(15, 8))
        for metric_name, values in self.history.items():
            plt.plot(range(len(values)), values, label=metric_name)
            
        plt.title("Threshold Evolution Over Time")
        plt.xlabel("Updates")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(vis_dir / f"threshold_evolution_{timestamp}.png")
        plt.close()
        
    def export_violations(self, format_type: str) -> None:
        """Export violation data in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            data = {
                metric: [(t.isoformat(), sev, val) for t, sev, val in violations]
                for metric, violations in self.violation_history.items()
            }
            with open(f"violations_{timestamp}.json", 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == 'csv':
            with open(f"violations_{timestamp}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Timestamp', 'Severity', 'Value'])
                for metric, violations in self.violation_history.items():
                    for t, sev, val in violations:
                        writer.writerow([metric, t.isoformat(), sev, val])
                        
    def export_threshold_history(self, format_type: str) -> None:
        """Export threshold history in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            with open(f"threshold_history_{timestamp}.json", 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
                
    def export_metrics_summary(self) -> None:
        """Export metrics summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trends = self.analyze_violation_trends()
        
        with open(f"metrics_summary_{timestamp}.json", 'w') as f:
            summary = {
                'trends': trends,
                'thresholds': {
                    metric: {
                        'warning': config.warning_threshold,
                        'critical': config.critical_threshold,
                        'violations_count': len(self.violation_history.get(metric, []))
                    }
                    for metric, config in self.metric_violation_configs.items()
                }
            }
            json.dump(summary, f, indent=2, default=str)

    def set_forecast_method(self, method: ForecastMethod) -> None:
        """Set the forecasting method to use."""
        self._forecast_method = method

    def _generate_forecast(self, times: List[datetime], values: np.ndarray) -> List[Tuple[datetime, float]]:
        """Generate forecast using the selected method."""
        if not hasattr(self, '_forecast_method'):
            self._forecast_method = ForecastMethod.POLYNOMIAL
            
        # Convert numpy array to list for forecasting
        values_list = values.tolist()
        times_list = list(times)  # Convert to list if needed
        
        if self._forecast_method == ForecastMethod.POLYNOMIAL:
            return ModelSelector._polynomial_forecast(times_list, np.array(values_list), 5)
        elif self._forecast_method == ForecastMethod.ARIMA:
            return ModelSelector._arima_forecast(times_list, np.array(values_list), 5)
        elif self._forecast_method == ForecastMethod.PROPHET:
            return ModelSelector._prophet_forecast(times_list, np.array(values_list), 5)
        return []

    def analyze_trend_detailed(self, metric_name: str) -> Optional[TrendAnalysis]:
        """Analyze trend with detailed metrics."""
        if metric_name not in self.history:
            return None
            
        values = np.array(self.history[metric_name])
        if len(values) < 10:
            return None
            
        times = [datetime.now() - timedelta(seconds=i*self.monitoring_config.update_interval)
                for i in range(len(values)-1, -1, -1)]
        
        # Detect trend
        try:
            x = np.array(range(len(values)))
            y = np.array(values)
            # Get regression results and handle type conversion
            try:
                # Unpack the regression results and convert to numpy arrays first
                slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
                # Convert to numpy arrays first to ensure proper typing
                r_value_arr = np.array(r_value, dtype=np.float64)
                p_value_arr = np.array(p_value, dtype=np.float64)
                slope_arr = np.array(slope, dtype=np.float64)
                
                # Then convert to Python floats
                confidence = float(abs(r_value_arr))
                p_value = float(p_value_arr)
                slope = float(slope_arr)
            except (TypeError, ValueError, AttributeError):
                confidence = 0.0
                p_value = 1.0
                slope = 0.0
            
            if p_value < 0.05:
                if slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "stable"
        except Exception:
            trend = "unknown"
            confidence = 0.0
            
        # Detect seasonality
        try:
            from scipy.signal import periodogram
            freqs, spectrum = periodogram(values - np.mean(values))
            if max(spectrum[1:]) > 0.1:
                seasonality = 1.0 / freqs[np.argmax(spectrum[1:]) + 1]
            else:
                seasonality = None
        except Exception:
            seasonality = None
            
        # Detect outliers
        outliers = []
        mean = np.mean(values)
        std = np.std(values)
        for i, value in enumerate(values):
            if abs(value - mean) > 3 * std:
                outliers.append((times[i], value))
                
        # Detect change points
        change_points = []
        window = min(10, len(values) // 4)
        if window > 0:
            for i in range(window, len(values) - window):
                before = values[i-window:i]
                after = values[i:i+window]
                if abs(np.mean(after) - np.mean(before)) > 2 * std:
                    change_points.append((times[i], values[i]))
                    
        # Generate forecast
        forecast = self._generate_forecast(times, values)
        
        return TrendAnalysis(
            trend=trend,
            confidence=confidence,
            seasonality=seasonality,
            outliers=outliers,
            change_points=change_points,
            forecast=forecast
        )

class PerformanceProfiler:
    def __init__(self) -> None:
        self.feathering = EnhancedLassoFeathering()
        self.output_dir = Path("benchmarks/profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_metrics: List[MemoryMetrics] = []
        self.operation_metrics: List[OperationMetrics] = []
        self.threshold_manager = ThresholdManager()
        self.io_metrics: Dict[str, List[float]] = {
            'read_latency': [],
            'write_latency': [],
            'bandwidth_usage': []
        }
        self.network_metrics: Dict[str, List[float]] = {
            'latency': [],
            'bandwidth_usage': [],
            'packet_loss': []
        }
        # Initialize psutil for system metrics
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_io_time = time.time()
        
        # Set up visualization style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _create_test_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create test images and masks of various sizes and complexities."""
        images = {}
        masks = {}
        
        # Simple circular mask
        size = (1024, 1024)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=np.uint8)
        cv2.circle(mask, (size[1]//2, size[0]//2), size[0]//4, (255,), -1)
        images["simple"] = img
        masks["simple"] = mask
        
        # Complex pattern
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=np.uint8)
        for i in range(10):
            center = (
                np.random.randint(0, size[1]),
                np.random.randint(0, size[0])
            )
            radius = np.random.randint(20, 100)
            cv2.circle(mask, center, radius, (255,), -1)
        images["complex"] = img
        masks["complex"] = mask
        
        # High resolution
        size = (4096, 4096)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=np.uint8)
        cv2.circle(mask, (size[1]//2, size[0]//2), size[0]//4, (255,), -1)
        images["high_res"] = img
        masks["high_res"] = mask
        
        return images, masks
        
    def profile_cpu(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> None:
        """Profile CPU usage with cProfile."""
        profiler = cProfile.Profile()
        
        for name in images:
            print(f"\nProfiling {name} image processing...")
            profiler.enable()
            _ = self.feathering.apply_lasso_feathering(
                images[name],
                masks[name],
                content_aware=True
            )
            profiler.disable()
            
            # Save stats
            output = io.StringIO()
            stats = pstats.Stats(profiler, stream=output).sort_stats('cumulative')
            stats.print_stats(30)  # Top 30 functions
            
            profile_path = self.output_dir / f"cpu_profile_{name}.txt"
            profile_path.write_text(output.getvalue())
            print(f"CPU profile saved to {profile_path}")
            
    def visualize_memory_usage(self) -> None:
        """Create visualizations of memory usage patterns.
        
        Generates several plots:
        1. Memory usage over time for each operation
        2. Peak memory comparison across operations
        3. Memory allocation/deallocation patterns
        4. Memory efficiency (freed/allocated ratio)
        """
        if not self.memory_metrics:
            print("No memory metrics available for visualization")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Memory Timeline
        plt.figure(figsize=(15, 8))
        for image_name in {m.image_name for m in self.memory_metrics}:
            image_metrics = [m for m in self.memory_metrics if m.image_name == image_name]
            times = range(len(image_metrics))
            plt.plot(times, [m.post_memory for m in image_metrics], 
                    label=f"{image_name}", marker='o')
            
        plt.title("Memory Usage Over Time")
        plt.xlabel("Operation Sequence")
        plt.ylabel("Memory Usage (MB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(vis_dir / f"memory_timeline_{timestamp}.png")
        plt.close()
        
        # 2. Peak Memory by Operation
        plt.figure(figsize=(12, 6))
        operation_peaks = defaultdict(list)
        for m in self.memory_metrics:
            operation_peaks[m.operation].append(m.peak_memory)
            
        operations = list(operation_peaks.keys())
        peak_means = [np.mean(peaks) for peaks in operation_peaks.values()]
        peak_stds = [np.std(peaks) for peaks in operation_peaks.values()]
        
        plt.bar(operations, peak_means, yerr=peak_stds, capsize=5)
        plt.title("Peak Memory Usage by Operation")
        plt.xlabel("Operation")
        plt.ylabel("Peak Memory (MB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(vis_dir / f"peak_memory_{timestamp}.png")
        plt.close()
        
        # 3. Memory Allocation Patterns
        plt.figure(figsize=(12, 6))
        for op in operations:
            op_metrics = [m for m in self.memory_metrics if m.operation == op]
            allocated = [m.total_allocated for m in op_metrics]
            freed = [m.total_freed for m in op_metrics]
            plt.scatter(allocated, freed, label=op, alpha=0.7)
            
        plt.plot([0, max(m.total_allocated for m in self.memory_metrics)],
                 [0, max(m.total_allocated for m in self.memory_metrics)],
                 'k--', alpha=0.3, label='Perfect Memory Release')
        plt.title("Memory Allocation vs Deallocation")
        plt.xlabel("Allocated Memory (MB)")
        plt.ylabel("Freed Memory (MB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(vis_dir / f"allocation_pattern_{timestamp}.png")
        plt.close()
        
        # 4. Memory Efficiency
        plt.figure(figsize=(12, 6))
        efficiency_data = []
        for m in self.memory_metrics:
            if m.total_allocated > 0:
                efficiency = m.total_freed / m.total_allocated
                efficiency_data.append({
                    'operation': m.operation,
                    'image': m.image_name,
                    'efficiency': efficiency
                })
                
        if efficiency_data:
            efficiency_df = defaultdict(list)
            for d in efficiency_data:
                efficiency_df[d['operation']].append(d['efficiency'])
                
            operations = list(efficiency_df.keys())
            plt.boxplot([efficiency_df[op] for op in operations])
            plt.xticks(range(1, len(operations) + 1), operations)
            plt.title("Memory Release Efficiency by Operation")
            plt.ylabel("Freed/Allocated Ratio")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(vis_dir / f"memory_efficiency_{timestamp}.png")
            plt.close()
            
        # Generate HTML report
        html_report = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "img { max-width: 100%; margin: 20px 0; }",
            "h2 { color: #2c3e50; }",
            ".metric { margin: 10px 0; padding: 10px; background: #f7f9fc; }",
            "</style>",
            "</head><body>",
            "<h1>Memory Profile Visualization Report</h1>",
            f"<p>Generated on: {timestamp}</p>",
            
            "<h2>Memory Usage Timeline</h2>",
            f'<img src="memory_timeline_{timestamp}.png" alt="Memory Timeline">',
            "<p>Shows how memory usage changes over time for each image processing sequence.</p>",
            
            "<h2>Peak Memory Usage</h2>",
            f'<img src="peak_memory_{timestamp}.png" alt="Peak Memory">',
            "<p>Compares peak memory usage across different operations.</p>",
            
            "<h2>Memory Allocation Patterns</h2>",
            f'<img src="allocation_pattern_{timestamp}.png" alt="Allocation Pattern">',
            "<p>Visualizes the relationship between allocated and freed memory.</p>",
            
            "<h2>Memory Efficiency</h2>",
            f'<img src="memory_efficiency_{timestamp}.png" alt="Memory Efficiency">',
            "<p>Shows how efficiently memory is released after operations.</p>",
            
            "<h2>Key Findings</h2>"
        ]
        
        # Add key metrics
        total_metrics = len(self.memory_metrics)
        avg_peak = np.mean([m.peak_memory for m in self.memory_metrics])
        max_peak = max(m.peak_memory for m in self.memory_metrics)
        avg_efficiency = np.mean([m.total_freed/m.total_allocated 
                                if m.total_allocated > 0 else 0 
                                for m in self.memory_metrics])
        
        html_report.extend([
            "<div class='metric'>",
            f"<p>Total operations profiled: {total_metrics}</p>",
            f"<p>Average peak memory: {avg_peak:.2f} MB</p>",
            f"<p>Maximum peak memory: {max_peak:.2f} MB</p>",
            f"<p>Average memory release efficiency: {avg_efficiency:.2%}</p>",
            "</div>",
            "</body></html>"
        ])
        
        # Save HTML report
        (vis_dir / f"memory_report_{timestamp}.html").write_text("\n".join(html_report))
        print(f"Visualization report saved to {vis_dir}/memory_report_{timestamp}.html")

    @memory_profile
    def profile_memory_detailed(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> None:
        """Profile memory usage with memory_profiler for line-by-line analysis."""
        if not HAS_MEMORY_PROFILER:
            print("memory_profiler not available, skipping detailed memory profiling")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = self.output_dir / f"memory_profile_detailed_{timestamp}.txt"
        
        with open(profile_path, "w") as f:
            f.write("Detailed Memory Profile\n")
            f.write("=====================\n\n")
            
            for name in images:
                f.write(f"\nProcessing {name} image:\n")
                f.write("-" * 40 + "\n")
                
                # Record initial memory state
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                f.write(f"Initial memory usage: {initial_memory:.2f} MB\n")
                
                operations = [
                    ("Feature extraction", lambda: self.feathering.create_advanced_features(
                        images[name],
                        [(50, 50)]
                    )),
                    ("Basic feathering", lambda: self.feathering.apply_lasso_feathering(
                        images[name],
                        masks[name],
                        content_aware=False
                    )),
                    ("Content-aware feathering", lambda: self.feathering.apply_lasso_feathering(
                        images[name],
                        masks[name],
                        content_aware=True
                    )),
                    ("Color-aware feathering", lambda: self.feathering.apply_color_aware_feathering(
                        images[name],
                        masks[name]
                    ))
                ]
                
                for op_name, operation in operations:
                    try:
                        import gc
                        gc.collect()
                        
                        pre_mem = process.memory_info().rss / 1024 / 1024
                        _ = operation()
                        post_mem = process.memory_info().rss / 1024 / 1024
                        peak_mem = process.memory_info().peak_wset / 1024 / 1024
                        
                        # Store metrics for visualization
                        self.memory_metrics.append(MemoryMetrics(
                            operation=op_name,
                            image_name=name,
                            pre_memory=pre_mem,
                            post_memory=post_mem,
                            peak_memory=peak_mem,
                            total_allocated=0,  # Will be updated in allocation profiling
                            total_freed=0,      # Will be updated in allocation profiling
                            timestamp=datetime.now()
                        ))
                        
                        f.write(f"\n{op_name}:\n")
                        f.write(f"  Memory before: {pre_mem:.2f} MB\n")
                        f.write(f"  Memory after: {post_mem:.2f} MB\n")
                        f.write(f"  Memory change: {post_mem - pre_mem:.2f} MB\n")
                        f.write(f"  Peak memory: {peak_mem:.2f} MB\n")
                        
                    except Exception as e:
                        f.write(f"\nError in {op_name}: {e}\n")
                        
                final_memory = process.memory_info().rss / 1024 / 1024
                f.write(f"\nFinal memory usage: {final_memory:.2f} MB\n")
                f.write(f"Total memory change: {final_memory - initial_memory:.2f} MB\n")
                
        print(f"Detailed memory profile saved to {profile_path}")

    def profile_memory_allocation(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> None:
        """Profile memory allocations using tracemalloc."""
        tracemalloc.start()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = self.output_dir / f"memory_allocation_{timestamp}.txt"
        
        def check_for_leaks(snapshot1: tracemalloc.Snapshot, snapshot2: tracemalloc.Snapshot) -> List[Dict[str, Any]]:
            """Analyze snapshots for potential memory leaks."""
            leaks = []
            stats = snapshot2.compare_to(snapshot1, 'lineno')
            for stat in stats:
                if stat.size_diff > 1024 * 1024:  # Only report leaks > 1MB
                    leaks.append({
                        'location': f"{stat.traceback[0].filename}:{stat.traceback[0].lineno}",
                        'size': stat.size_diff / (1024 * 1024),
                        'count_diff': stat.count_diff
                    })
            return leaks

        with open(profile_path, "w") as f:
            f.write("Memory Allocation Profile\n")
            f.write("=======================\n\n")
            
            for name in images:
                image_size = images[name].nbytes / (1024 * 1024)  # MB
                f.write(f"\nProcessing {name} image:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Image size: {images[name].shape}, Memory: {image_size:.2f} MB\n\n")
                
                # Take snapshot before processing
                snapshot1 = tracemalloc.take_snapshot()
                
                operations = [
                    ("Feature extraction", lambda: self.feathering.create_advanced_features(
                        images[name],
                        [(50, 50)]
                    )),
                    ("Basic feathering", lambda: self.feathering.apply_lasso_feathering(
                        images[name],
                        masks[name],
                        content_aware=False
                    )),
                    ("Content-aware feathering", lambda: self.feathering.apply_lasso_feathering(
                        images[name],
                        masks[name],
                        content_aware=True
                    )),
                    ("Color-aware feathering", lambda: self.feathering.apply_color_aware_feathering(
                        images[name],
                        masks[name]
                    ))
                ]
                
                for op_name, operation in operations:
                    try:
                        # Clear any cached data
                        import gc
                        gc.collect()
                        
                        # Take snapshot before operation
                        op_snapshot1 = tracemalloc.take_snapshot()
                        
                        # Record start time and initial memory
                        start_time = time.time()
                        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                        
                        # Execute operation
                        _ = operation()
                        
                        # Record end time and final memory
                        end_time = time.time()
                        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                        
                        # Take snapshot after operation
                        op_snapshot2 = tracemalloc.take_snapshot()
                        
                        # Compare snapshots
                        stats = op_snapshot2.compare_to(op_snapshot1, 'lineno')
                        
                        # Calculate memory metrics
                        total_alloc = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
                        total_freed = abs(sum(stat.size_diff for stat in stats if stat.size_diff < 0))
                        
                        # Update threshold manager
                        self.threshold_manager.update_history('memory_ratio', total_alloc / image_size)
                        self.threshold_manager.update_history('memory_release', total_freed / total_alloc if total_alloc > 0 else 1.0)
                        
                        # Create metrics object
                        metrics = OperationMetrics(
                            operation=op_name,
                            duration=end_time - start_time,
                            cpu_percent=psutil.Process().cpu_percent(),
                            memory_used=end_mem - start_mem,
                            gpu_used=None  # Will be updated in GPU profiling if available
                        )
                        
                        # Get optimization suggestions
                        suggestions = self.threshold_manager.get_optimization_suggestions(
                            metrics,
                            image_size
                        )
                        
                        # Check for memory leaks
                        leaks = check_for_leaks(op_snapshot1, op_snapshot2)
                        
                        # Update corresponding MemoryMetrics object
                        for metric in self.memory_metrics:
                            if metric.operation == op_name and metric.image_name == name:
                                metric.total_allocated = total_alloc / (1024 * 1024)  # Convert to MB
                                metric.total_freed = total_freed / (1024 * 1024)      # Convert to MB
                                break
                        
                        f.write(f"\n{op_name} Memory Analysis:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Duration: {end_time - start_time:.3f} seconds\n")
                        f.write(f"Memory at start: {start_mem:.2f} MB\n")
                        f.write(f"Memory at end: {end_mem:.2f} MB\n")
                        f.write(f"Memory change: {end_mem - start_mem:.2f} MB\n")
                        
                        # Write top 10 memory allocations
                        f.write("\nTop 10 Memory Allocations:\n")
                        for stat in stats[:10]:
                            f.write(f"{stat}\n")
                            
                        f.write(f"\nTotal allocated: {total_alloc / (1024 * 1024):.2f} MB\n")
                        f.write(f"Total freed: {total_freed / (1024 * 1024):.2f} MB\n")
                        f.write(f"Net change: {(total_alloc - total_freed) / (1024 * 1024):.2f} MB\n")
                        
                        # Report memory leaks
                        if leaks:
                            f.write("\nPotential Memory Leaks Detected:\n")
                            for leak in leaks:
                                f.write(f"  Location: {leak['location']}\n")
                                f.write(f"  Size: {leak['size']:.2f} MB\n")
                                f.write(f"  Object count difference: {leak['count_diff']}\n")
                        
                        # Write optimization suggestions
                        if suggestions:
                            f.write("\nOptimization Suggestions:\n")
                            for suggestion in suggestions:
                                f.write(f"{suggestion}\n")
                        
                    except Exception as e:
                        f.write(f"\nError in {op_name}: {e}\n")
                        
                # Take final snapshot
                snapshot2 = tracemalloc.take_snapshot()
                
                # Compare overall changes
                f.write("\nOverall Memory Changes:\n")
                f.write("-" * 30 + "\n")
                
                stats = snapshot2.compare_to(snapshot1, 'lineno')
                for stat in stats[:10]:
                    f.write(f"{stat}\n")
                    
        print(f"Memory allocation profile saved to {profile_path}")
        tracemalloc.stop()
        
    def profile_gpu(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> None:
        """Profile GPU operations if available."""
        if not HAS_TORCH or not TORCH_MODULE:
            print("PyTorch profiler not available, skipping GPU profiling")
            return
            
        def get_gpu_memory() -> Optional[float]:
            """Get current GPU memory usage in MB."""
            try:
                if torch.cuda.is_available():
                    return torch.cuda.memory_allocated() / (1024 * 1024)
                return None
            except Exception:
                return None
            
        for name in images:
            print(f"\nProfiling {name} image GPU operations...")
            
            # Record initial GPU state
            initial_gpu_mem = get_gpu_memory()
            
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.output_dir / "gpu_profile")
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            ) as prof:
                try:
                    start_time = time.time()
                    start_gpu_mem = get_gpu_memory()
                    
                    # Run operation
                    _ = self.feathering.apply_lasso_feathering(
                        images[name],
                        masks[name],
                        content_aware=True
                    )
                    
                    end_time = time.time()
                    end_gpu_mem = get_gpu_memory()
                    
                    # Record metrics
                    if start_gpu_mem is not None and end_gpu_mem is not None:
                        self.operation_metrics.append(OperationMetrics(
                            operation=f"GPU_{name}",
                            duration=end_time - start_time,
                            cpu_percent=psutil.Process().cpu_percent(),
                            memory_used=psutil.Process().memory_info().rss / (1024 * 1024),
                            gpu_used=end_gpu_mem - start_gpu_mem
                        ))
                        
                except Exception as e:
                    print(f"Error during GPU profiling: {e}")
                    continue
                
            # Save profiler output
            profile_path = self.output_dir / f"gpu_profile_{name}.txt"
            with open(profile_path, "w") as f:
                f.write("GPU Profile Summary\n")
                f.write("=================\n\n")
                
                # Write profiler summary
                f.write(str(prof.key_averages().table(
                    sort_by="cuda_time_total", 
                    row_limit=10
                )))
                
                # Write memory summary
                if initial_gpu_mem is not None and end_gpu_mem is not None:
                    f.write("\n\nGPU Memory Summary:\n")
                    f.write(f"Initial GPU Memory: {initial_gpu_mem:.2f} MB\n")
                    f.write(f"Final GPU Memory: {end_gpu_mem:.2f} MB\n")
                    f.write(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB\n")
                    
            print(f"GPU profile saved to {profile_path}")
            
            # Generate GPU memory timeline visualization
            if self.operation_metrics:
                plt.figure(figsize=(12, 6))
                gpu_metrics = [m for m in self.operation_metrics if m.gpu_used is not None]
                if gpu_metrics:
                    # Convert Optional[float] to float by filtering None values
                    gpu_memory_values = [m.gpu_used for m in gpu_metrics if m.gpu_used is not None]
                    if gpu_memory_values:  # Only plot if we have valid values
                        plt.plot(
                            range(len(gpu_memory_values)),
                            gpu_memory_values,
                            marker='o',
                            label='GPU Memory'
                        )
                        plt.title(f"GPU Memory Usage Timeline - {name}")
                        plt.xlabel("Operation Sequence")
                        plt.ylabel("GPU Memory (MB)")
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(self.output_dir / f"gpu_memory_timeline_{name}.png")
                        plt.close()

    def profile_feature_extraction(self, images: Dict[str, np.ndarray]) -> None:
        """Profile feature extraction performance."""
        results = []
        
        for name in images:
            print(f"\nProfiling feature extraction for {name} image...")
            start_time = time.time()
            try:
                # Profile the public method apply_lasso_feathering with content_aware=True
                # This internally uses feature extraction
                mask = np.zeros(images[name].shape[:2], dtype=np.uint8)
                cv2.circle(
                    mask,
                    (mask.shape[1]//2, mask.shape[0]//2),
                    min(mask.shape)//4,
                    (255,),
                    -1
                )
                _ = self.feathering.apply_lasso_feathering(
                    images[name],
                    mask,
                    content_aware=True
                )
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                continue
                
            duration = time.time() - start_time
            
            results.append({
                "image": name,
                "time": duration,
            })
            
        # Save results
        output = ["Feature Extraction Performance:\n"]
        output.append("| Image | Time (s) |")
        output.append("|-------|----------|")
        for result in results:
            output.append(
                f"| {result['image']} | {result['time']:.3f} |"
            )
            
        profile_path = self.output_dir / "feature_extraction_profile.md"
        profile_path.write_text('\n'.join(output))
        print(f"Feature extraction profile saved to {profile_path}")
        
    def profile_color_processing(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> None:
        """Profile color-aware processing performance."""
        results = []
        
        for name in images:
            print(f"\nProfiling color processing for {name} image...")
            
            # Profile LAB conversion
            start_time = time.time()
            _ = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            lab_time = time.time() - start_time
            
            # Profile channel processing
            start_time = time.time()
            _ = self.feathering.apply_color_aware_feathering(
                images[name],
                masks[name]
            )
            color_time = time.time() - start_time
            
            results.append({
                "image": name,
                "lab_conversion": lab_time,
                "processing": color_time,
                "total": lab_time + color_time
            })
            
        # Save results
        output = ["Color Processing Performance:\n"]
        output.append("| Image | LAB Conv. (s) | Processing (s) | Total (s) |")
        output.append("|-------|---------------|----------------|-----------|")
        for result in results:
            output.append(
                f"| {result['image']} | {result['lab_conversion']:.3f} | "
                f"{result['processing']:.3f} | {result['total']:.3f} |"
            )
            
        profile_path = self.output_dir / "color_processing_profile.md"
        profile_path.write_text('\n'.join(output))
        print(f"Color processing profile saved to {profile_path}")
        
    def analyze_correlations(self) -> None:
        """Analyze correlations between different metrics and generate insights."""
        if not self.operation_metrics:
            print("No operation metrics available for correlation analysis")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_path = self.output_dir / f"correlation_analysis_{timestamp}.md"
        
        with open(analysis_path, "w") as f:
            f.write("# Performance Correlation Analysis\n\n")
            
            # Prepare data for correlation analysis
            durations = np.array([m.duration for m in self.operation_metrics])
            cpu_usage = np.array([m.cpu_percent for m in self.operation_metrics])
            memory_usage = np.array([m.memory_used for m in self.operation_metrics])
            
            # Calculate correlations
            f.write("## Metric Correlations\n\n")
            f.write("| Metric Pair | Correlation |\n")
            f.write("|-------------|-------------|\n")
            
            # Duration vs CPU
            corr_duration_cpu = np.corrcoef(durations, cpu_usage)[0, 1]
            f.write(f"| Duration vs CPU Usage | {corr_duration_cpu:.3f} |\n")
            
            # Duration vs Memory
            corr_duration_mem = np.corrcoef(durations, memory_usage)[0, 1]
            f.write(f"| Duration vs Memory Usage | {corr_duration_mem:.3f} |\n")
            
            # CPU vs Memory
            corr_cpu_mem = np.corrcoef(cpu_usage, memory_usage)[0, 1]
            f.write(f"| CPU vs Memory Usage | {corr_cpu_mem:.3f} |\n")
            
            # Add GPU correlations if available
            gpu_metrics = [m.gpu_used for m in self.operation_metrics if m.gpu_used is not None]
            if gpu_metrics:
                gpu_usage = np.array(gpu_metrics)
                if len(gpu_usage) == len(durations):
                    corr_duration_gpu = np.corrcoef(durations, gpu_usage)[0, 1]
                    corr_cpu_gpu = np.corrcoef(cpu_usage, gpu_usage)[0, 1]
                    corr_mem_gpu = np.corrcoef(memory_usage, gpu_usage)[0, 1]
                    
                    f.write(f"| Duration vs GPU Usage | {corr_duration_gpu:.3f} |\n")
                    f.write(f"| CPU vs GPU Usage | {corr_cpu_gpu:.3f} |\n")
                    f.write(f"| Memory vs GPU Usage | {corr_mem_gpu:.3f} |\n")
            
            # Generate insights
            f.write("\n## Performance Insights\n\n")
            
            # Analyze duration patterns
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            slow_ops = [(m.operation, m.duration) for m in self.operation_metrics 
                       if m.duration > mean_duration + 2*std_duration]
            
            if slow_ops:
                f.write("### Slow Operations\n")
                f.write("The following operations are significantly slower than average:\n\n")
                for op, dur in slow_ops:
                    f.write(f"- {op}: {dur:.3f}s ({(dur/mean_duration - 1)*100:.1f}% slower than average)\n")
            
            # Analyze memory patterns
            mean_memory = np.mean(memory_usage)
            std_memory = np.std(memory_usage)
            high_mem_ops = [(m.operation, m.memory_used) for m in self.operation_metrics 
                           if m.memory_used > mean_memory + 2*std_memory]
            
            if high_mem_ops:
                f.write("\n### High Memory Usage\n")
                f.write("The following operations use significantly more memory than average:\n\n")
                for op, mem in high_mem_ops:
                    f.write(f"- {op}: {mem:.1f}MB ({(mem/mean_memory - 1)*100:.1f}% higher than average)\n")
            
            # Analyze CPU-Memory relationship
            if abs(corr_cpu_mem) > 0.7:
                f.write("\n### CPU-Memory Correlation\n")
                if corr_cpu_mem > 0:
                    f.write("Strong positive correlation between CPU and memory usage suggests ")
                    f.write("operations are compute-bound and may benefit from parallelization.\n")
                else:
                    f.write("Strong negative correlation between CPU and memory usage suggests ")
                    f.write("potential memory-bound operations that could benefit from optimization.\n")
            
            # Add recommendations
            f.write("\n## Recommendations\n\n")
            
            if slow_ops:
                f.write("1. Consider optimizing the identified slow operations through:\n")
                f.write("   - Algorithmic improvements\n")
                f.write("   - Parallelization where possible\n")
                f.write("   - Caching intermediate results\n")
            
            if high_mem_ops:
                f.write("\n2. Address high memory usage by:\n")
                f.write("   - Implementing batch processing\n")
                f.write("   - Optimizing data structures\n")
                f.write("   - Adding memory limits and cleanup\n")
            
            if gpu_metrics:
                gpu_util = np.mean(gpu_metrics) / np.max(gpu_metrics) if gpu_metrics else 0
                if gpu_util < 0.5:
                    f.write("\n3. Improve GPU utilization:\n")
                    f.write("   - Increase batch sizes\n")
                    f.write("   - Move more operations to GPU\n")
                    f.write("   - Optimize memory transfers\n")
            
        print(f"Correlation analysis saved to {analysis_path}")
        
        # Visualize correlations
        plt.figure(figsize=(10, 8))
        metrics = ['Duration', 'CPU', 'Memory']
        corr_matrix = np.array([
            [1.0, corr_duration_cpu, corr_duration_mem],
            [corr_duration_cpu, 1.0, corr_cpu_mem],
            [corr_duration_mem, corr_cpu_mem, 1.0]
        ])
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            xticklabels=metrics,
            yticklabels=metrics,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"correlation_heatmap_{timestamp}.png")
        plt.close()

    def collect_io_metrics(self) -> Dict[str, float]:
        """Collect IO metrics including read/write latency and bandwidth usage."""
        current_time = time.time()
        current_disk_io = psutil.disk_io_counters()
        if current_disk_io is None or self.last_disk_io is None:
            return {
                'read_latency': 0.0,
                'write_latency': 0.0,
                'bandwidth_usage': 0.0
            }
            
        time_delta = current_time - self.last_io_time

        # Calculate IO rates
        read_bytes = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
        write_bytes = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
        
        # Calculate latencies (ms)
        read_latency = (current_disk_io.read_time / current_disk_io.read_count 
                       if current_disk_io.read_count > 0 else 0)
        write_latency = (current_disk_io.write_time / current_disk_io.write_count 
                        if current_disk_io.write_count > 0 else 0)
        
        # Calculate bandwidth usage (assuming max bandwidth of 500MB/s for example)
        max_bandwidth = 500 * 1024 * 1024  # 500MB/s in bytes
        total_bandwidth = read_bytes + write_bytes
        bandwidth_usage = (total_bandwidth / max_bandwidth) * 100 if max_bandwidth > 0 else 0
        
        # Update last values
        self.last_disk_io = current_disk_io
        self.last_io_time = current_time
        
        # Store metrics
        metrics = {
            'read_latency': float(read_latency),
            'write_latency': float(write_latency),
            'bandwidth_usage': float(bandwidth_usage)
        }
        
        for key, value in metrics.items():
            self.io_metrics[key].append(value)
            
        return metrics

    def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics including latency, bandwidth usage, and packet loss."""
        current_time = time.time()
        current_net_io = psutil.net_io_counters()
        if current_net_io is None or self.last_net_io is None:
            return {
                'latency': 0.0,
                'bandwidth_usage': 0.0,
                'packet_loss': 0.0
            }
            
        time_delta = current_time - self.last_io_time
        
        # Calculate network rates
        bytes_sent = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / time_delta
        bytes_recv = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / time_delta
        packets_sent = (current_net_io.packets_sent - self.last_net_io.packets_sent) / time_delta
        packets_recv = (current_net_io.packets_recv - self.last_net_io.packets_recv) / time_delta
        
        # Calculate packet loss
        packet_loss = 0.0
        if packets_sent > 0:
            packet_loss = ((current_net_io.packets_sent - current_net_io.packets_recv) / 
                          current_net_io.packets_sent) * 100
        
        # Calculate network latency (using ping)
        try:
            ping_result = subprocess.run(['ping', '-n', '1', 'localhost'], 
                                       capture_output=True, text=True, timeout=1)
            match = re.search(r'time[=<](\d+)ms', ping_result.stdout)
            latency = float(match.group(1)) if match else 0.0
        except (subprocess.TimeoutExpired, AttributeError, ValueError):
            latency = 0.0
        
        # Calculate bandwidth usage (assuming 1Gbps network)
        max_bandwidth = 125_000_000  # 1Gbps in bytes/s
        total_bandwidth = bytes_sent + bytes_recv
        bandwidth_usage = (total_bandwidth / max_bandwidth) * 100 if max_bandwidth > 0 else 0
        
        # Update last values
        self.last_net_io = current_net_io
        
        # Store metrics
        metrics = {
            'latency': float(latency),
            'bandwidth_usage': float(bandwidth_usage),
            'packet_loss': float(packet_loss)
        }
        
        for key, value in metrics.items():
            self.network_metrics[key].append(value)
            
        return metrics

    def run_all_profiles(self) -> None:
        """Run all profiling operations."""
        print("Creating test data...")
        images, masks = self._create_test_data()
        
        print("\nRunning CPU profiling...")
        self.profile_cpu(images, masks)
        
        print("\nRunning detailed memory profiling...")
        self.profile_memory_detailed(images, masks)
        
        print("\nRunning memory allocation profiling...")
        self.profile_memory_allocation(images, masks)
        
        print("\nRunning GPU profiling...")
        self.profile_gpu(images, masks)
        
        print("\nProfiling feature extraction...")
        self.profile_feature_extraction(images)
        
        print("\nProfiling color processing...")
        self.profile_color_processing(images, masks)
        
        print("\nAnalyzing correlations...")
        self.analyze_correlations()
        
        print("\nGenerating visualizations...")
        self.visualize_memory_usage()
        
        print("\nVisualizing threshold evolution...")
        self.threshold_manager.visualize_thresholds(self.output_dir)
        
        print("\nGenerating summary...")
        self.generate_summary()
        
        print("\nExporting violation data...")
        self.threshold_manager.export_violations('json')
        self.threshold_manager.export_violations('csv')
        self.threshold_manager.export_threshold_history('json')
        self.threshold_manager.export_metrics_summary()
        
    def generate_summary(self) -> None:
        """Generate a summary of all profiling results."""
        summary = ["# Performance Profiling Summary\n"]
        
        # Add CPU profiling summary
        summary.append("## CPU Performance\n")
        for profile in self.output_dir.glob("cpu_profile_*.txt"):
            summary.append(f"### {profile.stem}\n")
            content = profile.read_text().split('\n')[:20]  # Top 20 lines
            summary.extend([f"```", *content, "```\n"])
            
        # Add detailed memory profiling summary
        if list(self.output_dir.glob("memory_profile_detailed_*.txt")):
            summary.append("## Detailed Memory Usage\n")
            for profile in self.output_dir.glob("memory_profile_detailed_*.txt"):
                summary.append(f"### {profile.stem}\n")
                content = profile.read_text().split('\n')[:20]
                summary.extend([f"```", *content, "```\n"])
            
        # Add memory allocation profiling summary
        if list(self.output_dir.glob("memory_allocation_*.txt")):
            summary.append("## Memory Allocation\n")
            for profile in self.output_dir.glob("memory_allocation_*.txt"):
                summary.append(f"### {profile.stem}\n")
                content = profile.read_text().split('\n')[:20]
                summary.extend([f"```", *content, "```\n"])
            
        # Add feature extraction summary
        if (self.output_dir / "feature_extraction_profile.md").exists():
            summary.append("## Feature Extraction\n")
            summary.append((self.output_dir / "feature_extraction_profile.md").read_text())
            
        # Add color processing summary
        if (self.output_dir / "color_processing_profile.md").exists():
            summary.append("## Color Processing\n")
            summary.append((self.output_dir / "color_processing_profile.md").read_text())
            
        # Add threshold evolution summary
        threshold_reports = list(self.output_dir.glob("visualizations/threshold_report_*.html"))
        if threshold_reports:
            summary.append("\n## Threshold Evolution\n")
            summary.append("Threshold evolution reports have been generated with visualizations showing how performance thresholds adapt over time. ")
            summary.append("The reports include:\n")
            summary.append("- Threshold change trends")
            summary.append("- Statistical analysis of metric values")
            summary.append("- Visual timeline of threshold adjustments")
            summary.append("\nReports can be found in the visualizations directory.\n")
            
        # Save summary
        summary_path = self.output_dir / "profile_summary.md"
        summary_path.write_text('\n'.join(summary))
        print(f"\nProfile summary saved to {summary_path}")

    def test_forecasting_methods(self) -> None:
        """Test and verify all forecasting methods."""
        print("\nTesting Forecasting Methods:")
        print("===========================")
        
        # Create synthetic test data with known patterns
        times = [datetime.now() + timedelta(hours=i) for i in range(48)]
        # Generate data with trend, seasonality, and noise
        trend = np.linspace(0, 10, 48)
        seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, 48))
        noise = np.random.normal(0, 0.5, 48)
        values = trend + seasonality + noise
        
        print("\nTesting with synthetic data:")
        print(f"Data points: {len(values)}")
        print(f"Time range: {times[0]} to {times[-1]}")
        
        # Test each forecasting method
        for method in ForecastMethod:
            print(f"\nTesting {method.name} forecasting:")
            try:
                self.threshold_manager._forecast_method = method
                forecast = self.threshold_manager._generate_forecast(times, np.array(values, dtype=np.float64))
                
                if forecast:
                    # Calculate error metrics on the last few points
                    test_size = min(5, len(values) // 3)
                    test_values = values[-test_size:]
                    pred_values = [f[1] for f in forecast[:test_size]]
                    rmse = np.sqrt(np.mean((test_values - pred_values) ** 2))
                    mae = np.mean(np.abs(test_values - pred_values))
                    
                    print(f"Forecast generated {len(forecast)} points")
                    print(f"RMSE: {rmse:.3f}")
                    print(f"MAE: {mae:.3f}")
                else:
                    print("No forecast generated")
            except Exception as e:
                print(f"Error testing {method.name}: {e}")
        
        # Test automatic method selection
        print("\nTesting automatic method selection:")
        analysis = self.threshold_manager.analyze_trend_detailed("test_metric")
        if analysis:
            print(f"Selected method produced forecast with {len(analysis.forecast)} points")
            print(f"Detected trend: {analysis.trend} (confidence: {analysis.confidence:.2%})")
            if analysis.seasonality:
                print(f"Detected seasonality with period: {analysis.seasonality:.1f}")
            print(f"Found {len(analysis.outliers)} outliers and {len(analysis.change_points)} change points")
        else:
            print("No trend analysis produced")

if __name__ == "__main__":
    profiler = PerformanceProfiler()
    profiler.test_forecasting_methods()  # Test forecasting first
    profiler.threshold_manager.set_forecast_method(ForecastMethod.ARIMA)
    profiler.run_all_profiles() 