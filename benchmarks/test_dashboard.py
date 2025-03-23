"""
Test dashboard with simulated metrics.
"""

import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from .profile_performance import (
    ThresholdManager,
    MetricViolationConfig,
    ForecastMethod
)
from .performance_dashboard import launch_dashboard

def main() -> None:
    """Run the test dashboard with simulated metrics."""
    # Configure metrics with correlated patterns
    metrics = {
        'memory_ratio': {
            'base': 2.0,
            'noise': 0.2,  # Reduced noise
            'trend': 0.005,  # Reduced trend
            'warning': 2.5,  # Increased warning threshold
            'critical': 3.0,  # Increased critical threshold
            'phase': 0,
            'adaptive_window': 100  # Window for adaptive thresholds
        },
        'gpu_utilization': {
            'base': 0.3,
            'noise': 0.05,  # Reduced noise
            'trend': -0.0005,  # Reduced trend
            'warning': 0.75,  # Adjusted warning threshold
            'critical': 0.85,  # Adjusted critical threshold
            'phase': np.pi/2,
            'adaptive_window': 100
        },
        'gpu_threshold': {
            'base': 60.0,
            'noise': 5.0,  # Reduced noise
            'trend': 0.02,  # Reduced trend
            'warning': 80.0,  # Adjusted warning threshold
            'critical': 90.0,  # Adjusted critical threshold
            'phase': np.pi/4,
            'adaptive_window': 100
        }
    }
    
    # Create and configure threshold manager
    threshold_manager = ThresholdManager()
    
    # Configure thresholds with more sophisticated violation detection
    for metric_name, config in metrics.items():
        threshold_manager.configure_metric_thresholds(
            metric_name,
            MetricViolationConfig(
                warning_threshold=config['warning'],
                critical_threshold=config['critical'],
                consecutive_violations=5,  # Increased to reduce false positives
                notification_level='all',
                adaptive_window=config['adaptive_window'],
                min_violations_for_alert=3,  # Minimum violations before alerting
                cooldown_period=60  # Cooldown period between alerts in seconds
            )
        )
    
    # Initialize moving windows
    moving_windows = {metric: [] for metric in metrics}
    
    def update_metrics() -> None:
        """Update metrics with simulated values."""
        # Generate common factors with reduced volatility
        common_noise = random.gauss(0, 0.5)  # Reduced common noise
        periodic = 0.1 * np.sin(time.time() / 100)  # Reduced periodic component
        
        for metric_name, config in metrics.items():
            # Update moving window
            window = moving_windows[metric_name]
            
            # Generate value with reduced volatility
            value = (
                config['base'] +
                config['trend'] * time.time() +
                config['noise'] * (0.7 * common_noise + 0.3 * random.gauss(0, 0.5)) +
                periodic * np.sin(time.time() / 100 + config['phase'])
            )
            
            # Add to moving window
            window.append(value)
            if len(window) > config['adaptive_window']:
                window.pop(0)
            
            # Calculate adaptive thresholds
            if len(window) >= 30:  # Minimum window size for statistics
                mean = np.mean(window)
                std = np.std(window)
                
                # Adjust value based on historical patterns
                if value > mean + 3 * std:  # Potential outlier
                    value = mean + 2 * std  # Cap the value
                
                # Update metric with smoothed value
                threshold_manager.add_metric_update(metric_name, value)
            else:
                threshold_manager.add_metric_update(metric_name, value)
            
            # Add occasional correlated spikes with reduced frequency
            if random.random() < 0.02:  # Reduced spike probability
                spike_factor = 1.2  # Reduced spike magnitude
                if all(len(w) >= 30 for w in moving_windows.values()):
                    # Only add spike if all metrics are stable
                    for other_metric, other_config in metrics.items():
                        if other_metric != metric_name and random.random() < 0.4:
                            other_window = moving_windows[other_metric]
                            other_mean = np.mean(other_window)
                            other_value = other_mean * spike_factor
                            threshold_manager.add_metric_update(other_metric, other_value)
                    value *= spike_factor
                    threshold_manager.add_metric_update(metric_name, value)
    
    # Start background thread for metric updates
    import threading
    stop_event = threading.Event()
    
    def update_thread() -> None:
        while not stop_event.is_set():
            update_metrics()
            time.sleep(0.1)  # Update every 100ms
    
    update_thread = threading.Thread(target=update_thread)
    update_thread.daemon = True
    update_thread.start()
    
    try:
        # Launch dashboard
        launch_dashboard(threshold_manager)
    finally:
        stop_event.set()
        update_thread.join()

if __name__ == "__main__":
    main() 