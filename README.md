# HelixZone Image Processing Library

## Overview
HelixZone is a high-performance image processing library focusing on advanced selection and feathering operations. It combines machine learning techniques with traditional image processing to provide state-of-the-art results.

## Features
- Lasso-based feathering with content-aware processing
- Color-aware selection in LAB color space
- GPU-accelerated operations (via CUDA/OpenCL)
- Advanced texture and feature extraction
- Multi-threaded processing for optimal performance

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from helixzone.core.ml_utils import EnhancedLassoFeathering

# Initialize feathering
feathering = EnhancedLassoFeathering()

# Apply feathering
result = feathering.apply_lasso_feathering(
    image,
    mask,
    alpha=0.01,
    content_aware=True
)
```

## Technical Documentation

### Optimization Strategy
The library uses several sophisticated optimization techniques:

1. **ElasticNet Regression**
   - Two-stage fitting process:
     - Initial fit with high regularization (5x alpha) for stability
     - Refinement with target alpha using pre-fitted coefficients
   - Robust feature scaling with quantile range (1, 99)
   - Sample weights to handle outliers (points > 2 std dev get 0.5 weight)

2. **Color Processing**
   - LAB color space processing for perceptual uniformity
   - Channel-specific optimization:
     - L: 2.0x alpha for overall smoothness
     - A/B: 0.5x alpha for color preservation
   - Edge-aware processing with adaptive alpha values

3. **Performance Optimization**
   - Multi-threading for non-blocking UI updates
   - GPU acceleration for computationally intensive tasks
   - Memory-efficient sparse matrix operations
   - Adaptive patch sizes based on image content

### Performance Metrics
- Test coverage: 92% for core functionality
- Processing time: < 100ms for 1MP images
- Memory usage: < 2x image size
- GPU acceleration: Up to 10x speedup for large images

## Development Guidelines

### Project Structure
```
helixzone/
├── core/
│   ├── ml_utils.py      # Core ML functionality
│   ├── gpu_utils.py     # GPU acceleration
│   └── image_utils.py   # Image processing utilities
├── ui/
│   └── main_window.py   # PyQt6 UI components
└── tests/
    └── test_ml_utils.py # Comprehensive test suite
```

### Coding Standards
1. **Python Style**
   - Follow PEP 8 and PEP 257
   - Use type hints for all function signatures
   - Document all public functions and classes

2. **Testing**
   - Maintain > 90% test coverage
   - Include edge cases and error conditions
   - Use pytest for all tests

3. **Performance**
   - Profile code for bottlenecks
   - Use vectorized operations where possible
   - Implement GPU acceleration for heavy computations

### Error Handling
1. **Input Validation**
   - Validate image dimensions and types
   - Check mask compatibility
   - Verify coordinate bounds

2. **Resource Management**
   - Release GPU resources properly
   - Handle memory efficiently
   - Clean up temporary files

## Benchmarks and Optimization

### Feature Extraction
- Basic features: < 1ms per patch
- Gabor features: < 2ms per orientation
- Color features: < 1ms per channel

### Memory Usage
- Feature matrix: O(n_points * n_features)
- Sparse matrix optimization: ~50% memory reduction
- GPU memory: Maximum 4GB for 4K images

### Processing Time
| Operation | CPU Time | GPU Time |
|-----------|----------|----------|
| Feathering | 100ms   | 10ms     |
| Feature Extraction | 50ms | 5ms  |
| Color Processing | 150ms | 15ms  |

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License
MIT License - See LICENSE file for details

# HelixZone Performance Monitoring

A comprehensive performance monitoring and profiling system for HelixZone, featuring a real-time dashboard for visualizing metrics, thresholds, and violations.

## Features

- Real-time performance metric visualization
- Dynamic threshold adjustment
- Automated violation detection and alerting
- Forecasting with multiple methods (ARIMA, Prophet, Polynomial)
- Interactive metric plots with customizable time windows
- Violation history tracking and analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For optimal forecasting capabilities, ensure you have the following optional dependencies:
```bash
pip install statsmodels prophet
```

## Usage

### Running the Dashboard

1. Start the performance monitoring dashboard:
```python
from benchmarks import launch_dashboard, ThresholdManager

# Create and configure threshold manager
threshold_manager = ThresholdManager()

# Launch the dashboard
launch_dashboard(threshold_manager)
```

2. For a demo with simulated data:
```bash
python -m benchmarks.test_dashboard
```

### Configuring Metrics

```python
from benchmarks import ThresholdConfig, MonitoringConfig, MetricViolationConfig

# Configure thresholds
threshold_config = ThresholdConfig(
    memory_ratio=3.0,
    memory_release=0.8,
    memory_retention=50.0,
    duration_outlier_std=2.0,
    gpu_utilization=0.5,
    cpu_threshold=80.0
)

# Configure monitoring
monitoring_config = MonitoringConfig(
    enabled=True,
    update_interval=1.0,  # seconds
    max_queue_size=1000,
    retention_period=3600  # seconds
)

# Create threshold manager
threshold_manager = ThresholdManager(
    config=threshold_config,
    monitoring_config=monitoring_config
)

# Configure specific metric thresholds
threshold_manager.configure_metric_thresholds(
    'memory_ratio',
    MetricViolationConfig(
        warning_threshold=1.2,
        critical_threshold=1.5,
        consecutive_violations=3,
        notification_level='all'
    )
)
```

### Adding Metrics

```python
# Add a metric update
threshold_manager.add_metric_update('memory_ratio', 2.5)
```

## Dashboard Features

### Metric Plots
- Real-time value plotting
- Dynamic threshold lines
- Warning and critical threshold indicators
- Forecast visualization
- Adjustable time window
- Selectable forecasting method

### Violation Panel
- Real-time violation notifications
- Severity-based color coding
- Historical violation tracking
- Violation details and timestamps

## Development

### Adding New Metrics

1. Define the metric configuration:
```python
metric_config = {
    'base': 2.0,
    'noise': 0.5,
    'trend': 0.01,
    'warning': 1.2,
    'critical': 1.5
}
```

2. Configure the threshold manager:
```python
threshold_manager.configure_metric_thresholds(
    'new_metric',
    MetricViolationConfig(
        warning_threshold=metric_config['warning'],
        critical_threshold=metric_config['critical']
    )
)
```

### Extending the Dashboard

The dashboard is built with PyQt6 and can be extended by:
- Adding new visualization widgets
- Implementing additional analysis features
- Creating custom metric processors
- Adding new forecasting methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
