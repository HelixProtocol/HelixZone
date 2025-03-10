"""
Performance monitoring dashboard for HelixZone.
"""

import sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox
)
from PyQt6.QtCore import QTimer

from .profile_performance import ThresholdManager

class DashboardWindow(QMainWindow):
    """Main window for the performance dashboard."""
    
    def __init__(self, threshold_manager: ThresholdManager) -> None:
        super().__init__()
        self.threshold_manager = threshold_manager
        self.setWindowTitle("HelixZone Performance Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add tabs
        tabs.addTab(self._create_metrics_tab(), "System Metrics")
        tabs.addTab(self._create_io_tab(), "I/O Metrics")
        tabs.addTab(self._create_network_tab(), "Network Metrics")
        tabs.addTab(self._create_analysis_tab(), "Analysis")
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(1000)  # Update every second
        
    def _create_metrics_tab(self) -> QWidget:
        """Create the system metrics monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add metric plots
        self.metric_figures = {}
        for metric_name in self.threshold_manager.metric_violation_configs:
            fig = Figure(figsize=(8, 3))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.metric_figures[metric_name] = (fig, canvas)
            
        return tab
        
    def _create_io_tab(self) -> QWidget:
        """Create the I/O metrics monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add I/O metric plots
        self.io_figures = {}
        io_metrics = ['read_latency', 'write_latency', 'bandwidth_usage']
        
        for metric in io_metrics:
            fig = Figure(figsize=(8, 3))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.io_figures[metric] = (fig, canvas)
            
        return tab
        
    def _create_network_tab(self) -> QWidget:
        """Create the network metrics monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add network metric plots
        self.network_figures = {}
        network_metrics = ['latency', 'bandwidth_usage', 'packet_loss']
        
        for metric in network_metrics:
            fig = Figure(figsize=(8, 3))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.network_figures[metric] = (fig, canvas)
            
        return tab
        
    def _create_analysis_tab(self) -> QWidget:
        """Create the analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add controls
        controls = QHBoxLayout()
        layout.addLayout(controls)
        
        # Add metric selector
        self.metric_selector = QComboBox()
        metrics = list(self.threshold_manager.metric_violation_configs.keys())
        metrics.extend(['io_read_latency', 'io_write_latency', 'io_bandwidth',
                       'net_latency', 'net_bandwidth', 'net_packet_loss'])
        self.metric_selector.addItems(metrics)
        controls.addWidget(QLabel("Metric:"))
        controls.addWidget(self.metric_selector)
        
        # Add export buttons
        export_csv = QPushButton("Export CSV")
        export_csv.clicked.connect(lambda: self.threshold_manager.export_violations('csv'))
        controls.addWidget(export_csv)
        
        export_json = QPushButton("Export JSON")
        export_json.clicked.connect(lambda: self.threshold_manager.export_violations('json'))
        controls.addWidget(export_json)
        
        # Add correlation plot
        self.corr_figure = Figure(figsize=(8, 6))
        self.corr_canvas = FigureCanvas(self.corr_figure)
        layout.addWidget(self.corr_canvas)
        
        return tab
        
    def update_plots(self) -> None:
        """Update all plots with current data."""
        # Update system metric plots
        for metric_name, (fig, canvas) in self.metric_figures.items():
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Get metric history
            history = self.threshold_manager.history.get(metric_name, [])
            if history:
                times = range(len(history))
                values = history
                
                # Plot metric values
                ax.plot(times, values, label='Value')
                
                # Plot thresholds
                config = self.threshold_manager.metric_violation_configs[metric_name]
                ax.axhline(y=config.warning_threshold, color='yellow', linestyle='--', label='Warning')
                ax.axhline(y=config.critical_threshold, color='red', linestyle='--', label='Critical')
                
                ax.set_title(f"{metric_name} Evolution")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                
            canvas.draw()
            
        # Update I/O metric plots
        for metric_name, (fig, canvas) in self.io_figures.items():
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Get I/O metric history
            history = self.threshold_manager.io_metrics.get(metric_name, [])
            if history:
                times = range(len(history))
                values = history
                
                # Plot metric values
                ax.plot(times, values, label='Value')
                
                ax.set_title(f"I/O {metric_name.replace('_', ' ').title()}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                
            canvas.draw()
            
        # Update network metric plots
        for metric_name, (fig, canvas) in self.network_figures.items():
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Get network metric history
            history = self.threshold_manager.network_metrics.get(metric_name, [])
            if history:
                times = range(len(history))
                values = history
                
                # Plot metric values
                ax.plot(times, values, label='Value')
                
                ax.set_title(f"Network {metric_name.replace('_', ' ').title()}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                
            canvas.draw()
            
        # Update correlation plot if on analysis tab
        if self.metric_selector.currentText():
            self.corr_figure.clear()
            ax = self.corr_figure.add_subplot(111)
            
            # Get all metric values
            metrics = {}
            
            # System metrics
            for metric in self.threshold_manager.metric_violation_configs:
                if metric in self.threshold_manager.history:
                    metrics[metric] = self.threshold_manager.history[metric]
            
            # I/O metrics
            for metric, values in self.threshold_manager.io_metrics.items():
                metrics[f"io_{metric}"] = values
            
            # Network metrics
            for metric, values in self.threshold_manager.network_metrics.items():
                metrics[f"net_{metric}"] = values
            
            # Find minimum length
            min_length = float('inf')
            for values in metrics.values():
                if values:
                    min_length = min(min_length, len(values))
            
            if min_length < float('inf'):
                # Prepare correlation data
                metric_names = list(metrics.keys())
                values = []
                for metric in metric_names:
                    if metrics[metric]:
                        values.append(metrics[metric][-min_length:])
                
                if values:
                    # Calculate correlation matrix
                    corr_matrix = np.corrcoef(values)
                    
                    # Plot correlation heatmap
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    self.corr_figure.colorbar(im)
                    
                    # Add labels
                    ax.set_xticks(range(len(metric_names)))
                    ax.set_yticks(range(len(metric_names)))
                    ax.set_xticklabels(metric_names, rotation=45)
                    ax.set_yticklabels(metric_names)
                    
                    ax.set_title("Metric Correlations")
                    
            self.corr_canvas.draw()

def launch_dashboard(threshold_manager: ThresholdManager) -> None:
    """Launch the performance monitoring dashboard."""
    app = QApplication(sys.argv)
    window = DashboardWindow(threshold_manager)
    window.show()
    sys.exit(app.exec()) 