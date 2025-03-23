"""
Progress indicator components for providing visual feedback for operations.

This module provides various progress indicator widgets that can be used
throughout the application to give visual feedback for long-running operations.
"""

import logging
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QProgressBar, QLabel, QHBoxLayout, 
    QVBoxLayout, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent

from ..core.task_manager import TaskManager, TaskStatus, get_task_manager

logger = logging.getLogger(__name__)


class CircularProgressIndicator(QWidget):
    """A circular progress indicator that shows indeterminate or determinate progress."""
    
    def __init__(self, parent=None, size=24):
        """Initialize the circular progress indicator.
        
        Args:
            parent: Parent widget
            size: Size of the indicator in pixels
        """
        super().__init__(parent)
        self.setFixedSize(size, size)
        
        # Progress properties
        self._progress = 0
        self._max_progress = 100
        self._indeterminate = True
        self._angle = 0
        
        # Appearance
        self._color = QColor(0, 120, 215)  # Default blue color
        
        # Animation timer for indeterminate mode
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.setInterval(50)  # Update every 50ms
        
        # Start animation by default
        self.start_animation()
        
    def set_progress(self, value):
        """Set the progress value (0-100)."""
        self._progress = max(0, min(value, self._max_progress))
        self._indeterminate = False
        self.update()
        
    def set_indeterminate(self, indeterminate=True):
        """Set whether the progress is indeterminate."""
        self._indeterminate = indeterminate
        if indeterminate:
            self.start_animation()
        else:
            self.stop_animation()
        self.update()
    
    def start_animation(self):
        """Start the animation for indeterminate mode."""
        self._timer.start()
        
    def stop_animation(self):
        """Stop the animation."""
        self._timer.stop()
        
    def _update_animation(self):
        """Update the animation for indeterminate mode."""
        self._angle = (self._angle + 10) % 360
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the progress indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(240, 240, 240))
        painter.drawEllipse(2, 2, self.width() - 4, self.height() - 4)
        
        # Draw progress
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        
        if self._indeterminate:
            # Draw arc for indeterminate mode
            painter.save()
            painter.translate(self.width() / 2, self.height() / 2)
            painter.rotate(self._angle)
            painter.drawPie(-self.width() / 2 + 2, -self.height() / 2 + 2,
                           self.width() - 4, self.height() - 4,
                           0, 120 * 16)  # 120 degrees (16 = 1 degree in QPainter)
            painter.restore()
        else:
            # Draw arc for determinate mode
            span_angle = int(360 * self._progress / self._max_progress) * 16
            painter.drawPie(2, 2, self.width() - 4, self.height() - 4,
                           90 * 16, -span_angle)  # Start from top (90 degrees)
            
        painter.end()


class CompactProgressIndicator(QWidget):
    """A compact progress indicator with optional label for inline display."""
    
    canceled = pyqtSignal()
    
    def __init__(self, parent=None, with_label=True, with_cancel=True):
        """Initialize the compact progress indicator.
        
        Args:
            parent: Parent widget
            with_label: Whether to include a label
            with_cancel: Whether to include a cancel button
        """
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Circular indicator
        self.indicator = CircularProgressIndicator(self, size=16)
        layout.addWidget(self.indicator)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setMinimumWidth(80)
        self.progress_bar.setMaximumWidth(150)
        layout.addWidget(self.progress_bar)
        
        # Label
        self.label = None
        if with_label:
            self.label = QLabel("Processing...")
            self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.label)
            
        # Cancel button
        self.cancel_button = None
        if with_cancel:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.setFixedHeight(20)
            self.cancel_button.setFixedWidth(60)
            self.cancel_button.clicked.connect(self.canceled.emit)
            layout.addWidget(self.cancel_button)
        
        # Set default to indeterminate
        self.set_indeterminate(True)
        
    def set_progress(self, value):
        """Set the progress value (0-100)."""
        self.indicator.set_progress(value)
        self.progress_bar.setValue(value)
        
    def set_indeterminate(self, indeterminate=True):
        """Set whether the progress is indeterminate."""
        self.indicator.set_indeterminate(indeterminate)
        if indeterminate:
            self.progress_bar.setRange(0, 0)  # Indeterminate mode
        else:
            self.progress_bar.setRange(0, 100)
            
    def set_text(self, text):
        """Set the label text if a label is present."""
        if self.label:
            self.label.setText(text)


class TaskProgressIndicator(QWidget):
    """A progress indicator that automatically updates based on a task."""
    
    def __init__(self, parent=None, task_id=None, compact=True):
        """Initialize the task progress indicator.
        
        Args:
            parent: Parent widget
            task_id: ID of the task to track
            compact: Whether to use compact mode
        """
        super().__init__(parent)
        
        self.task_id = task_id
        self.task_manager = get_task_manager()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create progress indicator
        if compact:
            self.progress = CompactProgressIndicator(self)
        else:
            self.progress = QProgressBar(self)
            self.progress.setTextVisible(True)
            
        layout.addWidget(self.progress)
        
        # Update timer
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_progress)
        self._update_timer.setInterval(100)  # Update every 100ms
        
        # Connect cancel signal if available
        if hasattr(self.progress, 'canceled'):
            self.progress.canceled.connect(self._cancel_task)
        
        # Start tracking if task_id is provided
        if task_id:
            self.track_task(task_id)
    
    def track_task(self, task_id):
        """Track a specific task."""
        self.task_id = task_id
        self._update_timer.start()
        self._update_progress()
    
    def stop_tracking(self):
        """Stop tracking the task."""
        self._update_timer.stop()
        
    def _update_progress(self):
        """Update progress based on task status."""
        if not self.task_id:
            return
            
        task = self.task_manager.get_task(self.task_id)
        if not task:
            self.stop_tracking()
            return
            
        progress = task.progress
        
        # Update progress
        if isinstance(self.progress, CompactProgressIndicator):
            if progress.indeterminate:
                self.progress.set_indeterminate(True)
            else:
                self.progress.set_indeterminate(False)
                self.progress.set_progress(progress.percent)
                
            # Update text if available
            if hasattr(self.progress, 'set_text'):
                self.progress.set_text(progress.message or "Processing...")
        else:
            # Standard QProgressBar
            if progress.indeterminate:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(progress.percent)
                
            # Update format string
            if progress.message:
                self.progress.setFormat(f"{progress.message}: %p%")
            else:
                self.progress.setFormat("%p%")
                
        # Check if task is completed
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.stop_tracking()
            
    def _cancel_task(self):
        """Cancel the current task."""
        if self.task_id:
            self.task_manager.cancel_task(self.task_id) 