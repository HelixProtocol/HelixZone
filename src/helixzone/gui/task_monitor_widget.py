"""Task monitor widget for displaying active tasks.

This module provides a widget that displays active tasks in the application,
showing their status and progress.
"""

import logging
from typing import Dict, List, Optional, Any
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QIcon, QFont

from ..core.task_manager import task_manager, Task, TaskStatus
from .progress_dialog import show_task_progress

# Configure logger
logger = logging.getLogger(__name__)


class TaskWidget(QFrame):
    """Widget for displaying a single task."""
    
    def __init__(self, task: Task, parent=None):
        """Initialize task widget.
        
        Args:
            task: Task to display
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.task_id = task.id
        
        # Setup UI
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Task name
        name_layout = QHBoxLayout()
        self.name_label = QLabel(f"<b>{task.name}</b>")
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        name_layout.addWidget(self.name_label)
        name_layout.addStretch(1)
        name_layout.addWidget(self.status_label)
        
        layout.addLayout(name_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(task.progress.percent))
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.message_label = QLabel(task.progress.message)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(task.can_cancel)
        self.cancel_button.clicked.connect(self._cancel_task)
        
        # Details button
        self.details_button = QPushButton("Details")
        self.details_button.clicked.connect(self._show_details)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.details_button)
        
        layout.addLayout(button_layout)
        
        # Update UI with current status
        self.update_from_task(task)
    
    def update_from_task(self, task: Optional[Task]) -> bool:
        """Update widget from task.
        
        Args:
            task: Task to update from, or None if not found
            
        Returns:
            True if updated successfully, False if task not found or invalid
        """
        if task is None:
            return False
        
        # Update progress
        self.progress_bar.setValue(int(task.progress.percent))
        
        # Update status
        self.message_label.setText(task.progress.message)
        
        # Update status text
        status_text = task.status.value
        self.status_label.setText(status_text)
        
        # Update cancel button
        self.cancel_button.setEnabled(task.can_cancel)
        
        # Update colors based on status
        if task.status == TaskStatus.COMPLETED:
            self.setStyleSheet("background-color: #e6f7e9;")
            self.cancel_button.setEnabled(False)
        elif task.status == TaskStatus.FAILED:
            self.setStyleSheet("background-color: #f9e6e6;")
            self.status_label.setText("Failed")
            self.cancel_button.setEnabled(False)
            if task.error:
                self.message_label.setText(f"Error: {task.error}")
        elif task.status == TaskStatus.CANCELLED:
            self.setStyleSheet("background-color: #f0f0f0;")
            self.cancel_button.setEnabled(False)
        
        return True
    
    def _cancel_task(self) -> None:
        """Cancel the task."""
        # Disable button
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        
        # Cancel task
        task_manager.cancel_task(self.task_id)
        
        # Update UI
        task = task_manager.get_task(self.task_id)
        if task:
            self.update_from_task(task)
    
    def _show_details(self) -> None:
        """Show task details in a dialog."""
        show_task_progress(self.task_id, parent=self.window())


class TaskMonitorWidget(QWidget):
    """Widget for monitoring tasks."""
    
    def __init__(self, parent=None):
        """Initialize task monitor widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 8, 8, 8)
        
        title_label = QLabel("<b>Task Monitor</b>")
        title_label.setFont(QFont("Arial", 10))
        
        self.task_count_label = QLabel("No active tasks")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.task_count_label)
        
        layout.addWidget(header)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create container for task widgets
        self.task_container = QWidget()
        self.task_layout = QVBoxLayout(self.task_container)
        self.task_layout.setContentsMargins(8, 8, 8, 8)
        self.task_layout.setSpacing(8)
        self.task_layout.addStretch(1)
        
        scroll_area.setWidget(self.task_container)
        layout.addWidget(scroll_area)
        
        # Add no tasks label
        self.no_tasks_label = QLabel("No active tasks")
        self.no_tasks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_tasks_label.setStyleSheet("color: gray; padding: 20px;")
        self.task_layout.insertWidget(0, self.no_tasks_label)
        
        # Track task widgets
        self.task_widgets: Dict[str, TaskWidget] = {}
        
        # Set up timer for updates
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(500)  # 500ms update interval
        self.update_timer.timeout.connect(self.update_tasks)
        self.update_timer.start()
    
    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        self.update_timer.start()
        self.update_tasks()
    
    def hideEvent(self, event) -> None:
        """Handle hide event."""
        super().hideEvent(event)
        self.update_timer.stop()
    
    def update_tasks(self) -> None:
        """Update task list."""
        # Get all tasks
        all_tasks = task_manager.get_all_tasks()
        
        # Create dict of task IDs to tasks
        task_dict = {task.id: task for task in all_tasks}
        
        # Keep track of active task IDs
        active_task_ids = set()
        
        # First update existing widgets
        for task_id, widget in list(self.task_widgets.items()):
            if task_id in task_dict:
                # Update widget
                widget.update_from_task(task_dict[task_id])
                
                # Keep track of active tasks
                if not task_dict[task_id].is_complete:
                    active_task_ids.add(task_id)
                elif task_dict[task_id].is_complete:
                    # Task completed during this update
                    # Keep completed task visible for a while
                    if task_dict[task_id].end_time is not None:
                        import time
                        age = time.time() - task_dict[task_id].end_time
                        if age < 5.0:  # Show completed tasks for 5 seconds
                            active_task_ids.add(task_id)
            else:
                # Task was removed, remove widget
                self.task_layout.removeWidget(widget)
                widget.deleteLater()
                del self.task_widgets[task_id]
        
        # Add widgets for new tasks
        for task_id, task in task_dict.items():
            if (task_id not in self.task_widgets) and (not task.is_complete or task_id in active_task_ids):
                # Create widget
                widget = TaskWidget(task)
                
                # Add to layout - insert at top (before stretch)
                self.task_layout.insertWidget(0, widget)
                
                # Keep track of widget
                self.task_widgets[task_id] = widget
                
                if not task.is_complete:
                    active_task_ids.add(task_id)
        
        # Remove widgets for completed tasks that are too old
        for task_id in list(self.task_widgets.keys()):
            if task_id not in active_task_ids:
                widget = self.task_widgets[task_id]
                self.task_layout.removeWidget(widget)
                widget.deleteLater()
                del self.task_widgets[task_id]
        
        # Update no tasks label
        self.no_tasks_label.setVisible(len(self.task_widgets) == 0)
        
        # Update task count
        active_count = len(active_task_ids)
        if active_count == 0:
            self.task_count_label.setText("No active tasks")
        else:
            self.task_count_label.setText(f"{active_count} active task{'s' if active_count != 1 else ''}")
    
    def get_task_count(self) -> int:
        """Get number of active tasks.
        
        Returns:
            Number of active tasks
        """
        return len([
            task_id for task_id, widget in self.task_widgets.items()
            if not task_manager.get_task(task_id).is_complete
        ]) 