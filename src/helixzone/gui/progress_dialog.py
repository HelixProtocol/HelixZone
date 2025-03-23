"""Progress dialog for displaying task progress to the user.

This module provides a customizable progress dialog that can be used to display
the progress of long-running tasks with support for task cancellation.
"""

import logging
from typing import Optional, Callable, List, Dict
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel, 
    QPushButton, QFrame, QSizePolicy, QScrollArea, QWidget
)
from PyQt6.QtGui import QCloseEvent

from ..core.task_manager import task_manager, Task, TaskStatus

# Configure logger
logger = logging.getLogger(__name__)


class ProgressDialog(QDialog):
    """Dialog for displaying task progress with cancellation support."""
    
    def __init__(
        self, 
        parent=None,
        title: str = "Task Progress",
        auto_close: bool = True,
        min_duration: int = 500,  # ms
        show_details: bool = True
    ):
        """Initialize progress dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            auto_close: Whether to auto-close on completion
            min_duration: Minimum duration to show dialog in ms
            show_details: Whether to show details by default
        """
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.resize(500, 200)
        self.setModal(True)
        
        # Settings
        self._auto_close = auto_close
        self._min_duration = min_duration
        self._start_time = None
        self._task_ids: List[str] = []
        self._cancelled = False
        
        # Create UI
        self._create_ui(show_details)
        
        # Create timer for updates
        self._timer = QTimer(self)
        self._timer.setInterval(100)  # 100ms updates
        self._timer.timeout.connect(self._update_progress)
        
        logger.debug("Progress dialog initialized")
    
    def _create_ui(self, show_details: bool) -> None:
        """Create the user interface.
        
        Args:
            show_details: Whether to show details by default
        """
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Task info
        self._task_label = QLabel("Preparing...")
        self._task_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._task_label.setWordWrap(True)
        layout.addWidget(self._task_label)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)
        
        # Status message
        self._status_label = QLabel("Starting...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        
        # Details area (initially hidden)
        self._details_frame = QFrame()
        self._details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._details_frame.setFrameShadow(QFrame.Shadow.Sunken)
        details_layout = QVBoxLayout(self._details_frame)
        details_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create scroll area for details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container for task progress widgets
        self._details_container = QWidget()
        self._details_layout = QVBoxLayout(self._details_container)
        self._details_layout.setContentsMargins(0, 0, 0, 0)
        self._details_layout.setSpacing(8)
        self._details_layout.addStretch(1)
        
        scroll_area.setWidget(self._details_container)
        details_layout.addWidget(scroll_area)
        
        # No tasks message
        self._no_tasks_label = QLabel("No active tasks")
        self._no_tasks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_tasks_label.setStyleSheet("color: gray;")
        self._details_layout.insertWidget(0, self._no_tasks_label)
        
        layout.addWidget(self._details_frame)
        self._details_frame.setVisible(show_details)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # Toggle details button
        self._details_button = QPushButton("Hide Details" if show_details else "Show Details")
        self._details_button.clicked.connect(self._toggle_details)
        
        # Cancel button
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._cancel_task)
        
        button_layout.addWidget(self._details_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self._cancel_button)
        
        layout.addLayout(button_layout)
        
        # Dictionary to track task progress widgets
        self._task_widgets: Dict[str, Dict] = {}
    
    def _toggle_details(self) -> None:
        """Toggle details visibility."""
        visible = not self._details_frame.isVisible()
        self._details_frame.setVisible(visible)
        self._details_button.setText("Hide Details" if visible else "Show Details")
        
        # Resize the dialog
        if visible:
            self.resize(self.width(), 400)
        else:
            self.resize(self.width(), 200)
    
    def _cancel_task(self) -> None:
        """Cancel the current task."""
        self._cancelled = True
        self._cancel_button.setEnabled(False)
        self._cancel_button.setText("Cancelling...")
        
        # Cancel all tasks
        for task_id in self._task_ids:
            task_manager.cancel_task(task_id)
        
        self._status_label.setText("Cancelling task, please wait...")
    
    def _update_task_widgets(self) -> None:
        """Update task widgets in the details area."""
        active_task_ids = set()
        
        # Get all tracked tasks
        for task_id in self._task_ids:
            task = task_manager.get_task(task_id)
            if task is None:
                continue
                
            active_task_ids.add(task_id)
            
            # Create or update widget for this task
            if task_id not in self._task_widgets:
                # Create new widget for this task
                task_frame = QFrame()
                task_frame.setFrameShape(QFrame.Shape.StyledPanel)
                task_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
                
                task_layout = QVBoxLayout(task_frame)
                task_layout.setContentsMargins(8, 8, 8, 8)
                task_layout.setSpacing(4)
                
                name_label = QLabel(f"<b>{task.name}</b>")
                status_label = QLabel("")
                status_label.setWordWrap(True)
                
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
                
                task_layout.addWidget(name_label)
                task_layout.addWidget(progress_bar)
                task_layout.addWidget(status_label)
                
                # Store widgets for later updates
                self._task_widgets[task_id] = {
                    'frame': task_frame,
                    'name_label': name_label,
                    'progress_bar': progress_bar,
                    'status_label': status_label
                }
                
                # Add to layout - insert before the stretch
                self._details_layout.insertWidget(
                    self._details_layout.count() - 1, task_frame
                )
            
            # Update the widget with current progress
            widgets = self._task_widgets[task_id]
            
            # Update progress
            widgets['progress_bar'].setValue(int(task.progress.percent))
            
            # Update status label
            status_text = f"{task.progress.message}"
            if task.status == TaskStatus.FAILED and task.error:
                status_text = f"Error: {task.error}"
            widgets['status_label'].setText(status_text)
            
            # Set frame style based on status
            if task.status == TaskStatus.COMPLETED:
                widgets['frame'].setStyleSheet("background-color: #e6f7e9;")
            elif task.status == TaskStatus.FAILED:
                widgets['frame'].setStyleSheet("background-color: #f9e6e6;")
            elif task.status == TaskStatus.CANCELLED:
                widgets['frame'].setStyleSheet("background-color: #f0f0f0;")
        
        # Remove widgets for tasks that are no longer in the list
        removed_task_ids = set(self._task_widgets.keys()) - active_task_ids
        for task_id in removed_task_ids:
            if task_id in self._task_widgets:
                # Remove widget
                widgets = self._task_widgets[task_id]
                self._details_layout.removeWidget(widgets['frame'])
                widgets['frame'].deleteLater()
                del self._task_widgets[task_id]
        
        # Show/hide no tasks label
        has_tasks = len(self._task_widgets) > 0
        self._no_tasks_label.setVisible(not has_tasks)
    
    def _update_progress(self) -> None:
        """Update progress display from task manager."""
        # Check if we have any tasks
        if not self._task_ids:
            return
            
        # For overall progress, use the primary task
        primary_task_id = self._task_ids[0]
        primary_task = task_manager.get_task(primary_task_id)
        
        if primary_task is None:
            logger.warning(f"Primary task {primary_task_id} not found")
            return
        
        # Update UI with primary task progress
        self._progress_bar.setValue(int(primary_task.progress.percent))
        self._status_label.setText(primary_task.progress.message)
        self._task_label.setText(primary_task.name)
        
        # Update detail widgets
        self._update_task_widgets()
        
        # Check if we should close
        all_complete = True
        for task_id in self._task_ids:
            task = task_manager.get_task(task_id)
            if task is not None and not task.is_complete:
                all_complete = False
                break
        
        if all_complete:
            logger.debug("All tasks completed")
            self._timer.stop()
            
            # Update cancel button
            self._cancel_button.setText("Close")
            self._cancel_button.setEnabled(True)
            self._cancel_button.clicked.disconnect()
            self._cancel_button.clicked.connect(self.accept)
            
            # Auto-close after min duration if enabled
            if self._auto_close and self._start_time is not None:
                import time
                elapsed = (time.time() - self._start_time) * 1000
                if elapsed >= self._min_duration:
                    self.accept()
                else:
                    # Schedule close after remaining time
                    delay = max(0, int(self._min_duration - elapsed))
                    QTimer.singleShot(delay, self.accept)
    
    def add_task(self, task_id: str) -> None:
        """Add a task to track.
        
        Args:
            task_id: ID of the task to track
        """
        if not self._timer.isActive():
            import time
            self._start_time = time.time()
            self._timer.start()
        
        if task_id not in self._task_ids:
            self._task_ids.append(task_id)
            logger.debug(f"Added task {task_id} to progress dialog")
            
            # Force immediate update
            self._update_progress()
    
    def track_task(self, task_id: str) -> None:
        """Track a task and show the dialog.
        
        Args:
            task_id: ID of the task to track
        """
        self.add_task(task_id)
        self.show()
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle close event.
        
        Args:
            event: Close event
        """
        # Stop the timer
        self._timer.stop()
        
        # Cancel tasks if requested
        if not self._cancelled and any(not task_manager.get_task(task_id).is_complete 
                                     for task_id in self._task_ids 
                                     if task_manager.get_task(task_id) is not None):
            self._cancel_task()
        
        # Accept the event
        event.accept()


def show_task_progress(
    task_id: str,
    parent=None,
    title: str = "Task Progress",
    auto_close: bool = True,
    min_duration: int = 500,
    show_details: bool = True
) -> ProgressDialog:
    """Show a progress dialog for a task.
    
    Args:
        task_id: ID of the task to track
        parent: Parent widget
        title: Dialog title
        auto_close: Whether to auto-close on completion
        min_duration: Minimum duration to show dialog in ms
        show_details: Whether to show details by default
        
    Returns:
        Progress dialog instance
    """
    dialog = ProgressDialog(
        parent=parent,
        title=title,
        auto_close=auto_close,
        min_duration=min_duration,
        show_details=show_details
    )
    dialog.track_task(task_id)
    return dialog


class TaskProgressTracker:
    """Utility class to track task progress and show a progress dialog."""
    
    def __init__(
        self,
        parent=None,
        auto_show: bool = True,
        auto_close: bool = True,
        min_duration: int = 500,
        dialog_delay: int = 300,  # Show dialog after this delay
        min_progress_step: float = 5.0,  # Min progress change to update
        title: str = "Task Progress"
    ):
        """Initialize progress tracker.
        
        Args:
            parent: Parent widget
            auto_show: Whether to automatically show dialog for long tasks
            auto_close: Whether to auto-close on completion
            min_duration: Minimum duration to show dialog in ms
            dialog_delay: Delay before showing dialog in ms
            min_progress_step: Minimum progress change to trigger update
            title: Dialog title
        """
        self._parent = parent
        self._auto_show = auto_show
        self._auto_close = auto_close
        self._min_duration = min_duration
        self._dialog_delay = dialog_delay
        self._min_progress_step = min_progress_step
        self._title = title
        
        self._dialog: Optional[ProgressDialog] = None
        self._timer: Optional[QTimer] = None
        self._tracked_tasks: List[str] = []
        self._last_progress: Dict[str, float] = {}
    
    def track_task(self, task_id: str, show_immediately: bool = False) -> None:
        """Track a task and optionally show progress dialog.
        
        Args:
            task_id: ID of the task to track
            show_immediately: Whether to show dialog immediately
        """
        # Store task
        if task_id not in self._tracked_tasks:
            self._tracked_tasks.append(task_id)
            self._last_progress[task_id] = 0.0
            
            # Set up progress callback
            task = task_manager.get_task(task_id)
            if task is not None:
                original_on_progress = task.on_progress
                
                def progress_callback(percent, message):
                    # Call original callback if any
                    if original_on_progress:
                        original_on_progress(percent, message)
                    
                    # Check if we should update (avoid too frequent updates)
                    if percent - self._last_progress.get(task_id, 0.0) >= self._min_progress_step:
                        self._last_progress[task_id] = percent
                        if self._dialog is not None:
                            # Ensure dialog is updated
                            pass  # Dialog updates automatically via timer
                
                task.on_progress = progress_callback
        
        # Show dialog if requested
        if show_immediately:
            self._show_dialog()
        elif self._auto_show and self._timer is None:
            # Schedule dialog to appear after delay
            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._show_dialog)
            self._timer.start(self._dialog_delay)
    
    def _show_dialog(self) -> None:
        """Create and show the progress dialog."""
        # Clear timer
        if self._timer is not None:
            if self._timer.isActive():
                self._timer.stop()
            self._timer = None
        
        # Create dialog if needed
        if self._dialog is None:
            self._dialog = ProgressDialog(
                parent=self._parent,
                title=self._title,
                auto_close=self._auto_close,
                min_duration=self._min_duration
            )
            
            # Add all tracked tasks
            for task_id in self._tracked_tasks:
                self._dialog.add_task(task_id)
            
            self._dialog.show()
    
    def is_tracking(self, task_id: str) -> bool:
        """Check if a task is being tracked.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if tracking
        """
        return task_id in self._tracked_tasks 