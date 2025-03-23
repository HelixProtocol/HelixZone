"""Error dialog for HelixZone."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QCheckBox, QDialogButtonBox,
    QMessageBox, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap
import traceback
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

class ErrorDialog(QDialog):
    """Custom error dialog with detailed information."""
    
    def __init__(
        self, 
        parent=None, 
        title: str = "Error", 
        message: str = "", 
        details: str = "",
        error_type: str = "Error"
    ):
        """Initialize error dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            message: Main error message
            details: Detailed error information
            error_type: Type of error (Error, Warning, Info)
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.error_type = error_type
        
        # Get icon based on error type
        if error_type.lower() == "warning":
            self.icon = QStyle.StandardPixmap.SP_MessageBoxWarning
        elif error_type.lower() == "info":
            self.icon = QStyle.StandardPixmap.SP_MessageBoxInformation
        else:
            self.icon = QStyle.StandardPixmap.SP_MessageBoxCritical
            
        # Create layout
        layout = QVBoxLayout()
        
        # Create message layout
        message_layout = QHBoxLayout()
        
        # Add icon
        icon_label = QLabel()
        pixmap = self.style().standardIcon(self.icon).pixmap(QSize(32, 32))
        icon_label.setPixmap(pixmap)
        message_layout.addWidget(icon_label, 0)
        
        # Add message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_layout.addWidget(message_label, 1)
        
        layout.addLayout(message_layout)
        
        # Add details if provided
        if details:
            details_edit = QTextEdit()
            details_edit.setReadOnly(True)
            details_edit.setText(details)
            details_edit.setFixedHeight(200)
            
            # Add "Show Details" checkbox
            details_check = QCheckBox("Show Details")
            details_check.toggled.connect(lambda checked: details_edit.setVisible(checked))
            layout.addWidget(details_check)
            
            # Initially hide details
            details_edit.setVisible(False)
            layout.addWidget(details_edit)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    @classmethod
    def show_error(
        cls, 
        parent=None, 
        title: str = "Error", 
        message: str = "", 
        details: str = "",
        error_type: str = "Error"
    ) -> int:
        """Show error dialog and return result.
        
        Args:
            parent: Parent widget
            title: Dialog title
            message: Main error message
            details: Detailed error information
            error_type: Type of error (Error, Warning, Info)
            
        Returns:
            Dialog result code
        """
        dialog = cls(parent, title, message, details, error_type)
        return dialog.exec()
        
    @classmethod
    def from_exception(
        cls, 
        parent=None, 
        title: str = "Error", 
        message: str = "An error occurred", 
        exception: Optional[Exception] = None
    ) -> int:
        """Create error dialog from exception.
        
        Args:
            parent: Parent widget
            title: Dialog title
            message: Main error message
            exception: Exception object
            
        Returns:
            Dialog result code
        """
        if exception:
            details = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        else:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type:
                details = ''.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                ))
            else:
                details = ""
                
        return cls.show_error(parent, title, message, details, "Error")


class ErrorNotification:
    """Singleton for managing error notifications."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ErrorNotification()
        return cls._instance
    
    def __init__(self):
        """Initialize error notification manager."""
        self.errors = {}
        self.error_count = 0
        self.parent = None
        self.logger = logging.getLogger('helixzone.errors')
        
    def set_parent(self, parent):
        """Set parent widget for dialogs."""
        self.parent = parent
        
    def show_error(
        self, 
        title: str, 
        message: str, 
        details: str = "", 
        log: bool = True,
        dialog: bool = True
    ) -> int:
        """Show error notification.
        
        Args:
            title: Error title
            message: Error message
            details: Error details
            log: Whether to log the error
            dialog: Whether to show dialog
            
        Returns:
            Dialog result code if shown, otherwise 0
        """
        # Log error if requested
        if log:
            self.logger.error(f"{title}: {message}\n{details}")
            
        # Track error
        error_key = f"{title}_{message}"
        if error_key in self.errors:
            self.errors[error_key]['count'] += 1
            self.errors[error_key]['last_time'] = datetime.now()
        else:
            self.errors[error_key] = {
                'count': 1,
                'first_time': datetime.now(),
                'last_time': datetime.now(),
                'title': title,
                'message': message
            }
        
        self.error_count += 1
        
        # Show dialog if requested
        if dialog and self.parent:
            return ErrorDialog.show_error(
                self.parent, 
                title, 
                message, 
                details, 
                "Error"
            )
        return 0
        
    def show_warning(
        self, 
        title: str, 
        message: str, 
        details: str = "", 
        log: bool = True,
        dialog: bool = True
    ) -> int:
        """Show warning notification.
        
        Args:
            title: Warning title
            message: Warning message
            details: Warning details
            log: Whether to log the warning
            dialog: Whether to show dialog
            
        Returns:
            Dialog result code if shown, otherwise 0
        """
        # Log warning if requested
        if log:
            self.logger.warning(f"{title}: {message}\n{details}")
            
        # Show dialog if requested
        if dialog and self.parent:
            return ErrorDialog.show_error(
                self.parent, 
                title, 
                message, 
                details, 
                "Warning"
            )
        return 0
        
    def show_from_exception(
        self,
        title: str,
        message: str,
        exception: Optional[Exception] = None,
        log: bool = True,
        dialog: bool = True
    ) -> int:
        """Show error notification from exception.
        
        Args:
            title: Error title
            message: Error message
            exception: Exception object
            log: Whether to log the error
            dialog: Whether to show dialog
            
        Returns:
            Dialog result code if shown, otherwise 0
        """
        if exception:
            details = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        else:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type:
                details = ''.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                ))
            else:
                details = ""
                
        # Log error if requested
        if log:
            self.logger.error(f"{title}: {message}\n{details}", exc_info=True)
            
        # Show dialog if requested
        if dialog and self.parent:
            return ErrorDialog.from_exception(
                self.parent, 
                title, 
                message, 
                exception
            )
        return 0
        
    def get_error_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of errors."""
        return self.errors
        
    def get_error_count(self) -> int:
        """Get number of errors."""
        return self.error_count
        
    def clear_errors(self) -> None:
        """Clear error tracking."""
        self.errors = {}
        self.error_count = 0
        
# Create global instance for easy access
error_notification = ErrorNotification.get_instance() 