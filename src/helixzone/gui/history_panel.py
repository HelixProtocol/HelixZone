"""
History panel to display operation history with thumbnail support.

This module provides a panel for viewing and interacting with the history
of operations performed on the image, including undo/redo functionality.
"""

import logging
from typing import Optional, List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QScrollArea, QSplitter,
    QFrame, QSizePolicy, QAbstractItemView
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QImage

from ..core.commands import Command, CommandStack

logger = logging.getLogger(__name__)


class HistoryThumbnail(QWidget):
    """A thumbnail representation of an operation in the history."""
    
    def __init__(self, parent=None, title="Operation", thumbnail=None):
        """Initialize the history thumbnail.
        
        Args:
            parent: Parent widget
            title: Operation title
            thumbnail: Optional QPixmap thumbnail
        """
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(20)
        layout.addWidget(self.title_label)
        
        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumSize(60, 40)
        layout.addWidget(self.thumbnail_label)
        
        # Set thumbnail if provided
        if thumbnail:
            self.set_thumbnail(thumbnail)
        else:
            # Create default no-thumbnail image
            self._create_default_thumbnail()
            
    def set_thumbnail(self, thumbnail):
        """Set the thumbnail image.
        
        Args:
            thumbnail: QPixmap or QImage
        """
        if isinstance(thumbnail, QImage):
            pixmap = QPixmap.fromImage(thumbnail.scaled(
                60, 40, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            pixmap = thumbnail.scaled(
                60, 40, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
        
        self.thumbnail_label.setPixmap(pixmap)
        
    def _create_default_thumbnail(self):
        """Create a default thumbnail for operations without images."""
        pixmap = QPixmap(60, 40)
        pixmap.fill(QColor(240, 240, 240))
        
        # Add a simple icon or pattern
        painter = QPainter(pixmap)
        painter.setPen(QColor(180, 180, 180))
        painter.drawRect(0, 0, 59, 39)
        painter.drawLine(0, 0, 59, 39)
        painter.drawLine(0, 39, 59, 0)
        painter.end()
        
        self.thumbnail_label.setPixmap(pixmap)


class HistoryWidget(QWidget):
    """Widget for displaying and interacting with operation history."""
    
    history_clicked = pyqtSignal(int)  # Emitted when a history item is clicked
    
    def __init__(self, command_stack, parent=None):
        """Initialize the history widget.
        
        Args:
            command_stack: The command stack to visualize
            parent: Parent widget
        """
        super().__init__(parent)
        self.command_stack = command_stack
        self.command_stack.stack_changed.connect(self.update_history)
        
        # Store thumbnails keyed by command ID
        self.thumbnails: Dict[int, QPixmap] = {}
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # History list
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history_list.currentRowChanged.connect(self._on_history_selected)
        layout.addWidget(self.history_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        button_layout.addWidget(self.undo_button)
        
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)
        button_layout.addWidget(self.redo_button)
        
        layout.addLayout(button_layout)
        
        # Update history on startup
        QTimer.singleShot(0, self.update_history)
    
    def update_history(self):
        """Update the history list from the command stack."""
        self.history_list.clear()
        
        # Get commands and index from stack
        commands = self.command_stack.get_commands()
        current_index = self.command_stack.get_index()
        
        # Add each command to the list
        for i, command in enumerate(commands):
            # Create item and thumbnail widget
            item = QListWidgetItem()
            thumbnail = self.thumbnails.get(id(command))
            thumbnail_widget = HistoryThumbnail(title=command.text, thumbnail=thumbnail)
            
            # Set item properties
            item.setSizeHint(thumbnail_widget.sizeHint())
            
            # Add to list
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, thumbnail_widget)
            
        # Highlight current position
        if commands:
            self.history_list.setCurrentRow(current_index)
            
        # Update button states
        self.undo_button.setEnabled(self.command_stack.can_undo())
        self.redo_button.setEnabled(self.command_stack.can_redo())
    
    def set_thumbnail(self, command, thumbnail):
        """Set a thumbnail for a specific command.
        
        Args:
            command: The command to associate with the thumbnail
            thumbnail: The thumbnail image (QPixmap or QImage)
        """
        self.thumbnails[id(command)] = thumbnail
        self.update_history()
    
    def _on_history_selected(self, index):
        """Handle history item selection."""
        if index >= 0:
            self.history_clicked.emit(index)
    
    def _undo(self):
        """Undo the last command."""
        self.command_stack.undo()
    
    def _redo(self):
        """Redo the next command."""
        self.command_stack.redo()


class HistoryPanel(QWidget):
    """A dockable panel for displaying operation history."""
    
    def __init__(self, command_stack, parent=None):
        """Initialize the history panel.
        
        Args:
            command_stack: The command stack to visualize
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("History")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(header)
        
        # History widget
        self.history_widget = HistoryWidget(command_stack)
        layout.addWidget(self.history_widget)
        
        # Default size
        self.setMinimumWidth(150)
        
    def jump_to_state(self, index):
        """Jump to a specific state in the history.
        
        Args:
            index: The index to jump to
        """
        current = self.history_widget.command_stack.get_index()
        
        if index < current:
            # Undo to reach the desired state
            for _ in range(current - index):
                self.history_widget.command_stack.undo()
        elif index > current:
            # Redo to reach the desired state
            for _ in range(index - current):
                self.history_widget.command_stack.redo()
                
    def add_thumbnail(self, command, thumbnail):
        """Add a thumbnail for a command.
        
        Args:
            command: The command to associate with the thumbnail
            thumbnail: The thumbnail image
        """
        self.history_widget.set_thumbnail(command, thumbnail) 