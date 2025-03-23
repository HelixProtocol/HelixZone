"""
Shortcut editor dialog for customizing keyboard shortcuts.

This module provides a dialog for users to view and customize
keyboard shortcuts for various operations in the application.
"""

import logging
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDialogButtonBox, QMessageBox, QKeySequenceEdit, QLineEdit,
    QFrame, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeySequence, QAction

from ..gui.shortcuts import ShortcutCategory, ShortcutInfo, get_shortcut_manager

logger = logging.getLogger(__name__)


class ShortcutEditorDialog(QDialog):
    """Dialog for viewing and editing keyboard shortcuts."""
    
    def __init__(self, parent=None):
        """Initialize the shortcut editor dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(700, 500)
        
        # Get the shortcut manager
        self.shortcut_manager = get_shortcut_manager()
        
        # Store original shortcuts for resetting
        self.original_shortcuts = {
            shortcut.id: shortcut.custom_sequence 
            for shortcut in self.shortcut_manager.get_all_shortcuts()
        }
        
        # Map from category enum to tab index
        self.category_tabs = {}
        
        # Map from shortcut ID to table row in each category
        self.shortcut_rows: Dict[ShortcutCategory, Dict[str, int]] = {}
        
        # Set up layout
        layout = QVBoxLayout(self)
        
        # Create tab widget for categories
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs for each category
        self._create_category_tabs()
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search shortcuts...")
        self.search_edit.textChanged.connect(self._filter_shortcuts)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Reset
        )
        layout.addWidget(button_box)
        
        reset_button = button_box.button(QDialogButtonBox.StandardButton.Reset)
        reset_button.setText("Reset All")
        reset_button.clicked.connect(self._reset_all_shortcuts)
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
    def _create_category_tabs(self):
        """Create tabs for each shortcut category."""
        # Create a tab for each category
        for i, category in enumerate(ShortcutCategory):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Create table for shortcuts
            table = QTableWidget()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Command", "Description", "Shortcut"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            
            # Populate table with shortcuts for this category
            shortcuts = self.shortcut_manager.get_shortcuts_by_category(category)
            table.setRowCount(len(shortcuts))
            
            # Initialize shortcut row map for this category
            self.shortcut_rows[category] = {}
            
            for row, shortcut in enumerate(shortcuts):
                # Store row index for this shortcut ID
                self.shortcut_rows[category][shortcut.id] = row
                
                # Command name
                name_item = QTableWidgetItem(shortcut.title)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, 0, name_item)
                
                # Description
                desc_item = QTableWidgetItem(shortcut.description)
                desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, 1, desc_item)
                
                # Shortcut editor
                shortcut_cell = QWidget()
                cell_layout = QHBoxLayout(shortcut_cell)
                cell_layout.setContentsMargins(2, 2, 2, 2)
                
                # Key sequence editor
                key_edit = QKeySequenceEdit(QKeySequence(shortcut.sequence))
                key_edit.editingFinished.connect(
                    lambda key_edit=key_edit, shortcut_id=shortcut.id: 
                    self._update_shortcut(shortcut_id, key_edit.keySequence().toString())
                )
                cell_layout.addWidget(key_edit)
                
                # Reset button
                reset_button = QPushButton("Reset")
                reset_button.setFixedWidth(60)
                reset_button.clicked.connect(
                    lambda checked=False, shortcut_id=shortcut.id: 
                    self._reset_shortcut(shortcut_id)
                )
                cell_layout.addWidget(reset_button)
                
                table.setCellWidget(row, 2, shortcut_cell)
            
            tab_layout.addWidget(table)
            
            # Add tab
            self.tab_widget.addTab(tab, category.name.replace("_", " ").title())
            
            # Store tab index
            self.category_tabs[category] = i
    
    def _filter_shortcuts(self, text):
        """Filter shortcuts based on search text."""
        text = text.lower()
        
        # Show/hide rows in all tables based on search text
        for category, rows in self.shortcut_rows.items():
            tab_index = self.category_tabs[category]
            tab = self.tab_widget.widget(tab_index)
            table = tab.findChild(QTableWidget)
            
            # Get shortcuts for this category
            shortcuts = self.shortcut_manager.get_shortcuts_by_category(category)
            
            for shortcut in shortcuts:
                row = rows.get(shortcut.id, -1)
                if row >= 0:
                    # Check if shortcut matches search text
                    title_match = text in shortcut.title.lower()
                    desc_match = text in shortcut.description.lower()
                    sequence_match = text in shortcut.sequence.lower()
                    
                    # Show/hide row based on match
                    table.setRowHidden(row, not (title_match or desc_match or sequence_match))
    
    def _update_shortcut(self, shortcut_id, key_sequence):
        """Update a shortcut's key sequence.
        
        Args:
            shortcut_id: ID of the shortcut to update
            key_sequence: New key sequence as string
        """
        # Don't set empty shortcuts
        if not key_sequence:
            return
            
        # Check for conflicts
        shortcut = self.shortcut_manager.get_shortcut(shortcut_id)
        if not shortcut:
            return
            
        other = self.shortcut_manager._find_shortcut_by_sequence(key_sequence)
        if other and other.id != shortcut_id:
            # Show confirmation dialog for conflicting shortcut
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Shortcut Conflict")
            msg_box.setText(f"The shortcut '{key_sequence}' is already used by '{other.title}'.")
            msg_box.setInformativeText("Do you want to reassign it?")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)
            
            if msg_box.exec() == QMessageBox.StandardButton.No:
                # Revert to original shortcut
                tab_index = self.category_tabs[shortcut.category]
                tab = self.tab_widget.widget(tab_index)
                table = tab.findChild(QTableWidget)
                row = self.shortcut_rows[shortcut.category][shortcut_id]
                
                shortcut_cell = table.cellWidget(row, 2)
                key_edit = shortcut_cell.findChild(QKeySequenceEdit)
                key_edit.setKeySequence(QKeySequence(shortcut.sequence))
                return
        
        # Update the shortcut
        self.shortcut_manager.set_custom_shortcut(shortcut_id, key_sequence)
    
    def _reset_shortcut(self, shortcut_id):
        """Reset a shortcut to its default key sequence.
        
        Args:
            shortcut_id: ID of the shortcut to reset
        """
        # Reset the shortcut in the manager
        self.shortcut_manager.reset_shortcut(shortcut_id)
        
        # Update the UI
        shortcut = self.shortcut_manager.get_shortcut(shortcut_id)
        if shortcut:
            tab_index = self.category_tabs[shortcut.category]
            tab = self.tab_widget.widget(tab_index)
            table = tab.findChild(QTableWidget)
            row = self.shortcut_rows[shortcut.category][shortcut_id]
            
            shortcut_cell = table.cellWidget(row, 2)
            key_edit = shortcut_cell.findChild(QKeySequenceEdit)
            key_edit.setKeySequence(QKeySequence(shortcut.default_sequence))
    
    def _reset_all_shortcuts(self):
        """Reset all shortcuts to their default key sequences."""
        # Show confirmation dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Reset All Shortcuts")
        msg_box.setText("Are you sure you want to reset all shortcuts to their default values?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        
        if msg_box.exec() == QMessageBox.StandardButton.Yes:
            # Reset all shortcuts in the manager
            self.shortcut_manager.reset_all_shortcuts()
            
            # Update the UI
            for category, rows in self.shortcut_rows.items():
                tab_index = self.category_tabs[category]
                tab = self.tab_widget.widget(tab_index)
                table = tab.findChild(QTableWidget)
                
                for shortcut_id, row in rows.items():
                    shortcut = self.shortcut_manager.get_shortcut(shortcut_id)
                    if shortcut:
                        shortcut_cell = table.cellWidget(row, 2)
                        key_edit = shortcut_cell.findChild(QKeySequenceEdit)
                        key_edit.setKeySequence(QKeySequence(shortcut.default_sequence))
    
    def reject(self):
        """Handle dialog rejection (cancel button)."""
        # Restore original shortcuts
        for shortcut_id, sequence in self.original_shortcuts.items():
            shortcut = self.shortcut_manager.get_shortcut(shortcut_id)
            if shortcut:
                shortcut.custom_sequence = sequence
                
                # Update the action if it exists
                if shortcut.action:
                    shortcut.action.setShortcut(QKeySequence(shortcut.sequence))
        
        # Save custom shortcuts
        self.shortcut_manager._save_custom_shortcuts()
        
        # Emit change signal
        self.shortcut_manager.shortcuts_changed.emit()
        
        super().reject()
        
    def show_category(self, category: ShortcutCategory):
        """Show a specific category tab.
        
        Args:
            category: The category to show
        """
        tab_index = self.category_tabs.get(category, 0)
        self.tab_widget.setCurrentIndex(tab_index) 