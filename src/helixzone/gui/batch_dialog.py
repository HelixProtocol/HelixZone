"""
Batch processing dialog for managing batch operations.

This module provides a dialog for setting up and monitoring
batch processing operations on multiple files.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QTabWidget, QWidget,
    QProgressBar, QFileDialog, QComboBox, QSpinBox,
    QCheckBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QGroupBox, QSizePolicy,
    QSplitter, QMessageBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QPixmap

from ..core.batch_processor import get_batch_processor, BatchItemStatus, BatchItem
from ..core.file_manager import get_file_manager, FileFormat

logger = logging.getLogger(__name__)


class BatchSetupDialog(QDialog):
    """Dialog for setting up a batch processing operation."""
    
    def __init__(self, parent=None):
        """Initialize the batch setup dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Setup")
        self.setMinimumSize(800, 600)
        
        # Initialize batch processor and file manager
        self.batch_processor = get_batch_processor()
        self.file_manager = get_file_manager()
        
        # Collect batch input data
        self.input_files = []
        self.output_directory = ""
        self.operation_id = ""
        self.parameters = {}
        
        # Create layout
        self.main_layout = QVBoxLayout(self)
        
        # Create tabs for setup steps
        self.setup_tabs = QTabWidget()
        self.main_layout.addWidget(self.setup_tabs)
        
        # Create tabs
        self.create_files_tab()
        self.create_operation_tab()
        self.create_parameters_tab()
        self.create_output_tab()
        self.create_summary_tab()
        
        # Create button box
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.go_to_prev_tab)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_to_next_tab)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.accept)
        self.start_button.setEnabled(False)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.start_button)
        
        self.main_layout.addLayout(button_layout)
        
        # Connect tab change signal
        self.setup_tabs.currentChanged.connect(self.on_tab_changed)
    
    def create_files_tab(self):
        """Create the tab for selecting input files."""
        files_tab = QWidget()
        layout = QVBoxLayout(files_tab)
        
        # Instructions
        label = QLabel("Select the files to process:")
        layout.addWidget(label)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.file_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_button = QPushButton("Add Files...")
        add_button.clicked.connect(self.add_files)
        
        add_folder_button = QPushButton("Add Folder...")
        add_folder_button.clicked.connect(self.add_folder)
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_files)
        
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_files)
        
        button_layout.addWidget(add_button)
        button_layout.addWidget(add_folder_button)
        button_layout.addWidget(remove_button)
        button_layout.addWidget(clear_button)
        
        layout.addLayout(button_layout)
        
        self.setup_tabs.addTab(files_tab, "1. Select Files")
    
    def create_operation_tab(self):
        """Create the tab for selecting the operation."""
        operation_tab = QWidget()
        layout = QVBoxLayout(operation_tab)
        
        # Instructions
        label = QLabel("Select the operation to perform:")
        layout.addWidget(label)
        
        # Operation selection
        self.operation_combo = QComboBox()
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)
        
        # Add available operations
        operations = self.batch_processor.get_operations()
        for op_id, operation in operations.items():
            self.operation_combo.addItem(operation.name, op_id)
            
        layout.addWidget(self.operation_combo)
        
        # Operation description
        self.operation_description = QLabel()
        self.operation_description.setWordWrap(True)
        self.operation_description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.operation_description)
        
        # Set initial operation description
        self.on_operation_changed(0)
        
        self.setup_tabs.addTab(operation_tab, "2. Select Operation")
    
    def create_parameters_tab(self):
        """Create the tab for setting operation parameters."""
        params_tab = QWidget()
        layout = QVBoxLayout(params_tab)
        
        # Instructions
        label = QLabel("Set operation parameters:")
        layout.addWidget(label)
        
        # Parameters form - will be populated dynamically
        self.params_form_layout = QFormLayout()
        
        # Create a scrollable area for the parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        params_widget = QWidget()
        params_widget.setLayout(self.params_form_layout)
        
        scroll_area.setWidget(params_widget)
        layout.addWidget(scroll_area)
        
        self.setup_tabs.addTab(params_tab, "3. Set Parameters")
    
    def create_output_tab(self):
        """Create the tab for selecting output options."""
        output_tab = QWidget()
        layout = QVBoxLayout(output_tab)
        
        # Instructions
        label = QLabel("Set output options:")
        layout.addWidget(label)
        
        # Output directory
        form_layout = QFormLayout()
        
        self.output_dir_label = QLabel("No output directory selected")
        self.output_dir_label.setWordWrap(True)
        
        output_dir_button = QPushButton("Select Output Directory...")
        output_dir_button.clicked.connect(self.select_output_directory)
        
        dir_layout = QVBoxLayout()
        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(output_dir_button)
        
        form_layout.addRow("Output Directory:", dir_layout)
        
        # File naming options
        self.naming_combo = QComboBox()
        self.naming_combo.addItem("Keep original names", "original")
        self.naming_combo.addItem("Numbered sequence", "sequence")
        self.naming_combo.addItem("Original name with suffix", "suffix")
        
        form_layout.addRow("File Naming:", self.naming_combo)
        
        # Sequence options (only visible when "Numbered sequence" is selected)
        self.sequence_prefix = QComboBox()
        self.sequence_prefix.setEditable(True)
        self.sequence_prefix.addItem("image_")
        self.sequence_prefix.addItem("output_")
        self.sequence_prefix.addItem("processed_")
        
        form_layout.addRow("Sequence Prefix:", self.sequence_prefix)
        
        # Suffix (only visible when "Original name with suffix" is selected)
        self.suffix_edit = QComboBox()
        self.suffix_edit.setEditable(True)
        self.suffix_edit.addItem("_processed")
        self.suffix_edit.addItem("_edited")
        self.suffix_edit.addItem("_output")
        
        form_layout.addRow("Suffix:", self.suffix_edit)
        
        # Output format options (if applicable)
        self.format_combo = QComboBox()
        for format_enum in FileFormat:
            if format_enum != FileFormat.UNKNOWN:
                self.format_combo.addItem(format_enum.name, format_enum)
        
        form_layout.addRow("Output Format:", self.format_combo)
        
        # Add form to layout
        layout.addLayout(form_layout)
        
        # Extra options
        options_group = QGroupBox("Additional Options")
        options_layout = QVBoxLayout(options_group)
        
        self.overwrite_check = QCheckBox("Overwrite existing files")
        options_layout.addWidget(self.overwrite_check)
        
        self.skip_errors_check = QCheckBox("Continue batch on errors")
        self.skip_errors_check.setChecked(True)
        options_layout.addWidget(self.skip_errors_check)
        
        layout.addWidget(options_group)
        layout.addStretch()
        
        self.setup_tabs.addTab(output_tab, "4. Output Options")
    
    def create_summary_tab(self):
        """Create the tab for showing a summary of the batch operation."""
        summary_tab = QWidget()
        layout = QVBoxLayout(summary_tab)
        
        # Title
        title_label = QLabel("Batch Processing Summary")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Summary info
        self.summary_layout = QFormLayout()
        
        self.summary_files = QLabel("0 files selected")
        self.summary_operation = QLabel("No operation selected")
        self.summary_output = QLabel("No output directory selected")
        
        self.summary_layout.addRow("Files to process:", self.summary_files)
        self.summary_layout.addRow("Operation:", self.summary_operation)
        self.summary_layout.addRow("Output directory:", self.summary_output)
        
        # Parameters will be added dynamically
        self.summary_params_group = QGroupBox("Parameters")
        self.summary_params_layout = QFormLayout(self.summary_params_group)
        
        layout.addLayout(self.summary_layout)
        layout.addWidget(self.summary_params_group)
        
        # Add a notice
        notice = QLabel(
            "Click 'Start Processing' to begin the batch operation. "
            "This will process all selected files using the specified parameters."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("font-style: italic;")
        layout.addWidget(notice)
        
        layout.addStretch()
        
        self.setup_tabs.addTab(summary_tab, "5. Summary")
    
    def on_tab_changed(self, index):
        """Handle tab changes.
        
        Args:
            index: New tab index
        """
        # Update button states
        self.prev_button.setEnabled(index > 0)
        self.next_button.setVisible(index < self.setup_tabs.count() - 1)
        self.start_button.setVisible(index == self.setup_tabs.count() - 1)
        
        # If we're on the summary tab, update the summary
        if index == 4:  # Summary tab
            self.update_summary()
            
            # Enable the start button if we have files and an output directory
            self.start_button.setEnabled(
                len(self.input_files) > 0 and 
                self.operation_id and 
                self.output_directory
            )
        
        # If we're on the parameters tab, update the parameters form
        if index == 2:  # Parameters tab
            self.update_parameters_form()
    
    def add_files(self):
        """Add files to the batch."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All Files (*)"
        )
        
        if files:
            for file_path in files:
                if file_path not in self.input_files:
                    self.input_files.append(file_path)
                    item = QListWidgetItem(os.path.basename(file_path))
                    item.setData(Qt.ItemDataRole.UserRole, file_path)
                    self.file_list.addItem(item)
    
    def add_folder(self):
        """Add all images from a folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        
        if folder:
            # Get all image files in the folder
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp']
            
            for root, _, files in os.walk(folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        file_path = os.path.join(root, file)
                        if file_path not in self.input_files:
                            self.input_files.append(file_path)
                            item = QListWidgetItem(os.path.basename(file_path))
                            item.setData(Qt.ItemDataRole.UserRole, file_path)
                            self.file_list.addItem(item)
    
    def remove_files(self):
        """Remove selected files from the batch."""
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path in self.input_files:
                self.input_files.remove(file_path)
            
            # Remove from list widget
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
    
    def clear_files(self):
        """Clear all files from the batch."""
        self.input_files = []
        self.file_list.clear()
    
    def on_operation_changed(self, index):
        """Handle operation selection change.
        
        Args:
            index: Index of the selected operation
        """
        if index >= 0:
            self.operation_id = self.operation_combo.currentData()
            operation = self.batch_processor.get_operation(self.operation_id)
            
            if operation:
                self.operation_description.setText(operation.description)
    
    def update_parameters_form(self):
        """Update the parameters form based on the selected operation."""
        # Clear existing parameters
        while self.params_form_layout.rowCount() > 0:
            self.params_form_layout.removeRow(0)
        
        # Get the selected operation
        operation = self.batch_processor.get_operation(self.operation_id)
        if not operation:
            return
        
        # Get the parameter schema
        schema = operation.get_parameters_schema()
        
        # Create form fields for each parameter
        for param_name, param_config in schema.items():
            param_type = param_config.get('type', 'string')
            description = param_config.get('description', '')
            default = param_config.get('default', None)
            
            # Create label with tooltip
            label = QLabel(f"{param_name}:")
            label.setToolTip(description)
            
            # Create the appropriate input widget
            widget = None
            
            if param_type == 'string':
                widget = QComboBox()
                widget.setEditable(True)
                
                # If enum values are provided, add them
                if 'enum' in param_config:
                    for value in param_config['enum']:
                        widget.addItem(value)
                
                # Set default if provided
                if default is not None:
                    widget.setCurrentText(str(default))
            
            elif param_type == 'integer':
                widget = QSpinBox()
                
                # Set range if provided
                if 'minimum' in param_config:
                    widget.setMinimum(param_config['minimum'])
                if 'maximum' in param_config:
                    widget.setMaximum(param_config['maximum'])
                
                # Set default if provided
                if default is not None:
                    widget.setValue(default)
            
            elif param_type == 'boolean':
                widget = QCheckBox()
                
                # Set default if provided
                if default is not None:
                    widget.setChecked(default)
            
            # Add the widget to the form
            if widget:
                self.params_form_layout.addRow(label, widget)
                
                # Store the widget for later retrieval
                widget.setObjectName(f"param_{param_name}")
    
    def get_parameters(self):
        """Get parameters from the form.
        
        Returns:
            Dictionary of parameter values
        """
        parameters = {}
        
        # Get the selected operation
        operation = self.batch_processor.get_operation(self.operation_id)
        if not operation:
            return parameters
        
        # Get the parameter schema
        schema = operation.get_parameters_schema()
        
        # Get values from form widgets
        for param_name, param_config in schema.items():
            param_type = param_config.get('type', 'string')
            widget_name = f"param_{param_name}"
            
            # Find the widget by name
            widget = self.findChild(QWidget, widget_name)
            if not widget:
                continue
            
            # Get value based on widget type
            if param_type == 'string':
                if isinstance(widget, QComboBox):
                    parameters[param_name] = widget.currentText()
            
            elif param_type == 'integer':
                if isinstance(widget, QSpinBox):
                    parameters[param_name] = widget.value()
            
            elif param_type == 'boolean':
                if isinstance(widget, QCheckBox):
                    parameters[param_name] = widget.isChecked()
            
            elif param_type == 'object':
                # For nested objects, use default value
                parameters[param_name] = param_config.get('default', {})
        
        return parameters
    
    def select_output_directory(self):
        """Select the output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if directory:
            self.output_directory = directory
            self.output_dir_label.setText(directory)
    
    def update_summary(self):
        """Update the summary tab with current batch settings."""
        # Update file count
        self.summary_files.setText(f"{len(self.input_files)} files selected")
        
        # Update operation
        operation = self.batch_processor.get_operation(self.operation_id)
        if operation:
            self.summary_operation.setText(f"{operation.name} ({self.operation_id})")
        else:
            self.summary_operation.setText("No operation selected")
        
        # Update output directory
        if self.output_directory:
            self.summary_output.setText(self.output_directory)
        else:
            self.summary_output.setText("No output directory selected")
        
        # Update parameters summary
        # First, clear existing parameters
        while self.summary_params_layout.rowCount() > 0:
            self.summary_params_layout.removeRow(0)
        
        # Get parameters
        parameters = self.get_parameters()
        
        # Add parameters to summary
        for param_name, param_value in parameters.items():
            value_str = str(param_value)
            if isinstance(param_value, dict) and param_value:
                value_str = f"{{...}} ({len(param_value)} items)"
            elif isinstance(param_value, bool):
                value_str = "Yes" if param_value else "No"
            
            self.summary_params_layout.addRow(f"{param_name}:", QLabel(value_str))
    
    def go_to_next_tab(self):
        """Go to the next tab."""
        current = self.setup_tabs.currentIndex()
        if current < self.setup_tabs.count() - 1:
            self.setup_tabs.setCurrentIndex(current + 1)
    
    def go_to_prev_tab(self):
        """Go to the previous tab."""
        current_index = self.setup_tabs.currentIndex()
        if current_index > 0:
            self.setup_tabs.setCurrentIndex(current_index - 1)
    
    def get_batch_items(self):
        """Create batch items from the selected files and settings.
        
        Returns:
            List of batch item dictionaries
        """
        if not self.input_files or not self.operation_id or not self.output_directory:
            return []
        
        # Get parameters
        self.parameters = self.get_parameters()
        
        # Create batch items
        batch_items = []
        
        # Get output format if specified
        output_format = self.format_combo.currentData()
        
        # File naming pattern
        naming_pattern = self.naming_combo.currentData()
        
        for i, input_path in enumerate(self.input_files):
            # Determine output path based on naming pattern
            output_filename = ""
            
            if naming_pattern == "original":
                # Keep original filename but potentially change extension
                output_filename = os.path.basename(input_path)
                if output_format != FileFormat.UNKNOWN:
                    # Change extension
                    base_name = os.path.splitext(output_filename)[0]
                    ext = FileFormat.get_extension(output_format)
                    output_filename = f"{base_name}.{ext}"
            
            elif naming_pattern == "sequence":
                # Numbered sequence
                prefix = self.sequence_prefix.currentText()
                if output_format != FileFormat.UNKNOWN:
                    ext = FileFormat.get_extension(output_format)
                else:
                    # Keep original extension
                    ext = os.path.splitext(input_path)[1][1:]  # Remove the dot
                
                output_filename = f"{prefix}{i+1:04d}.{ext}"
            
            elif naming_pattern == "suffix":
                # Original name with suffix
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                suffix = self.suffix_edit.currentText()
                
                if output_format != FileFormat.UNKNOWN:
                    ext = FileFormat.get_extension(output_format)
                else:
                    # Keep original extension
                    ext = os.path.splitext(input_path)[1][1:]  # Remove the dot
                
                output_filename = f"{base_name}{suffix}.{ext}"
            
            # Combine with output directory
            output_path = os.path.join(self.output_directory, output_filename)
            
            # Check if output file exists and we're not overwriting
            if os.path.exists(output_path) and not self.overwrite_check.isChecked():
                # Add a unique suffix to avoid overwriting
                base_name = os.path.splitext(output_filename)[0]
                ext = os.path.splitext(output_filename)[1]
                output_filename = f"{base_name}_copy{ext}"
                output_path = os.path.join(self.output_directory, output_filename)
            
            # Create batch item
            item = {
                'name': os.path.basename(input_path),
                'input_path': input_path,
                'output_path': output_path,
                'parameters': self.parameters.copy()
            }
            
            # For format conversion, add the format to parameters
            if self.operation_id == 'convert_format' and output_format != FileFormat.UNKNOWN:
                item['parameters']['format'] = output_format.name
            
            batch_items.append(item)
        
        return batch_items


class BatchProgressDialog(QDialog):
    """Dialog for monitoring batch processing progress."""
    
    def __init__(self, batch_id, parent=None):
        """Initialize the batch progress dialog.
        
        Args:
            batch_id: ID of the batch to monitor
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(800, 600)
        
        # Initialize batch processor
        self.batch_processor = get_batch_processor()
        self.batch_id = batch_id
        
        # Create layout
        self.main_layout = QVBoxLayout(self)
        
        # Create progress section
        progress_layout = QVBoxLayout()
        
        self.title_label = QLabel("Processing Batch")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        progress_layout.addWidget(self.title_label)
        
        self.status_label = QLabel("Starting batch processing...")
        progress_layout.addWidget(self.status_label)
        
        # Overall progress
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.overall_progress = QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        
        # Stats section
        stats_form = QFormLayout()
        
        self.stats_total = QLabel("0")
        self.stats_completed = QLabel("0")
        self.stats_failed = QLabel("0")
        self.stats_remaining = QLabel("0")
        
        stats_form.addRow("Total Files:", self.stats_total)
        stats_form.addRow("Completed:", self.stats_completed)
        stats_form.addRow("Failed:", self.stats_failed)
        stats_form.addRow("Remaining:", self.stats_remaining)
        
        progress_layout.addLayout(stats_form)
        
        self.main_layout.addLayout(progress_layout)
        
        # Create item list
        self.main_layout.addWidget(QLabel("Files:"))
        
        # Create table for items
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(4)
        self.items_table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Message"])
        self.items_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.items_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.items_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.items_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        self.main_layout.addWidget(self.items_table)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_batch)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        self.main_layout.addLayout(button_layout)
        
        # Connect to batch processor signals
        self.batch_processor.batch_completed.connect(self.on_batch_completed)
        self.batch_processor.batch_failed.connect(self.on_batch_failed)
        self.batch_processor.batch_cancelled.connect(self.on_batch_cancelled)
        self.batch_processor.item_started.connect(self.on_item_started)
        self.batch_processor.item_progress.connect(self.on_item_progress)
        self.batch_processor.item_completed.connect(self.on_item_completed)
        self.batch_processor.item_failed.connect(self.on_item_failed)
        
        # Set up timer to update status periodically
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(500)  # Update every 500ms
        
        # Initialize status and items table
        self.update_status()
    
    def update_status(self):
        """Update the dialog with current batch status."""
        batch_status = self.batch_processor.get_batch_status(self.batch_id)
        if not batch_status:
            return
        
        # Update progress bar
        self.overall_progress.setValue(int(batch_status['progress']))
        
        # Update stats
        self.stats_total.setText(str(batch_status['total_items']))
        self.stats_completed.setText(str(batch_status['completed_items']))
        self.stats_failed.setText(str(batch_status['failed_items']))
        self.stats_remaining.setText(
            str(batch_status['total_items'] - batch_status['completed_items'] - batch_status['failed_items'])
        )
        
        # Update status label
        if batch_status['is_complete']:
            if batch_status['failed_items'] > 0:
                self.status_label.setText(
                    f"Batch completed with {batch_status['failed_items']} errors. "
                    f"{batch_status['completed_items']} files processed successfully."
                )
            else:
                self.status_label.setText("Batch completed successfully!")
        else:
            self.status_label.setText(
                f"Processing... {int(batch_status['progress'])}% complete"
            )
        
        # Update items table if needed
        if self.items_table.rowCount() != len(batch_status['items']):
            # Resize the table
            self.items_table.setRowCount(len(batch_status['items']))
            
            # Add items
            for i, item in enumerate(batch_status['items']):
                # File name
                file_item = QTableWidgetItem(os.path.basename(item.input_path))
                self.items_table.setItem(i, 0, file_item)
                
                # Status
                status_item = QTableWidgetItem(item.status.name)
                self.items_table.setItem(i, 1, status_item)
                
                # Progress bar
                progress_bar = QProgressBar()
                progress_bar.setValue(int(item.progress))
                progress_bar.setTextVisible(True)
                self.items_table.setCellWidget(i, 2, progress_bar)
                
                # Message
                message_item = QTableWidgetItem(item.message if item.message else "")
                self.items_table.setItem(i, 3, message_item)
        else:
            # Just update existing items
            for i, item in enumerate(batch_status['items']):
                # Status
                status_item = QTableWidgetItem(item.status.name)
                self.items_table.setItem(i, 1, status_item)
                
                # Progress bar
                progress_bar = self.items_table.cellWidget(i, 2)
                if progress_bar and isinstance(progress_bar, QProgressBar):
                    progress_bar.setValue(int(item.progress))
                
                # Message
                message_item = QTableWidgetItem(item.message if item.message else "")
                self.items_table.setItem(i, 3, message_item)
        
        # Update button states
        if batch_status['is_complete']:
            self.cancel_button.setEnabled(False)
            self.close_button.setEnabled(True)
    
    def on_batch_completed(self, batch_id):
        """Handle batch completion.
        
        Args:
            batch_id: ID of the completed batch
        """
        if batch_id != self.batch_id:
            return
        
        self.title_label.setText("Batch Processing Completed")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.status_label.setText("All files processed successfully!")
    
    def on_batch_failed(self, batch_id, error_message):
        """Handle batch failure.
        
        Args:
            batch_id: ID of the failed batch
            error_message: Error message
        """
        if batch_id != self.batch_id:
            return
        
        self.title_label.setText("Batch Processing Failed")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
    
    def on_batch_cancelled(self, batch_id):
        """Handle batch cancellation.
        
        Args:
            batch_id: ID of the cancelled batch
        """
        if batch_id != self.batch_id:
            return
        
        self.title_label.setText("Batch Processing Cancelled")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.status_label.setText("Batch processing was cancelled.")
    
    def on_item_started(self, batch_id, item_id):
        """Handle item start.
        
        Args:
            batch_id: Batch ID
            item_id: Item ID
        """
        # This is handled by update_status
        pass
    
    def on_item_progress(self, batch_id, item_id, progress, message):
        """Handle item progress update.
        
        Args:
            batch_id: Batch ID
            item_id: Item ID
            progress: Progress percentage
            message: Progress message
        """
        # This is handled by update_status
        pass
    
    def on_item_completed(self, batch_id, item_id):
        """Handle item completion.
        
        Args:
            batch_id: Batch ID
            item_id: Item ID
        """
        # This is handled by update_status
        pass
    
    def on_item_failed(self, batch_id, item_id, error):
        """Handle item failure.
        
        Args:
            batch_id: Batch ID
            item_id: Item ID
            error: Error message
        """
        # This is handled by update_status
        pass
    
    def cancel_batch(self):
        """Cancel the batch processing."""
        reply = QMessageBox.question(
            self,
            "Cancel Batch",
            "Are you sure you want to cancel the batch processing?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.batch_processor.cancel_batch(self.batch_id)
            self.cancel_button.setEnabled(False)
    
    def closeEvent(self, event):
        """Handle dialog close event.
        
        Args:
            event: Close event
        """
        # If batch is complete, accept the close
        batch_status = self.batch_processor.get_batch_status(self.batch_id)
        if batch_status and batch_status['is_complete']:
            event.accept()
        else:
            # Ask if user wants to cancel the batch
            reply = QMessageBox.question(
                self,
                "Close Dialog",
                "Batch processing is still running. Do you want to cancel it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.batch_processor.cancel_batch(self.batch_id)
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                # Allow closing without cancelling
                event.accept()
            else:
                # Cancel the close
                event.ignore()


def run_batch_dialog(parent=None):
    """Show batch processing dialog and start processing if setup.
    
    Args:
        parent: Parent widget
        
    Returns:
        True if batch was started, False otherwise
    """
    # First, show setup dialog
    setup_dialog = BatchSetupDialog(parent)
    if setup_dialog.exec() != QDialog.Accepted:
        return False
    
    # Get batch items
    batch_items = setup_dialog.get_batch_items()
    if not batch_items:
        QMessageBox.warning(
            parent,
            "No Items",
            "No items were set up for batch processing."
        )
        return False
    
    # Create batch
    batch_processor = get_batch_processor()
    try:
        batch_id = batch_processor.create_batch(
            setup_dialog.operation_id,
            batch_items
        )
    except ValueError as e:
        QMessageBox.critical(
            parent,
            "Batch Setup Error",
            f"Failed to create batch: {str(e)}"
        )
        return False
    
    # Show progress dialog
    progress_dialog = BatchProgressDialog(batch_id, parent)
    
    # Start batch processing
    def on_batch_progress(progress, message):
        """Handle batch progress updates."""
        # Progress updates are handled by the dialog's timer
        pass
    
    batch_processor.start_batch(
        batch_id,
        max_concurrent_items=2,
        on_batch_progress=on_batch_progress
    )
    
    # Show dialog
    progress_dialog.exec()
    
    return True 