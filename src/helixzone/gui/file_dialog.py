"""
Enhanced file dialog component with optimized loading/saving operations.

This module provides a custom file dialog with preview capabilities,
recent files section, and integration with the FileManager.
"""

import os
import logging
from typing import Optional, List, Tuple, Callable

from PyQt6.QtWidgets import (
    QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QSplitter, QFrame,
    QLineEdit, QComboBox, QCheckBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, QSize, QFileInfo, pyqtSignal, QRect

from ..core.file_manager import get_file_manager, FileFormat
from ..core.task_manager import TaskProgress

logger = logging.getLogger(__name__)


class FilePreviewWidget(QWidget):
    """Widget that shows a preview of the selected image file."""
    
    def __init__(self, parent=None):
        """Initialize the preview widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Preview image label
        self.image_label = QLabel("No preview available")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }"
        )
        layout.addWidget(self.image_label)
        
        # File info section
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        
        # File info labels
        self.filename_label = QLabel("Filename: ")
        self.size_label = QLabel("Size: ")
        self.dimensions_label = QLabel("Dimensions: ")
        self.format_label = QLabel("Format: ")
        
        info_layout.addWidget(self.filename_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.dimensions_label)
        info_layout.addWidget(self.format_label)
        info_layout.addStretch()
        
        layout.addWidget(info_frame)
        layout.addStretch()
        
        # Initialize file manager
        self.file_manager = get_file_manager()
        
        # Current preview file
        self.current_file = None
    
    def update_preview(self, file_path: str) -> None:
        """Update the preview for the selected file.
        
        Args:
            file_path: Path to the selected file
        """
        if not file_path or not os.path.isfile(file_path):
            self.clear_preview()
            return
        
        self.current_file = file_path
        
        # Get metadata
        metadata = self.file_manager.get_metadata(file_path)
        if not metadata:
            self.clear_preview()
            self.image_label.setText("Preview not available")
            return
        
        # Update file info
        file_info = QFileInfo(file_path)
        file_size = file_info.size()
        size_text = self._format_file_size(file_size)
        
        self.filename_label.setText(f"Filename: {file_info.fileName()}")
        self.size_label.setText(f"Size: {size_text}")
        
        if metadata.width > 0 and metadata.height > 0:
            self.dimensions_label.setText(f"Dimensions: {metadata.width} × {metadata.height}")
        else:
            self.dimensions_label.setText("Dimensions: Unknown")
            
        format_description = FileFormat.get_description(metadata.format)
        self.format_label.setText(f"Format: {format_description}")
        
        # Load thumbnail
        try:
            # Load a small preview image
            image = self.file_manager.load_image(file_path)
            if image:
                # Scale to fit preview area while maintaining aspect ratio
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width() - 10, 
                    self.image_label.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Preview not available")
        except Exception as e:
            logger.error(f"Error loading preview: {e}")
            self.image_label.setText("Error loading preview")
    
    def clear_preview(self) -> None:
        """Clear the current preview."""
        self.current_file = None
        self.image_label.clear()
        self.image_label.setText("No preview available")
        self.filename_label.setText("Filename: ")
        self.size_label.setText("Size: ")
        self.dimensions_label.setText("Dimensions: ")
        self.format_label.setText("Format: ")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted file size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class RecentFilesWidget(QWidget):
    """Widget that displays and allows selection of recent files."""
    
    file_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the recent files widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setMinimumWidth(200)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel("Recent Files")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)
        
        # Recent files list
        self.files_list = QListWidget()
        self.files_list.setAlternatingRowColors(True)
        self.files_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.files_list)
        
        # Clear button
        self.clear_button = QPushButton("Clear Recent Files")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self.clear_button)
        
        # Initialize file manager
        self.file_manager = get_file_manager()
        
        # Refresh the list
        self.refresh_list()
    
    def refresh_list(self) -> None:
        """Refresh the recent files list."""
        self.files_list.clear()
        
        recent_files = self.file_manager.get_recent_files()
        for file_path in recent_files:
            if os.path.isfile(file_path):
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                item.setToolTip(file_path)
                
                # Try to add a small icon
                metadata = self.file_manager.get_metadata(file_path)
                if metadata and metadata.format != FileFormat.UNKNOWN:
                    # For now, just use a generic icon
                    # In a production app, you might want to generate thumbnails
                    item.setIcon(QIcon.fromTheme("image-x-generic"))
                
                self.files_list.addItem(item)
        
        # Update clear button state
        self.clear_button.setEnabled(self.files_list.count() > 0)
    
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item click event.
        
        Args:
            item: The clicked list item
        """
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and os.path.isfile(file_path):
            self.file_selected.emit(file_path)
    
    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self.file_manager.clear_recent_files()
        self.refresh_list()


class EnhancedFileDialog(QFileDialog):
    """Enhanced file dialog with preview and recent files."""
    
    def __init__(self, parent=None, caption="", directory="", 
                filter="", is_save_dialog=False, default_suffix=""):
        """Initialize the enhanced file dialog.
        
        Args:
            parent: Parent widget
            caption: Dialog caption
            directory: Initial directory
            filter: File filter string
            is_save_dialog: Whether this is a save dialog
            default_suffix: Default file suffix for save dialog
        """
        super().__init__(parent, caption, directory, filter)
        
        # Set dialog options
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if is_save_dialog:
            self.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            self.setDefaultSuffix(default_suffix)
        else:
            self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            self.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        # Create preview widget
        self.preview_widget = FilePreviewWidget(self)
        
        # Create recent files widget
        self.recent_files_widget = RecentFilesWidget(self)
        self.recent_files_widget.file_selected.connect(self._on_recent_file_selected)
        
        # Add widgets to layout
        layout = self.layout()
        
        # Create a splitter for the main content and sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Get the existing widgets
        file_view = None
        sidebar_layout = QVBoxLayout()
        
        # Find the file view widget (typically at index 1)
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget and isinstance(widget, QWidget) and not isinstance(widget, QLabel):
                if not file_view:
                    file_view = widget
                    break
        
        # If we found the file view, move it to our splitter
        if file_view:
            layout.removeWidget(file_view)
            splitter.addWidget(file_view)
            
            # Create a widget for the sidebar
            sidebar_widget = QWidget()
            sidebar_widget.setLayout(sidebar_layout)
            sidebar_layout.addWidget(self.recent_files_widget)
            sidebar_layout.addWidget(self.preview_widget)
            
            # Add sidebar to splitter
            splitter.addWidget(sidebar_widget)
            
            # Set splitter sizes (70% file view, 30% sidebar)
            splitter.setSizes([700, 300])
            
            # Add splitter to layout
            layout.addWidget(splitter, 1, 0, 1, layout.columnCount())
        
        # Connect to file selection change
        self.currentChanged.connect(self._on_selection_changed)
        self.fileSelected.connect(self._on_file_selected)
    
    def _on_selection_changed(self, path: str) -> None:
        """Handle selection change in the file dialog.
        
        Args:
            path: Path to the selected file
        """
        if os.path.isfile(path):
            self.preview_widget.update_preview(path)
    
    def _on_file_selected(self, path: str) -> None:
        """Handle file selection (when user clicks Open or Save).
        
        Args:
            path: Path to the selected file
        """
        # The file manager will automatically add this to recent files
        # when loading or saving the file
        pass
    
    def _on_recent_file_selected(self, path: str) -> None:
        """Handle selection from recent files list.
        
        Args:
            path: Path to the selected file
        """
        if self.acceptMode() == QFileDialog.AcceptMode.AcceptOpen:
            # For open dialog, select the file
            self.selectFile(path)
            self.preview_widget.update_preview(path)
        else:
            # For save dialog, just set the directory and file name
            self.selectFile(path)


class SaveOptionsWidget(QWidget):
    """Widget for configuring save options like format and quality."""
    
    options_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the save options widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.format_combo = QComboBox()
        
        # Add supported formats
        self.formats = [
            (FileFormat.PNG, "PNG - Portable Network Graphics"),
            (FileFormat.JPEG, "JPEG - Joint Photographic Experts Group"),
            (FileFormat.TIFF, "TIFF - Tagged Image File Format"),
            (FileFormat.WEBP, "WebP - Web Picture Format"),
            (FileFormat.BMP, "BMP - Bitmap Image"),
            (FileFormat.GIF, "GIF - Graphics Interchange Format")
        ]
        
        for format_enum, format_name in self.formats:
            self.format_combo.addItem(format_name, format_enum)
        
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        # Quality settings
        quality_layout = QHBoxLayout()
        self.quality_label = QLabel("Quality:")
        self.quality_combo = QComboBox()
        
        # Add quality options
        quality_options = [
            ("Maximum (100%)", 100),
            ("High (90%)", 90),
            ("Medium (75%)", 75),
            ("Low (50%)", 50)
        ]
        
        for quality_name, quality_value in quality_options:
            self.quality_combo.addItem(quality_name, quality_value)
        
        quality_layout.addWidget(self.quality_label)
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)
        
        # Additional options
        self.preserve_metadata_check = QCheckBox("Preserve metadata (EXIF, ICC profile)")
        self.preserve_metadata_check.setChecked(True)
        layout.addWidget(self.preserve_metadata_check)
        
        # Connect signals
        self.format_combo.currentIndexChanged.connect(self._update_options)
        self.quality_combo.currentIndexChanged.connect(self.options_changed.emit)
        self.preserve_metadata_check.toggled.connect(self.options_changed.emit)
        
        # Initial update
        self._update_options()
    
    def _update_options(self) -> None:
        """Update available options based on selected format."""
        current_format = self.get_selected_format()
        
        # Update quality control visibility
        quality_visible = current_format in [FileFormat.JPEG, FileFormat.WEBP]
        self.quality_label.setVisible(quality_visible)
        self.quality_combo.setVisible(quality_visible)
        
        # Emit signal
        self.options_changed.emit()
    
    def get_selected_format(self) -> FileFormat:
        """Get the currently selected format.
        
        Returns:
            Selected file format enum
        """
        return self.format_combo.currentData()
    
    def get_selected_quality(self) -> int:
        """Get the currently selected quality.
        
        Returns:
            Quality value (0-100)
        """
        return self.quality_combo.currentData()
    
    def get_preserve_metadata(self) -> bool:
        """Get whether to preserve metadata.
        
        Returns:
            True if metadata should be preserved
        """
        return self.preserve_metadata_check.isChecked()
    
    def set_format(self, format: FileFormat) -> None:
        """Set the current format.
        
        Args:
            format: Format to select
        """
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == format:
                self.format_combo.setCurrentIndex(i)
                break


def get_open_filename(parent=None, caption="Open Image", 
                     directory="", filter=None) -> Optional[str]:
    """Show an enhanced file open dialog.
    
    Args:
        parent: Parent widget
        caption: Dialog caption
        directory: Initial directory
        filter: File filter string
        
    Returns:
        Selected file path or None if canceled
    """
    # Use file manager's filter if none specified
    if filter is None:
        filter = FileFormat.get_all_filters()
    
    dialog = EnhancedFileDialog(parent, caption, directory, filter)
    
    if dialog.exec() == QFileDialog.DialogCode.Accepted:
        file_names = dialog.selectedFiles()
        if file_names:
            return file_names[0]
    
    return None


def get_save_filename(parent=None, caption="Save Image", 
                     directory="", filter=None, 
                     format=FileFormat.PNG, 
                     options_callback=None) -> Tuple[Optional[str], Optional[dict]]:
    """Show an enhanced file save dialog with format options.
    
    Args:
        parent: Parent widget
        caption: Dialog caption
        directory: Initial directory
        filter: File filter string
        format: Default format
        options_callback: Optional callback to configure save options
        
    Returns:
        Tuple of (selected file path or None if canceled, save options dict)
    """
    # Use file manager's filter if none specified
    if filter is None:
        filter = FileFormat.get_all_filters()
    
    dialog = EnhancedFileDialog(
        parent, caption, directory, filter, 
        is_save_dialog=True, 
        default_suffix=FileFormat.get_extension(format)
    )
    
    # Add save options widget
    options_widget = SaveOptionsWidget(dialog)
    options_widget.set_format(format)
    
    # Add to dialog layout
    dialog.layout().addWidget(options_widget, dialog.layout().rowCount(), 0, 1, dialog.layout().columnCount())
    
    # Call the options callback if provided
    if options_callback:
        options_callback(options_widget)
    
    if dialog.exec() == QFileDialog.DialogCode.Accepted:
        file_names = dialog.selectedFiles()
        if file_names:
            # Get save options
            save_options = {
                'format': options_widget.get_selected_format(),
                'quality': options_widget.get_selected_quality(),
                'preserve_metadata': options_widget.get_preserve_metadata()
            }
            return file_names[0], save_options
    
    return None, None


def get_export_filename(parent=None, caption="Export Image", 
                       directory="", format=FileFormat.PNG) -> Tuple[Optional[str], Optional[dict]]:
    """Show a specialized dialog for exporting images.
    
    Args:
        parent: Parent widget
        caption: Dialog caption
        directory: Initial directory
        format: Default format
        
    Returns:
        Tuple of (selected file path or None if canceled, export options dict)
    """
    # For export, use a specific filter for the requested format
    filter = FileFormat.get_filter_string(format)
    
    return get_save_filename(
        parent, caption, directory, filter, format,
        options_callback=None  # Could customize export options here
    ) 