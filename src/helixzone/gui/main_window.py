from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QMenuBar,
    QApplication, QWidget, QVBoxLayout, QLabel,
    QFileDialog, QMessageBox, QButtonGroup, QAbstractButton, QDialog, QStatusBar, QPushButton, QTabWidget, QProgressDialog, QDialogButtonBox, QFormLayout, QGroupBox, QSpinBox, QRadioButton, QSlider, QCheckBox, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QAction, QIcon, QImage
from .canvas import Canvas, CanvasView
from .layer_widget import LayerWidget
from .tool_options import ToolOptionsWidget
from typing import Optional
import os
import logging
from .error_dialog import error_notification
from ..core.logging_manager import get_logger
from ..core.task_manager import task_manager
from .task_monitor_widget import TaskMonitorWidget
import cv2
from ..core.file_manager import get_file_manager, FileFormat
from .file_dialog import get_open_filename, get_save_filename, get_export_filename
from .batch_dialog import run_batch_dialog
from .export_dialog import show_export_dialog

logger = get_logger('helixzone.gui.main_window')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HelixZone Image Editor")
        self.setMinimumSize(1024, 768)
        
        # Setup error notification
        error_notification.set_parent(self)
        
        # Initialize canvas
        try:
            self.canvas_view = CanvasView()
            self.setCentralWidget(self.canvas_view)
        except Exception as e:
            error_notification.show_from_exception(
                "Canvas Initialization Error",
                "Failed to initialize canvas",
                e
            )
            logger.error("Failed to initialize canvas", exc_info=True)
        
        # Initialize UI components
        try:
            self.setup_menubar()
            self.setup_dock_widgets()  # Create tool options first
            self.setup_toolbar()       # Then setup toolbar which uses tool options
            self.setup_statusbar()
        except Exception as e:
            error_notification.show_from_exception(
                "UI Initialization Error",
                "Failed to initialize UI components",
                e
            )
            logger.error("Failed to initialize UI components", exc_info=True)
            
        # Set up error status check timer
        self.error_check_timer = QTimer(self)
        self.error_check_timer.timeout.connect(self.update_error_status)
        self.error_check_timer.start(5000)  # Check every 5 seconds
        
        # Initialize file manager
        self.file_manager = get_file_manager()
        
        # Set up recent files menu
        self._setup_recent_files_menu()
        
        # Connect file menu actions
        self.connect_file_actions()
        
    def setup_statusbar(self):
        """Set up status bar with error indicators."""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Add error indicator
        self.error_label = QLabel("No Errors", self)
        self.error_label.mousePressEvent = lambda e: self.show_error_summary()
        self.statusBar.addPermanentWidget(QLabel())  # Spacer
        
        # Update error status initially
        self.update_error_status()
        
    def update_error_status(self):
        """Update error status in status bar."""
        error_count = error_notification.get_error_count()
        if error_count > 0:
            self.error_label.setText(f"{error_count} Error(s)")
            # Can't set icons directly on QLabel - we'd need to use setPixmap instead
            # Just update the text for now
            self.statusBar.addPermanentWidget(self.error_label)
        else:
            if self.error_label.parent():
                self.statusBar.removeWidget(self.error_label)
            
    def show_error_summary(self):
        """Show summary of errors."""
        errors = error_notification.get_error_summary()
        if not errors:
            QMessageBox.information(self, "Error Summary", "No errors to display")
            return
            
        # Format error summary
        error_text = "Error Summary:\n\n"
        for error_key, error_info in errors.items():
            error_text += f"- {error_info['title']}: {error_info['message']}\n"
            error_text += f"  Count: {error_info['count']}\n"
            error_text += f"  First: {error_info['first_time']}\n"
            error_text += f"  Last: {error_info['last_time']}\n\n"
            
        # Show in error dialog
        from .error_dialog import ErrorDialog
        ErrorDialog.show_error(
            self,
            "Error Summary",
            f"There are {len(errors)} error types with a total of {error_notification.get_error_count()} occurrences",
            error_text,
            "Info"
        )
        
    def setup_menubar(self):
        menubar = self.menuBar()
        if menubar is None:
            return
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        if file_menu is None:
            return
        
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        batch_action = QAction("Batch Processing...", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self.run_batch_processing)
        file_menu.addAction(batch_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Layer Menu
        layer_menu = menubar.addMenu("&Layer")
        if layer_menu is None:
            return
        
        new_layer_action = QAction("New Layer", self)
        new_layer_action.setShortcut("Ctrl+Shift+N")
        new_layer_action.triggered.connect(self.add_layer)
        layer_menu.addAction(new_layer_action)
        
        merge_visible_action = QAction("Merge Visible", self)
        merge_visible_action.triggered.connect(self.merge_visible_layers)
        layer_menu.addAction(merge_visible_action)
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        if edit_menu is None:
            return
        
        # Undo action
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.canvas_view.canvas.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        
        # Redo action
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.canvas_view.canvas.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)
        
        # Connect command stack signals
        self.canvas_view.canvas.command_stack.changed.connect(self.update_undo_redo)
        
    def update_undo_redo(self):
        """Update the enabled state and text of undo/redo actions."""
        try:
            command_stack = self.canvas_view.canvas.command_stack
            
            # Update undo action
            self.undo_action.setEnabled(command_stack.can_undo())
            self.undo_action.setText(command_stack.get_undo_text())
            
            # Update redo action
            self.redo_action.setEnabled(command_stack.can_redo())
            self.redo_action.setText(command_stack.get_redo_text())
        except Exception as e:
            logger.error("Failed to update undo/redo actions", exc_info=True)
        
    def new_file(self):
        """Create a new blank image."""
        try:
            # Create a new white image
            new_image = QImage(QSize(800, 600), QImage.Format.Format_ARGB32)
            new_image.fill(Qt.GlobalColor.white)
            self.canvas_view.canvas.set_image(new_image)
        except Exception as e:
            error_notification.show_from_exception(
                "New File Error",
                "Failed to create new file",
                e
            )
        
    def open_file(self):
        """Open an image file."""
        try:
            if not hasattr(self, 'current_file'):
                self.save_file_as()
            else:
                image = self.canvas_view.canvas.get_image()
                if image is not None:
                    success = image.save(self.current_file)
                    if success:
                        logger.info(f"Saved image to: {self.current_file}")
                    else:
                        error_notification.show_error(
                            "Save Error",
                            f"Failed to save image to {self.current_file}"
                        )
        except Exception as e:
            error_notification.show_from_exception(
                "File Dialog Error",
                "Failed to open file dialog",
                e
            )
                
    def save_file(self):
        """Save the current image."""
        try:
            if not hasattr(self, 'current_file'):
                self.save_file_as()
            else:
                image = self.canvas_view.canvas.get_image()
                if image is not None:
                    success = image.save(self.current_file)
                    if success:
                        logger.info(f"Saved image to: {self.current_file}")
                    else:
                        error_notification.show_error(
                            "Save Error",
                            f"Failed to save image to {self.current_file}"
                        )
        except Exception as e:
            error_notification.show_from_exception(
                "Save Error",
                "Failed to save file",
                e
            )
            
    def save_file_as(self):
        """Save the current image to a new file."""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            
            if file_name:
                try:
                    image = self.canvas_view.canvas.get_image()
                    if image is not None:
                        success = image.save(file_name)
                        if success:
                            self.current_file = file_name
                            logger.info(f"Saved image as: {file_name}")
                        else:
                            error_notification.show_error(
                                "Save Error",
                                f"Failed to save image to {file_name}"
                            )
                except Exception as e:
                    error_notification.show_from_exception(
                        "Save As Error",
                        f"Could not save image to: {file_name}",
                        e
                    )
        except Exception as e:
            error_notification.show_from_exception(
                "File Dialog Error",
                "Failed to open save dialog",
                e
            )
        
    def setup_toolbar(self):
        # Main toolbar
        main_toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, main_toolbar)
        main_toolbar.setMovable(False)
        
        # Tool button group for exclusive selection
        tool_group = QButtonGroup(self)
        
        # Add all tools
        tool_actions = [
            ("Brush", "brush"),
            ("Eraser", "eraser"),
            ("Rectangle Selection", "rectangle_selection"),
            ("Ellipse Selection", "ellipse_selection"),
            ("Lasso Selection", "lasso_selection"),
            ("Magnetic Lasso", "magnetic_lasso"),
        ]
        
        for name, identifier in tool_actions:
            action = QAction(name, self)
            action.setCheckable(True)
            action.setData(identifier)  # Store tool identifier
            main_toolbar.addAction(action)
            
            # Add to button group and connect
            tool_button = main_toolbar.widgetForAction(action)
            if tool_button is not None and isinstance(tool_button, QAbstractButton):
                tool_group.addButton(tool_button)
                tool_button.clicked.connect(lambda checked, i=identifier: self.on_tool_changed(i))
        
        # Set brush as default tool
        first_action = main_toolbar.actions()[0]
        first_action.setChecked(True)
        self.on_tool_changed("brush")
        
    def on_tool_changed(self, tool_identifier: str) -> None:
        """Handle tool selection changes."""
        # Update the tool manager
        self.canvas_view.canvas.tool_manager.set_tool(tool_identifier)
        
        # Update tool options widget
        self.tool_options_widget.update_for_tool(tool_identifier)

    def select_tool(self, tool_identifier: str) -> None:
        """Select a tool programmatically (mainly for testing)."""
        # Find and check the corresponding action
        for action in self.findChildren(QAction):
            if action.data() == tool_identifier:
                action.setChecked(True)
                break
        
        # Update tool
        self.on_tool_changed(tool_identifier)
        
    def setup_dock_widgets(self):
        """Set up the dock widgets."""
        # Tool Options
        tool_options_dock = QDockWidget("Tool Options", self)
        tool_options_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.tool_options_widget = ToolOptionsWidget(self)
        tool_options_dock.setWidget(self.tool_options_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, tool_options_dock)
        
        # Layer Widget
        layer_dock = QDockWidget("Layers", self)
        layer_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.layer_widget = LayerWidget(self.canvas_view.canvas.layer_stack)
        layer_dock.setWidget(self.layer_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, layer_dock)
        
        # Connect layer widget signals
        self.layer_widget.update_layer_list()
        
    def add_layer(self):
        """Add a new layer to the canvas."""
        try:
            self.canvas_view.canvas.layer_stack.add_layer()
            logger.info("Added new layer")
        except Exception as e:
            error_notification.show_from_exception(
                "Layer Error",
                "Failed to add new layer",
                e
            )
        
    def merge_visible_layers(self):
        """Merge all visible layers into a new layer."""
        try:
            self.canvas_view.canvas.layer_stack.merge_visible()
            logger.info("Merged visible layers")
        except Exception as e:
            error_notification.show_from_exception(
                "Layer Error",
                "Failed to merge visible layers",
                e
            )
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Check for unsaved changes
            # ...existing code...
            
            # Clean up resources
            if hasattr(self, 'error_check_timer'):
                self.error_check_timer.stop()
                
            # Log application exit
            logger.info("Application closed")
            
            event.accept()
        except Exception as e:
            logger.error("Error during application close", exc_info=True)
            event.accept()  # Accept anyway to allow closing 

    def _setup_status_bar(self):
        """Set up status bar with task monitoring."""
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Add cursor position widget
        self.cursor_pos_label = QLabel("X: 0, Y: 0")
        self.status_bar.addPermanentWidget(self.cursor_pos_label)
        
        # Add task indicator widget
        self.task_indicator = QPushButton("Tasks: 0")
        self.task_indicator.setFlat(True)
        self.task_indicator.clicked.connect(self._toggle_task_monitor)
        self.status_bar.addPermanentWidget(self.task_indicator)
        
        # Create task monitor dock widget (initially hidden)
        self.task_monitor_dock = QDockWidget("Task Monitor", self)
        self.task_monitor_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | 
                                              Qt.DockWidgetArea.BottomDockWidgetArea)
        self.task_monitor_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | 
                                          QDockWidget.DockWidgetFeature.DockWidgetMovable)
        
        # Create task monitor widget
        self.task_monitor = TaskMonitorWidget()
        self.task_monitor_dock.setWidget(self.task_monitor)
        
        # Add dock widget but hide it initially
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.task_monitor_dock)
        self.task_monitor_dock.hide()
        
        # Connect close event to update button state
        self.task_monitor_dock.visibilityChanged.connect(self._update_task_indicator)
        
        # Set up timer to update task count
        self.task_timer = QTimer(self)
        self.task_timer.setInterval(1000)  # 1 second interval
        self.task_timer.timeout.connect(self._update_task_indicator)
        self.task_timer.start()
    
    def _toggle_task_monitor(self):
        """Toggle visibility of task monitor."""
        if self.task_monitor_dock.isVisible():
            self.task_monitor_dock.hide()
        else:
            self.task_monitor_dock.show()
    
    def _update_task_indicator(self):
        """Update task indicator with current count."""
        active_tasks = len(task_manager.get_active_tasks())
        
        if active_tasks > 0:
            self.task_indicator.setText(f"Tasks: {active_tasks}")
            self.task_indicator.setStyleSheet("QPushButton { color: blue; font-weight: bold; }")
        else:
            self.task_indicator.setText("Tasks: 0")
            self.task_indicator.setStyleSheet("")
    
    def _update_cursor_position(self, x, y):
        """Update cursor position display."""
        self.cursor_pos_label.setText(f"X: {x}, Y: {y}")

    def _setup_recent_files_menu(self):
        """Set up the recent files submenu."""
        # Find the recent files menu
        if hasattr(self, 'recent_files_menu'):
            # Clear existing entries
            self.recent_files_menu.clear()
            
            # Add entries for recent files
            recent_files = self.file_manager.get_recent_files()
            if recent_files:
                for i, file_path in enumerate(recent_files):
                    action = QAction(f"{i+1}. {os.path.basename(file_path)}", self)
                    action.setData(file_path)
                    action.setStatusTip(f"Open {file_path}")
                    action.triggered.connect(self._open_recent_file)
                    self.recent_files_menu.addAction(action)
                
                self.recent_files_menu.addSeparator()
                clear_action = QAction("Clear Recent Files", self)
                clear_action.triggered.connect(self._clear_recent_files)
                self.recent_files_menu.addAction(clear_action)
            else:
                no_recent_action = QAction("No Recent Files", self)
                no_recent_action.setEnabled(False)
                self.recent_files_menu.addAction(no_recent_action)
    
    def connect_file_actions(self):
        """Connect file menu actions to their handlers."""
        # Connect file actions
        if hasattr(self, 'action_new'):
            self.action_new.triggered.connect(self.new_document)
        
        if hasattr(self, 'action_open'):
            self.action_open.triggered.connect(self.open_document)
        
        if hasattr(self, 'action_save'):
            self.action_save.triggered.connect(self.save_document)
        
        if hasattr(self, 'action_save_as'):
            self.action_save_as.triggered.connect(self.save_document_as)
        
        if hasattr(self, 'action_export'):
            self.action_export.triggered.connect(self.export_document)
        
        if hasattr(self, 'action_close'):
            self.action_close.triggered.connect(self.close_document)
        
        if hasattr(self, 'action_exit'):
            self.action_exit.triggered.connect(self.close)
    
    def open_document(self):
        """Open an image file selected by the user."""
        file_path = get_open_filename(self, "Open Image")
        if file_path:
            self._load_document(file_path)
            self._setup_recent_files_menu()
    
    def _open_recent_file(self):
        """Open a file from the recent files menu."""
        action = self.sender()
        if action and isinstance(action, QAction):
            file_path = action.data()
            if file_path and os.path.isfile(file_path):
                self._load_document(file_path)
    
    def _clear_recent_files(self):
        """Clear the recent files list."""
        self.file_manager.clear_recent_files()
        self._setup_recent_files_menu()
    
    def _load_document(self, file_path):
        """Load a document from a file path."""
        if not file_path or not os.path.isfile(file_path):
            logger.error(f"Invalid file path: {file_path}")
            QMessageBox.critical(self, "Error", f"Could not open file: {file_path}")
            return
            
        # Get file manager
        file_manager = get_file_manager()
        
        # Show progress dialog
        progress_dialog = QProgressDialog("Loading image...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Loading")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(500)  # Show after 500ms
        progress_dialog.setValue(0)
        
        def on_progress(progress):
            """Handle progress updates."""
            if progress.indeterminate:
                progress_dialog.setRange(0, 0)
            else:
                progress_dialog.setRange(0, 100)
                progress_dialog.setValue(progress.percent)
                
            if progress.message:
                progress_dialog.setLabelText(progress.message)
                
        def on_complete(image):
            """Handle load completion."""
            progress_dialog.setValue(100)
            
            if image:
                # Create new tab with the loaded image
                file_name = os.path.basename(file_path)
                self._create_tab_with_image(image, file_name, file_path)
                
                # Update recent files menu
                self._setup_recent_files_menu()
            else:
                QMessageBox.critical(self, "Error", f"Failed to load image: {file_path}")
                
        def on_error(err):
            """Handle load error."""
            progress_dialog.setValue(100)
            QMessageBox.critical(self, "Error", f"Error loading image: {str(err)}")
            
        def on_cancel():
            """Handle user cancellation."""
            # Cancel the task
            if task_id:
                file_manager.cancel_task(task_id)
                
        # Connect cancel button
        progress_dialog.canceled.connect(on_cancel)
        
        # Extract file extension to check for special formats
        ext = os.path.splitext(file_path)[1].lower()
        is_raw = ext in ['.arw', '.cr2', '.cr3', '.dng', '.nef', '.orf', '.pef', '.raf', '.rw2', '.srw', '.x3f']
        is_hdr = ext in ['.hdr', '.exr', '.pfm']
        
        # If Raw or HDR, show options dialog first
        load_options = {}
        
        if is_raw:
            # Ask about RAW processing options
            raw_dialog = QDialog(self)
            raw_dialog.setWindowTitle("RAW Processing Options")
            raw_dialog.resize(400, 300)
            
            layout = QVBoxLayout(raw_dialog)
            
            # Processing mode
            mode_group = QGroupBox("Processing Mode")
            mode_layout = QVBoxLayout(mode_group)
            
            srgb_radio = QRadioButton("Standard (sRGB)")
            srgb_radio.setChecked(True)
            linear_radio = QRadioButton("Linear (no tone mapping)")
            custom_radio = QRadioButton("Custom")
            
            mode_layout.addWidget(srgb_radio)
            mode_layout.addWidget(linear_radio)
            mode_layout.addWidget(custom_radio)
            
            layout.addWidget(mode_group)
            
            # White balance
            wb_group = QGroupBox("White Balance")
            wb_layout = QVBoxLayout(wb_group)
            
            auto_wb = QCheckBox("Auto White Balance")
            auto_wb.setChecked(True)
            
            wb_layout.addWidget(auto_wb)
            
            layout.addWidget(wb_group)
            
            # Other options
            options_group = QGroupBox("Additional Options")
            options_layout = QFormLayout(options_group)
            
            brightness_slider = QSlider(Qt.Orientation.Horizontal)
            brightness_slider.setRange(50, 150)
            brightness_slider.setValue(100)
            brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            brightness_slider.setTickInterval(10)
            
            options_layout.addRow("Brightness:", brightness_slider)
            
            highlight_recovery = QCheckBox("Highlight Recovery")
            highlight_recovery.setChecked(True)
            options_layout.addRow("Recovery:", highlight_recovery)
            
            layout.addWidget(options_group)
            
            # Buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)
            
            button_box.accepted.connect(raw_dialog.accept)
            button_box.rejected.connect(raw_dialog.reject)
            
            # Show dialog
            if raw_dialog.exec() == QDialog.Accepted:
                # Get options
                if srgb_radio.isChecked():
                    load_options['raw_processing_mode'] = 'SRGB'
                elif linear_radio.isChecked():
                    load_options['raw_processing_mode'] = 'LINEAR'
                else:
                    load_options['raw_processing_mode'] = 'CUSTOM'
                    
                load_options['auto_white_balance'] = auto_wb.isChecked()
                load_options['brightness'] = brightness_slider.value() / 100.0
                load_options['highlight_recovery'] = highlight_recovery.isChecked()
            else:
                # User cancelled
                return
                
        elif is_hdr:
            # Ask about HDR processing options
            hdr_dialog = QDialog(self)
            hdr_dialog.setWindowTitle("HDR Processing Options")
            hdr_dialog.resize(400, 200)
            
            layout = QVBoxLayout(hdr_dialog)
            
            # Tone mapping
            tone_group = QGroupBox("Tone Mapping")
            tone_layout = QVBoxLayout(tone_group)
            
            tone_map = QCheckBox("Apply Tone Mapping")
            tone_map.setChecked(True)
            tone_layout.addWidget(tone_map)
            
            exposure_slider = QSlider(Qt.Orientation.Horizontal)
            exposure_slider.setRange(-30, 30)
            exposure_slider.setValue(0)
            exposure_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            exposure_slider.setTickInterval(10)
            
            exposure_label = QLabel("0.0 EV")
            exposure_slider.valueChanged.connect(
                lambda v: exposure_label.setText(f"{v/10:.1f} EV"))
            
            exposure_layout = QHBoxLayout()
            exposure_layout.addWidget(QLabel("Exposure:"))
            exposure_layout.addWidget(exposure_slider)
            exposure_layout.addWidget(exposure_label)
            
            tone_layout.addLayout(exposure_layout)
            
            layout.addWidget(tone_group)
            
            # Buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)
            
            button_box.accepted.connect(hdr_dialog.accept)
            button_box.rejected.connect(hdr_dialog.reject)
            
            # Show dialog
            if hdr_dialog.exec() == QDialog.Accepted:
                # Get options
                load_options['tone_map'] = tone_map.isChecked()
                load_options['exposure'] = exposure_slider.value() / 10.0
            else:
                # User cancelled
                return
        
        # Start loading task
        task_id = file_manager.load_image_async(
            file_path,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
            options=load_options
        )
    
    def _create_tab_with_image(self, image, title, file_path=None):
        """Create a new tab with the given image.
        
        Args:
            image: QImage to display
            title: Tab title
            file_path: Optional original file path
        """
        # Create a new canvas with the image
        canvas = Canvas(self)
        canvas.set_image(image)
        canvas.file_path = file_path
        
        # Create a new tab with the canvas
        if hasattr(self, 'document_tabs'):
            index = self.document_tabs.addTab(canvas, title)
            self.document_tabs.setCurrentIndex(index)
            
            # Update window title
            self.setWindowTitle(f"{title} - HelixZone")
    
    def new_document(self):
        """Create a new empty document."""
        # Show dialog to get dimensions
        dialog = QDialog(self)
        dialog.setWindowTitle("New Image")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Width and height inputs
        form_layout = QFormLayout()
        width_input = QSpinBox()
        width_input.setRange(1, 10000)
        width_input.setValue(1920)
        
        height_input = QSpinBox()
        height_input.setRange(1, 10000)
        height_input.setValue(1080)
        
        form_layout.addRow("Width:", width_input)
        form_layout.addRow("Height:", height_input)
        
        # Background options
        bg_group = QGroupBox("Background")
        bg_layout = QVBoxLayout(bg_group)
        
        white_radio = QRadioButton("White")
        transparent_radio = QRadioButton("Transparent")
        white_radio.setChecked(True)
        
        bg_layout.addWidget(white_radio)
        bg_layout.addWidget(transparent_radio)
        
        # Add to main layout
        layout.addLayout(form_layout)
        layout.addWidget(bg_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec() == QDialog.Accepted:
            width = width_input.value()
            height = height_input.value()
            
            # Create image with selected background
            if transparent_radio.isChecked():
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.transparent)
            else:
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.white)
            
            # Create new tab
            self._create_tab_with_image(image, "Untitled")
    
    def save_document(self):
        """Save the current document."""
        canvas = self._get_current_canvas()
        if not canvas:
            return
        
        if hasattr(canvas, 'file_path') and canvas.file_path:
            self._save_document_to_path(canvas, canvas.file_path)
        else:
            self.save_document_as()
    
    def save_document_as(self):
        """Save the current document with a new filename."""
        canvas = self._get_current_canvas()
        if not canvas:
            return
        
        # Get save path and options
        file_path, options = get_save_filename(
            self, "Save Image", 
            directory="" if not hasattr(canvas, 'file_path') or not canvas.file_path else canvas.file_path
        )
        
        if file_path:
            self._save_document_to_path(canvas, file_path, options)
    
    def export_document(self):
        """Export the current document with advanced options."""
        canvas = self._get_current_canvas()
        if not canvas:
            return
            
        # Get current file format
        current_format = FileFormat.PNG
        if canvas.file_path:
            ext = os.path.splitext(canvas.file_path)[1].lower()
            current_format = FileFormat.from_extension(ext)
            
        # Show export options dialog
        accepted, export_options = show_export_dialog(self, current_format)
        if not accepted:
            return
            
        # Get export file path
        file_path, _ = get_export_filename(self, "Export Image", "", export_options['format'])
        if not file_path:
            return
            
        # Export with options
        self._save_document_to_path(canvas, file_path, export_options, is_export=True)
    
    def _save_document_to_path(self, canvas, file_path, options=None, is_export=False):
        """Save the canvas to the specified path.
        
        Args:
            canvas: Canvas to save
            file_path: Path to save to
            options: Optional save options
            is_export: Whether this is an export operation
        """
        # Get the image from canvas
        image = canvas.get_image()
        if not image:
            QMessageBox.warning(
                self, 
                "No Image", 
                "There is no image to save."
            )
            return
        
        # Show status message
        operation = "Exporting" if is_export else "Saving"
        self.status_label.setText(f"{operation} {os.path.basename(file_path)}...")
        
        # Create progress dialog
        progress_dialog = QProgressDialog(f"{operation} image...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle(f"{operation} Image")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(500)  # Show after 500ms
        
        # Extract options
        format = None
        quality = 90
        if options:
            format = options.get('format')
            quality = options.get('quality', 90)
        
        # Define callbacks
        def on_progress(progress):
            progress_dialog.setValue(int(progress.percent))
            progress_dialog.setLabelText(progress.message)
            QApplication.processEvents()
        
        def on_complete(success):
            progress_dialog.close()
            
            if not success:
                QMessageBox.critical(
                    self, 
                    f"Error {operation} Image", 
                    f"Failed to {operation.lower()} the image to {file_path}."
                )
                self.status_label.setText(f"Failed to {operation.lower()} image")
                return
            
            # If it's a regular save (not export), update the canvas file path
            if not is_export:
                canvas.file_path = file_path
                index = self.document_tabs.indexOf(canvas)
                if index >= 0:
                    self.document_tabs.setTabText(index, os.path.basename(file_path))
                    self.setWindowTitle(f"{os.path.basename(file_path)} - HelixZone")
            
            self.status_label.setText(f"{operation} complete: {os.path.basename(file_path)}")
            
            # Update recent files menu
            self._setup_recent_files_menu()
        
        def on_error(err):
            progress_dialog.close()
            QMessageBox.critical(
                self, 
                f"Error {operation} Image", 
                f"An error occurred while {operation.lower()} the image: {str(err)}"
            )
            self.status_label.setText("Error saving image")
        
        # Start async saving
        task_id = self.file_manager.save_image_async(
            image, 
            file_path,
            format=format,
            quality=quality,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error
        )
        
        # Connect cancel button
        def on_cancel():
            self.task_manager.cancel_task(task_id)
            self.status_label.setText("Saving cancelled")
        
        progress_dialog.canceled.connect(on_cancel)
        progress_dialog.show()
    
    def close_document(self):
        """Close the current document."""
        index = self.document_tabs.currentIndex()
        if index >= 0:
            self._close_tab(index)
    
    def _close_tab(self, index):
        """Close the tab at the specified index.
        
        Args:
            index: Index of the tab to close
        """
        canvas = self.document_tabs.widget(index)
        
        # Check if the document has unsaved changes
        if hasattr(canvas, 'has_unsaved_changes') and canvas.has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "The document has unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                # Need to save current tab first
                current_index = self.document_tabs.currentIndex()
                self.document_tabs.setCurrentIndex(index)
                self.save_document()
                
                # Check if still has unsaved changes (user might have cancelled save dialog)
                if hasattr(canvas, 'has_unsaved_changes') and canvas.has_unsaved_changes:
                    # User cancelled save, don't close tab
                    self.document_tabs.setCurrentIndex(current_index)
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return
        
        # Close the tab
        self.document_tabs.removeTab(index)
        
        # If no tabs left, update window title
        if self.document_tabs.count() == 0:
            self.setWindowTitle("HelixZone")
    
    def _get_current_canvas(self):
        """Get the canvas in the current tab.
        
        Returns:
            Canvas widget or None if no tabs open
        """
        if not hasattr(self, 'document_tabs'):
            return None
        
        index = self.document_tabs.currentIndex()
        if index < 0:
            return None
        
        canvas = self.document_tabs.widget(index)
        return canvas 

    def run_batch_processing(self):
        """Launch the batch processing dialog."""
        run_batch_dialog(self)

    def save_as(self):
        """Alias for save_file_as."""
        self.save_file_as() 