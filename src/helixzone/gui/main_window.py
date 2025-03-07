from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QMenuBar,
    QApplication, QWidget, QVBoxLayout, QLabel,
    QFileDialog, QMessageBox, QButtonGroup, QAbstractButton
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QImage
from .canvas import CanvasView
from .layer_widget import LayerWidget
from .tool_options import ToolOptionsWidget
from typing import Optional
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HelixZone Image Editor")
        self.setMinimumSize(1024, 768)
        
        # Initialize canvas
        self.canvas_view = CanvasView()
        self.setCentralWidget(self.canvas_view)
        
        # Initialize UI components
        self.setup_menubar()
        self.setup_dock_widgets()  # Create tool options first
        self.setup_toolbar()       # Then setup toolbar which uses tool options
        
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
        command_stack = self.canvas_view.canvas.command_stack
        
        # Update undo action
        self.undo_action.setEnabled(command_stack.can_undo())
        self.undo_action.setText(command_stack.get_undo_text())
        
        # Update redo action
        self.redo_action.setEnabled(command_stack.can_redo())
        self.redo_action.setText(command_stack.get_redo_text())
        
    def new_file(self):
        """Create a new blank image."""
        # Create a new white image
        new_image = QImage(QSize(800, 600), QImage.Format.Format_ARGB32)
        new_image.fill(Qt.GlobalColor.white)
        self.canvas_view.canvas.set_image(new_image)
        
    def open_file(self):
        """Open an image file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        
        if file_name:
            try:
                image = QImage(file_name)
                if image.isNull():
                    raise Exception("Failed to load image")
                self.canvas_view.canvas.set_image(image)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Could not open image: {str(e)}"
                )
                
    def save_file(self):
        """Save the current image."""
        if not hasattr(self, 'current_file'):
            self.save_file_as()
        else:
            image = self.canvas_view.canvas.get_image()
            if image is not None:
                image.save(self.current_file)
            
    def save_file_as(self):
        """Save the current image to a new file."""
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
                    image.save(file_name)
                    self.current_file = file_name
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Could not save image: {str(e)}"
                )
        
    def setup_toolbar(self):
        # Main toolbar
        main_toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, main_toolbar)
        main_toolbar.setMovable(False)
        
        # Tool button group for exclusive selection
        tool_group = QButtonGroup(self)
        
        # Add basic tools
        tool_actions = [
            ("Select", "select"),
            ("Move", "move"),
            ("Brush", "brush"),
            ("Eraser", "eraser"),
            ("Crop", "crop"),
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
            
            # Connect tool selection
            if identifier in ['brush', 'eraser']:
                action.triggered.connect(
                    lambda checked, tool=identifier: self.select_tool(tool)
                )
        
        # Set brush as default tool
        buttons = tool_group.buttons()
        if len(buttons) > 2:  # Check for valid index
            buttons[2].setChecked(True)  # Brush is third in the list
            self.select_tool('brush')
            
    def setup_dock_widgets(self):
        # Layers dock
        layers_dock = QDockWidget("Layers", self)
        layers_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | 
                                  Qt.DockWidgetArea.LeftDockWidgetArea)
        
        self.layer_widget = LayerWidget(self.canvas_view.canvas.layer_stack)
        layers_dock.setWidget(self.layer_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, layers_dock)
        
        # Connect layer signals
        self.layer_widget.layer_added.connect(self.canvas_view.canvas.update)
        self.layer_widget.layer_removed.connect(self.canvas_view.canvas.update)
        self.layer_widget.layer_selected.connect(self.canvas_view.canvas.update)
        
        # Tools dock
        tools_dock = QDockWidget("Tool Options", self)
        tools_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | 
                                 Qt.DockWidgetArea.LeftDockWidgetArea)
        
        # Create tool options widget
        self.tool_options = ToolOptionsWidget(self.canvas_view.canvas.tool_manager)
        tools_dock.setWidget(self.tool_options)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, tools_dock)
    
    def add_layer(self):
        """Add a new layer."""
        self.layer_widget.add_layer()
    
    def merge_visible_layers(self):
        """Merge all visible layers into a new layer."""
        merged = self.canvas_view.canvas.layer_stack.merge_visible()
        if merged:
            new_layer = self.canvas_view.canvas.layer_stack.add_layer(name="Merged")
            new_layer.set_image(merged)
            self.layer_widget.update_layer_list()
            self.canvas_view.canvas.update()
    
    def select_tool(self, tool_name):
        """Switch to a different tool."""
        self.canvas_view.canvas.tool_manager.set_tool(tool_name)
        self.tool_options.update_for_tool(tool_name) 