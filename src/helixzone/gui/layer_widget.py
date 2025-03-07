from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QSlider,
    QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from ..core.commands import LayerCommand

class LayerItem(QWidget):
    """Widget representing a single layer in the layer list."""
    
    visibility_changed = pyqtSignal(bool)
    opacity_changed = pyqtSignal(float)
    
    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Visibility toggle
        self.visibility_cb = QCheckBox()
        self.visibility_cb.setChecked(layer.visible)
        self.visibility_cb.stateChanged.connect(
            lambda state: self.visibility_changed.emit(state == Qt.CheckState.Checked)
        )
        layout.addWidget(self.visibility_cb)
        
        # Layer name
        name_label = QLabel(layer.name)
        layout.addWidget(name_label)
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(layer.opacity * 100))
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_changed.emit(value / 100.0)
        )
        layout.addWidget(self.opacity_slider)
        
        self.setLayout(layout)

class LayerWidget(QWidget):
    """Widget for managing layers."""
    
    layer_added = pyqtSignal()
    layer_removed = pyqtSignal(int)
    layer_selected = pyqtSignal(int)
    
    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        
        layout = QVBoxLayout()
        
        # Layer list
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.on_layer_selected)
        layout.addWidget(self.layer_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_button = QPushButton("+")
        add_button.clicked.connect(self.add_layer)
        button_layout.addWidget(add_button)
        
        remove_button = QPushButton("-")
        remove_button.clicked.connect(self.remove_layer)
        button_layout.addWidget(remove_button)
        
        move_up_button = QPushButton("↑")
        move_up_button.clicked.connect(self.move_layer_up)
        button_layout.addWidget(move_up_button)
        
        move_down_button = QPushButton("↓")
        move_down_button.clicked.connect(self.move_layer_down)
        button_layout.addWidget(move_down_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Update the list
        self.update_layer_list()
    
    def update_layer_list(self):
        """Update the layer list widget."""
        self.layer_list.clear()
        for layer in reversed(self.layer_stack.layers):  # Top layer first
            item = QListWidgetItem()
            layer_widget = LayerItem(layer)
            item.setSizeHint(layer_widget.sizeHint())
            
            # Connect signals
            layer_widget.visibility_changed.connect(layer.set_visible)
            layer_widget.opacity_changed.connect(layer.set_opacity)
            
            self.layer_list.addItem(item)
            self.layer_list.setItemWidget(item, layer_widget)
    
    def add_layer(self):
        """Add a new layer."""
        layer = self.layer_stack.add_layer()
        
        # Create undo command
        command = LayerCommand(
            self.layer_stack,
            "Add Layer",
            undo_func=lambda: (
                self.layer_stack.remove_layer(len(self.layer_stack.layers) - 1),
                self.update_layer_list(),
                self.layer_removed.emit(len(self.layer_stack.layers))
            ),
            redo_func=lambda: (
                self.layer_stack.add_layer(layer),
                self.update_layer_list(),
                self.layer_added.emit()
            )
        )
        
        # Execute command
        self.layer_stack.canvas.command_stack.push(command)
    
    def remove_layer(self):
        """Remove the selected layer."""
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            # Convert UI row to stack index (reversed)
            stack_index = len(self.layer_stack.layers) - 1 - current_row
            layer = self.layer_stack.layers[stack_index]
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Remove Layer",
                undo_func=lambda: (
                    self.layer_stack.layers.insert(stack_index, layer),
                    self.update_layer_list(),
                    self.layer_added.emit()
                ),
                redo_func=lambda: (
                    self.layer_stack.remove_layer(stack_index),
                    self.update_layer_list(),
                    self.layer_removed.emit(stack_index)
                )
            )
            
            # Execute command
            self.layer_stack.canvas.command_stack.push(command)
    
    def move_layer_up(self):
        """Move the selected layer up."""
        current_row = self.layer_list.currentRow()
        if current_row > 0:
            # Convert UI rows to stack indices (reversed)
            stack_from = len(self.layer_stack.layers) - 1 - current_row
            stack_to = stack_from + 1
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Move Layer Up",
                undo_func=lambda: (
                    self.layer_stack.move_layer(stack_to, stack_from),
                    self.update_layer_list(),
                    self.layer_list.setCurrentRow(current_row)
                ),
                redo_func=lambda: (
                    self.layer_stack.move_layer(stack_from, stack_to),
                    self.update_layer_list(),
                    self.layer_list.setCurrentRow(current_row - 1)
                )
            )
            
            # Execute command
            self.layer_stack.canvas.command_stack.push(command)
    
    def move_layer_down(self):
        """Move the selected layer down."""
        current_row = self.layer_list.currentRow()
        if current_row < self.layer_list.count() - 1:
            # Convert UI rows to stack indices (reversed)
            stack_from = len(self.layer_stack.layers) - 1 - current_row
            stack_to = stack_from - 1
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Move Layer Down",
                undo_func=lambda: (
                    self.layer_stack.move_layer(stack_to, stack_from),
                    self.update_layer_list(),
                    self.layer_list.setCurrentRow(current_row)
                ),
                redo_func=lambda: (
                    self.layer_stack.move_layer(stack_from, stack_to),
                    self.update_layer_list(),
                    self.layer_list.setCurrentRow(current_row + 1)
                )
            )
            
            # Execute command
            self.layer_stack.canvas.command_stack.push(command)
    
    def on_layer_selected(self, row):
        """Handle layer selection."""
        if row >= 0:
            # Convert UI row to stack index (reversed)
            stack_index = len(self.layer_stack.layers) - 1 - row
            self.layer_stack.set_active_layer(stack_index)
            self.layer_selected.emit(stack_index) 