from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QSlider,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QMenu, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QColor
from ..core.layer import Layer, AdjustmentLayer
from ..core.commands import LayerCommand

class LayerItem(QWidget):
    """Widget representing a single layer in the layer list."""
    
    visibility_changed = pyqtSignal(bool)
    opacity_changed = pyqtSignal(float)
    edit_adjustment_requested = pyqtSignal()
    
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
        
        # Icon for adjustment layers
        if isinstance(layer, AdjustmentLayer):
            icon_label = QLabel()
            icon_label.setPixmap(self._get_adjustment_icon(layer.adjustment_type))
            layout.addWidget(icon_label)
        
        # Layer name
        name_label = QLabel(layer.name)
        name_label.setStyleSheet("font-weight: bold;" if isinstance(layer, AdjustmentLayer) else "")
        layout.addWidget(name_label)
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(layer.opacity * 100))
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_changed.emit(value / 100.0)
        )
        layout.addWidget(self.opacity_slider)
        
        # Edit button for adjustment layers
        if isinstance(layer, AdjustmentLayer):
            edit_button = QPushButton("Edit")
            edit_button.setFixedWidth(40)
            edit_button.clicked.connect(self.edit_adjustment_requested.emit)
            layout.addWidget(edit_button)
        
        self.setLayout(layout)
    
    def _get_adjustment_icon(self, adjustment_type: str) -> QPixmap:
        """Generate an icon for the adjustment type."""
        icon = QPixmap(16, 16)
        icon.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(icon)
        
        # Set color based on adjustment type
        color_map = {
            "brightness_contrast": QColor(255, 204, 0),
            "levels": QColor(0, 204, 255),
            "curves": QColor(204, 0, 255),
            "hue_saturation": QColor(204, 255, 0),
            "color_balance": QColor(255, 0, 204),
            "threshold": QColor(0, 0, 0)
        }
        
        color = color_map.get(adjustment_type, QColor(128, 128, 128))
        
        # Draw a colored circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(2, 2, 12, 12)
        
        painter.end()
        return icon


class AdjustmentDialog(QDialog):
    """Dialog for editing adjustment layer parameters."""
    
    def __init__(self, adjustment_layer, parent=None):
        super().__init__(parent)
        self.adjustment_layer = adjustment_layer
        self.setWindowTitle(f"Edit {adjustment_layer.name}")
        self.resize(400, 300)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create parameter controls based on adjustment type
        if adjustment_layer.adjustment_type == "brightness_contrast":
            self._create_brightness_contrast_controls(layout)
        elif adjustment_layer.adjustment_type == "levels":
            self._create_levels_controls(layout)
        elif adjustment_layer.adjustment_type == "curves":
            self._create_curves_controls(layout)
        elif adjustment_layer.adjustment_type == "hue_saturation":
            self._create_hue_saturation_controls(layout)
        elif adjustment_layer.adjustment_type == "color_balance":
            self._create_color_balance_controls(layout)
        elif adjustment_layer.adjustment_type == "threshold":
            self._create_threshold_controls(layout)
        
        # Add button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_brightness_contrast_controls(self, layout):
        """Create controls for brightness/contrast adjustment."""
        group = QGroupBox("Brightness/Contrast")
        form_layout = QFormLayout(group)
        
        # Brightness slider
        brightness = self.adjustment_layer.parameters.get("brightness", 0)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(brightness)
        self.brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.brightness_slider.setTickInterval(20)
        
        # Brightness value label
        self.brightness_value = QLabel(f"{brightness}")
        self.brightness_slider.valueChanged.connect(
            lambda value: (
                self.brightness_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("brightness", value)
            )
        )
        
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_value)
        form_layout.addRow("Brightness:", brightness_layout)
        
        # Contrast slider
        contrast = self.adjustment_layer.parameters.get("contrast", 0)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(contrast)
        self.contrast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.contrast_slider.setTickInterval(20)
        
        # Contrast value label
        self.contrast_value = QLabel(f"{contrast}")
        self.contrast_slider.valueChanged.connect(
            lambda value: (
                self.contrast_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("contrast", value)
            )
        )
        
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_value)
        form_layout.addRow("Contrast:", contrast_layout)
        
        layout.addWidget(group)
    
    def _create_levels_controls(self, layout):
        """Create controls for levels adjustment."""
        group = QGroupBox("Levels")
        form_layout = QFormLayout(group)
        
        # Black point slider
        black_point = self.adjustment_layer.parameters.get("black_point", 0)
        self.black_point_slider = QSlider(Qt.Orientation.Horizontal)
        self.black_point_slider.setRange(0, 255)
        self.black_point_slider.setValue(black_point)
        
        self.black_point_value = QLabel(f"{black_point}")
        self.black_point_slider.valueChanged.connect(
            lambda value: (
                self.black_point_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("black_point", value)
            )
        )
        
        black_layout = QHBoxLayout()
        black_layout.addWidget(self.black_point_slider)
        black_layout.addWidget(self.black_point_value)
        form_layout.addRow("Black Point:", black_layout)
        
        # White point slider
        white_point = self.adjustment_layer.parameters.get("white_point", 255)
        self.white_point_slider = QSlider(Qt.Orientation.Horizontal)
        self.white_point_slider.setRange(0, 255)
        self.white_point_slider.setValue(white_point)
        
        self.white_point_value = QLabel(f"{white_point}")
        self.white_point_slider.valueChanged.connect(
            lambda value: (
                self.white_point_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("white_point", value)
            )
        )
        
        white_layout = QHBoxLayout()
        white_layout.addWidget(self.white_point_slider)
        white_layout.addWidget(self.white_point_value)
        form_layout.addRow("White Point:", white_layout)
        
        # Mid point (gamma) spinner
        mid_point = self.adjustment_layer.parameters.get("mid_point", 1.0)
        self.mid_point_spinner = QDoubleSpinBox()
        self.mid_point_spinner.setRange(0.1, 10.0)
        self.mid_point_spinner.setSingleStep(0.1)
        self.mid_point_spinner.setValue(mid_point)
        self.mid_point_spinner.valueChanged.connect(
            lambda value: self.adjustment_layer.set_parameter("mid_point", value)
        )
        
        form_layout.addRow("Mid Point (Gamma):", self.mid_point_spinner)
        
        layout.addWidget(group)
    
    def _create_curves_controls(self, layout):
        """Create controls for curves adjustment."""
        # Note: A full curves implementation would include a curve editor widget.
        # For simplicity, we'll use basic controls here.
        group = QGroupBox("Curves")
        form_layout = QFormLayout(group)
        
        # Channel selector
        curve_type = self.adjustment_layer.parameters.get("curve_type", "rgb")
        self.curve_type_combo = QComboBox()
        self.curve_type_combo.addItems(["RGB", "Red", "Green", "Blue"])
        
        # Map curve type to index
        type_to_index = {"rgb": 0, "r": 1, "g": 2, "b": 3}
        self.curve_type_combo.setCurrentIndex(type_to_index.get(curve_type, 0))
        
        self.curve_type_combo.currentIndexChanged.connect(
            lambda index: self.adjustment_layer.set_parameter(
                "curve_type", ["rgb", "r", "g", "b"][index]
            )
        )
        
        form_layout.addRow("Channel:", self.curve_type_combo)
        
        # Add message about curve editing
        message = QLabel("Advanced curve editing will be available in a future update.")
        message.setWordWrap(True)
        form_layout.addRow(message)
        
        layout.addWidget(group)
    
    def _create_hue_saturation_controls(self, layout):
        """Create controls for hue/saturation adjustment."""
        group = QGroupBox("Hue/Saturation")
        form_layout = QFormLayout(group)
        
        # Hue slider
        hue = self.adjustment_layer.parameters.get("hue", 0)
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(hue)
        
        self.hue_value = QLabel(f"{hue}")
        self.hue_slider.valueChanged.connect(
            lambda value: (
                self.hue_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("hue", value)
            )
        )
        
        hue_layout = QHBoxLayout()
        hue_layout.addWidget(self.hue_slider)
        hue_layout.addWidget(self.hue_value)
        form_layout.addRow("Hue:", hue_layout)
        
        # Saturation slider
        saturation = self.adjustment_layer.parameters.get("saturation", 0)
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(saturation)
        
        self.saturation_value = QLabel(f"{saturation}")
        self.saturation_slider.valueChanged.connect(
            lambda value: (
                self.saturation_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("saturation", value)
            )
        )
        
        saturation_layout = QHBoxLayout()
        saturation_layout.addWidget(self.saturation_slider)
        saturation_layout.addWidget(self.saturation_value)
        form_layout.addRow("Saturation:", saturation_layout)
        
        # Lightness slider
        lightness = self.adjustment_layer.parameters.get("lightness", 0)
        self.lightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.lightness_slider.setRange(-100, 100)
        self.lightness_slider.setValue(lightness)
        
        self.lightness_value = QLabel(f"{lightness}")
        self.lightness_slider.valueChanged.connect(
            lambda value: (
                self.lightness_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("lightness", value)
            )
        )
        
        lightness_layout = QHBoxLayout()
        lightness_layout.addWidget(self.lightness_slider)
        lightness_layout.addWidget(self.lightness_value)
        form_layout.addRow("Lightness:", lightness_layout)
        
        layout.addWidget(group)
    
    def _create_color_balance_controls(self, layout):
        """Create controls for color balance adjustment."""
        # Shadows group
        shadows_group = QGroupBox("Shadows")
        shadows_layout = QFormLayout(shadows_group)
        shadows = self.adjustment_layer.parameters.get("shadows", [0, 0, 0])
        
        # Red slider
        self.shadows_r_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadows_r_slider.setRange(-100, 100)
        self.shadows_r_slider.setValue(shadows[0])
        
        self.shadows_r_value = QLabel(f"{shadows[0]}")
        self.shadows_r_slider.valueChanged.connect(
            lambda value: (
                self.shadows_r_value.setText(f"{value}"),
                self._update_color_balance("shadows", 0, value)
            )
        )
        
        shadows_r_layout = QHBoxLayout()
        shadows_r_layout.addWidget(self.shadows_r_slider)
        shadows_r_layout.addWidget(self.shadows_r_value)
        shadows_layout.addRow("Red:", shadows_r_layout)
        
        # Green slider
        self.shadows_g_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadows_g_slider.setRange(-100, 100)
        self.shadows_g_slider.setValue(shadows[1])
        
        self.shadows_g_value = QLabel(f"{shadows[1]}")
        self.shadows_g_slider.valueChanged.connect(
            lambda value: (
                self.shadows_g_value.setText(f"{value}"),
                self._update_color_balance("shadows", 1, value)
            )
        )
        
        shadows_g_layout = QHBoxLayout()
        shadows_g_layout.addWidget(self.shadows_g_slider)
        shadows_g_layout.addWidget(self.shadows_g_value)
        shadows_layout.addRow("Green:", shadows_g_layout)
        
        # Blue slider
        self.shadows_b_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadows_b_slider.setRange(-100, 100)
        self.shadows_b_slider.setValue(shadows[2])
        
        self.shadows_b_value = QLabel(f"{shadows[2]}")
        self.shadows_b_slider.valueChanged.connect(
            lambda value: (
                self.shadows_b_value.setText(f"{value}"),
                self._update_color_balance("shadows", 2, value)
            )
        )
        
        shadows_b_layout = QHBoxLayout()
        shadows_b_layout.addWidget(self.shadows_b_slider)
        shadows_b_layout.addWidget(self.shadows_b_value)
        shadows_layout.addRow("Blue:", shadows_b_layout)
        
        layout.addWidget(shadows_group)
        
        # Midtones group (similar structure)
        midtones_group = QGroupBox("Midtones")
        midtones_layout = QFormLayout(midtones_group)
        midtones = self.adjustment_layer.parameters.get("midtones", [0, 0, 0])
        
        # Red, Green, Blue sliders (similar to shadows)
        # ... (skipping implementation for brevity)
        
        layout.addWidget(midtones_group)
        
        # Highlights group (similar structure)
        highlights_group = QGroupBox("Highlights")
        highlights_layout = QFormLayout(highlights_group)
        highlights = self.adjustment_layer.parameters.get("highlights", [0, 0, 0])
        
        # Red, Green, Blue sliders (similar to shadows)
        # ... (skipping implementation for brevity)
        
        layout.addWidget(highlights_group)
    
    def _create_threshold_controls(self, layout):
        """Create controls for threshold adjustment."""
        group = QGroupBox("Threshold")
        form_layout = QFormLayout(group)
        
        threshold = self.adjustment_layer.parameters.get("threshold", 127)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(threshold)
        
        self.threshold_value = QLabel(f"{threshold}")
        self.threshold_slider.valueChanged.connect(
            lambda value: (
                self.threshold_value.setText(f"{value}"),
                self.adjustment_layer.set_parameter("threshold", value)
            )
        )
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value)
        form_layout.addRow("Threshold:", threshold_layout)
        
        layout.addWidget(group)
    
    def _update_color_balance(self, group, channel, value):
        """Update a color balance parameter."""
        # Get current values
        values = self.adjustment_layer.parameters.get(group, [0, 0, 0]).copy()
        
        # Update the specified channel
        values[channel] = value
        
        # Update the parameter
        self.adjustment_layer.set_parameter(group, values)


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
        add_button.setToolTip("Add Layer")
        add_button.clicked.connect(self._show_add_menu)
        button_layout.addWidget(add_button)
        
        remove_button = QPushButton("-")
        remove_button.setToolTip("Remove Layer")
        remove_button.clicked.connect(self.remove_layer)
        button_layout.addWidget(remove_button)
        
        move_up_button = QPushButton("↑")
        move_up_button.setToolTip("Move Layer Up")
        move_up_button.clicked.connect(self.move_layer_up)
        button_layout.addWidget(move_up_button)
        
        move_down_button = QPushButton("↓")
        move_down_button.setToolTip("Move Layer Down")
        move_down_button.clicked.connect(self.move_layer_down)
        button_layout.addWidget(move_down_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Update the list
        self.update_layer_list()
    
    def _show_add_menu(self):
        """Show menu for adding different layer types."""
        menu = QMenu(self)
        
        # Add normal layer
        add_normal = menu.addAction("New Layer")
        add_normal.triggered.connect(self.add_layer)
        
        # Add submenu for adjustment layers
        adj_menu = menu.addMenu("Adjustment Layer")
        
        # Add different adjustment types
        adjustments = [
            ("Brightness/Contrast", "brightness_contrast"),
            ("Levels", "levels"),
            ("Curves", "curves"),
            ("Hue/Saturation", "hue_saturation"),
            ("Color Balance", "color_balance"),
            ("Threshold", "threshold")
        ]
        
        for name, adj_type in adjustments:
            action = adj_menu.addAction(name)
            action.triggered.connect(lambda checked=False, adj_type=adj_type: self.add_adjustment_layer(adj_type))
        
        # Show the menu
        menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))
    
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
            
            # Connect edit signal for adjustment layers
            if isinstance(layer, AdjustmentLayer):
                layer_widget.edit_adjustment_requested.connect(
                    lambda layer=layer: self._edit_adjustment_layer(layer)
                )
            
            self.layer_list.addItem(item)
            self.layer_list.setItemWidget(item, layer_widget)
    
    def _edit_adjustment_layer(self, layer):
        """Open dialog to edit adjustment layer parameters."""
        dialog = AdjustmentDialog(layer, self)
        dialog.exec()
    
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
        
        # Update the list
        self.update_layer_list()
    
    def add_adjustment_layer(self, adjustment_type):
        """Add a new adjustment layer."""
        layer = self.layer_stack.add_adjustment_layer(adjustment_type)
        
        # Define the undo and redo functions
        def undo_function():
            self.layer_stack.remove_layer(len(self.layer_stack.layers) - 1)
            self.update_layer_list()
            self.layer_removed.emit(len(self.layer_stack.layers))
        
        def redo_function():
            self.layer_stack.layers.append(layer)
            self.layer_stack.active_layer_index = len(self.layer_stack.layers) - 1
            self.update_layer_list()
            self.layer_added.emit()
        
        # Create undo command
        command = LayerCommand(
            self.layer_stack,
            f"Add {layer.name}",
            undo_func=undo_function,
            redo_func=redo_function
        )
        
        # Execute command
        self.layer_stack.canvas.command_stack.push(command)
        
        # Update the list
        self.update_layer_list()
        
        # Open edit dialog for the new adjustment layer
        self._edit_adjustment_layer(layer)
    
    def remove_layer(self):
        """Remove the selected layer."""
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            # Convert UI row to stack index (reversed)
            stack_index = len(self.layer_stack.layers) - 1 - current_row
            layer = self.layer_stack.layers[stack_index]
            
            # Define undo and redo functions
            def undo_function():
                self.layer_stack.layers.insert(stack_index, layer)
                self.update_layer_list()
                self.layer_added.emit()
            
            def redo_function():
                self.layer_stack.remove_layer(stack_index)
                self.update_layer_list()
                self.layer_removed.emit(stack_index)
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Remove Layer",
                undo_func=undo_function,
                redo_func=redo_function
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
            
            # Define undo and redo functions
            def undo_function():
                self.layer_stack.move_layer(stack_to, stack_from)
                self.update_layer_list()
                self.layer_list.setCurrentRow(current_row)
            
            def redo_function():
                self.layer_stack.move_layer(stack_from, stack_to)
                self.update_layer_list()
                self.layer_list.setCurrentRow(current_row - 1)
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Move Layer Up",
                undo_func=undo_function,
                redo_func=redo_function
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
            
            # Define undo and redo functions
            def undo_function():
                self.layer_stack.move_layer(stack_to, stack_from)
                self.update_layer_list()
                self.layer_list.setCurrentRow(current_row)
            
            def redo_function():
                self.layer_stack.move_layer(stack_from, stack_to)
                self.update_layer_list()
                self.layer_list.setCurrentRow(current_row + 1)
            
            # Create undo command
            command = LayerCommand(
                self.layer_stack,
                "Move Layer Down",
                undo_func=undo_function,
                redo_func=redo_function
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