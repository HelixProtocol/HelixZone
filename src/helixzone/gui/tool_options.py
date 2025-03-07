from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QColorDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

class ToolOptionsWidget(QWidget):
    """Widget for controlling tool options."""
    
    def __init__(self, tool_manager, parent=None):
        super().__init__(parent)
        self.tool_manager = tool_manager
        
        layout = QVBoxLayout()
        
        # Brush Size
        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.update_size)
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_spin)
        layout.addLayout(size_layout)
        
        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.valueChanged.connect(self.update_opacity)
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_spin)
        layout.addLayout(opacity_layout)
        
        # Color Button
        self.color_button = QPushButton()
        self.color_button.setFixedSize(40, 40)
        self.current_color = QColor(0, 0, 0)
        self.update_color_button()
        self.color_button.clicked.connect(self.choose_color)
        layout.addWidget(self.color_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_size(self, size):
        """Update the tool size."""
        tool = self.tool_manager.get_current_tool()
        if tool:
            tool.size = size
            
    def update_opacity(self, opacity):
        """Update the tool opacity."""
        tool = self.tool_manager.get_current_tool()
        if tool:
            tool.opacity = opacity
            if hasattr(tool, 'color'):
                color = tool.color
                color.setAlphaF(opacity)
                tool.color = color
                
    def choose_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(
            self.current_color,
            self,
            "Choose Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )
        
        if color.isValid():
            self.current_color = color
            self.update_color_button()
            tool = self.tool_manager.get_current_tool()
            if tool and hasattr(tool, 'color'):
                tool.color = color
                
    def update_color_button(self):
        """Update the color button's appearance."""
        style = f"""
            QPushButton {{
                background-color: {self.current_color.name()};
                border: 2px solid #666666;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                border: 2px solid #999999;
            }}
        """
        self.color_button.setStyleSheet(style)
        
    def update_for_tool(self, tool_name):
        """Update the widget for the current tool."""
        if tool_name == 'eraser':
            self.color_button.setEnabled(False)
        else:
            self.color_button.setEnabled(True) 