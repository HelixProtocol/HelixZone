from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QColorDialog, QStackedWidget
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from typing import TYPE_CHECKING, Optional, cast
from ..core.tools import SelectionTool

if TYPE_CHECKING:
    from .main_window import MainWindow

class ToolOptionsWidget(QWidget):
    """Widget for controlling tool options."""
    
    def __init__(self, parent: Optional['MainWindow'] = None):
        super().__init__(parent)
        self.main_window = cast('MainWindow', parent)
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
        # Main layout
        layout = QVBoxLayout()
        
        # Create stacked widget for different tool options
        self.stacked_widget = QStackedWidget()
        
        # Brush/Eraser options
        brush_widget = QWidget()
        brush_layout = QVBoxLayout()
        
        # Brush Size
        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.update_size)
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_spin)
        brush_layout.addLayout(size_layout)
        
        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.valueChanged.connect(self.update_opacity)
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_spin)
        brush_layout.addLayout(opacity_layout)
        
        # Color Button
        self.current_color = QColor(0, 0, 0)
        self.color_button = QPushButton()
        self.color_button.setFixedSize(40, 40)
        self.update_color_button()
        self.color_button.clicked.connect(self.choose_color)
        brush_layout.addWidget(self.color_button)
        
        brush_layout.addStretch()
        brush_widget.setLayout(brush_layout)
        
        # Selection options
        selection_widget = QWidget()
        selection_layout = QVBoxLayout()
        
        # Selection mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_label = QLabel("New Selection")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_label)
        selection_layout.addLayout(mode_layout)
        
        # Feather radius
        feather_layout = QHBoxLayout()
        feather_label = QLabel("Feather:")
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 100)
        self.feather_spin.setValue(0)
        self.feather_spin.valueChanged.connect(self.update_feather)
        feather_layout.addWidget(feather_label)
        feather_layout.addWidget(self.feather_spin)
        selection_layout.addLayout(feather_layout)
        
        # Selection actions
        actions_layout = QHBoxLayout()
        
        # Cut button
        self.cut_button = QPushButton("Cut")
        self.cut_button.clicked.connect(self.cut_selection)
        actions_layout.addWidget(self.cut_button)
        
        # Paste button
        self.paste_button = QPushButton("Paste")
        self.paste_button.clicked.connect(self.paste_selection)
        actions_layout.addWidget(self.paste_button)
        
        # Recolor button
        self.recolor_button = QPushButton("Recolor")
        self.recolor_button.clicked.connect(self.recolor_selection)
        actions_layout.addWidget(self.recolor_button)
        
        selection_layout.addLayout(actions_layout)
        
        selection_layout.addStretch()
        selection_widget.setLayout(selection_layout)
        
        # Add widgets to stacked widget
        self.stacked_widget.addWidget(brush_widget)      # Index 0: Brush/Eraser
        self.stacked_widget.addWidget(selection_widget)  # Index 1: Selection tools
        
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
    
    def setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts for selection operations."""
        # Cut shortcut (Ctrl+X)
        cut_shortcut = QShortcut(QKeySequence.StandardKey.Cut, self)
        cut_shortcut.activated.connect(self.cut_selection)
        
        # Copy shortcut (Ctrl+C)
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy_shortcut.activated.connect(self.copy_selection)
        
        # Paste shortcut (Ctrl+V)
        paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self)
        paste_shortcut.activated.connect(self.start_paste)
    
    def get_current_tool(self):
        """Get the current tool from the tool manager."""
        if self.main_window and hasattr(self.main_window, 'canvas_view'):
            return self.main_window.canvas_view.canvas.tool_manager.get_current_tool()
        return None
    
    def update_size(self, size: int) -> None:
        """Update the tool size."""
        tool = self.get_current_tool()
        if tool and hasattr(tool, 'size'):
            tool.size = size
    
    def update_opacity(self, opacity: int) -> None:
        """Update the tool opacity."""
        tool = self.get_current_tool()
        if tool and hasattr(tool, 'opacity'):
            tool.opacity = opacity / 100.0
    
    def update_feather(self, radius: int) -> None:
        """Update the selection feather radius."""
        tool = self.get_current_tool()
        if tool and hasattr(tool, 'feather_radius'):
            tool.feather_radius = radius
    
    def choose_color(self) -> None:
        """Open color picker dialog."""
        color = QColorDialog.getColor(self.current_color, self, "Choose Color")
        if color.isValid():
            self.current_color = color
            self.update_color_button()
            # Update tool color
            tool = self.get_current_tool()
            if tool and hasattr(tool, 'color'):
                tool.color = color
    
    def update_color_button(self) -> None:
        """Update the color button appearance."""
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
    
    def update_for_tool(self, tool_name: str) -> None:
        """Update the widget for the selected tool."""
        if tool_name in ['brush', 'eraser']:
            self.stacked_widget.setCurrentIndex(0)  # Show brush options
            tool = self.get_current_tool()
            if tool:
                if hasattr(tool, 'size'):
                    self.size_spin.setValue(int(tool.size))
                if hasattr(tool, 'opacity'):
                    self.opacity_spin.setValue(int(tool.opacity * 100))
                if hasattr(tool, 'color'):
                    self.current_color = tool.color
                    self.update_color_button()
        elif tool_name in ['rectangle_selection', 'ellipse_selection', 'lasso_selection']:
            self.stacked_widget.setCurrentIndex(1)  # Show selection options
            tool = self.get_current_tool()
            if tool and hasattr(tool, 'feather_radius'):
                self.feather_spin.setValue(tool.feather_radius)

    def cut_selection(self) -> None:
        """Cut the selected area."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            tool.cut_selection()

    def recolor_selection(self) -> None:
        """Recolor the selected area."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            # Use the current color
            tool.recolor_selection(self.current_color) 

    def copy_selection(self) -> None:
        """Copy the selected area."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            tool.copy_selection()

    def start_paste(self) -> None:
        """Start floating paste mode."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            # Get the current mouse position in canvas coordinates
            canvas = self.main_window.canvas_view.canvas
            cursor_pos = canvas.mapFromGlobal(self.cursor().pos())
            transformed_pos = canvas.get_transformed_pos(cursor_pos)
            tool.start_floating_paste(transformed_pos)

    def paste_selection(self) -> None:
        """Paste the previously cut content."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            # Start floating paste mode at current mouse position
            self.start_paste()