from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QColorDialog, QStackedWidget
)
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from typing import TYPE_CHECKING, Optional, cast
from ..core.tools import SelectionTool, Tool, MagneticLassoSelection

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
        
        # Magnetic Lasso options
        magnetic_lasso_widget = QWidget()
        magnetic_layout = QVBoxLayout()
        
        # Edge Width
        edge_width_layout = QHBoxLayout()
        edge_width_label = QLabel("Edge Width:")
        self.edge_width_spin = QSpinBox()
        self.edge_width_spin.setRange(1, 50)
        self.edge_width_spin.setValue(10)
        self.edge_width_spin.valueChanged.connect(self.update_edge_width)
        edge_width_layout.addWidget(edge_width_label)
        edge_width_layout.addWidget(self.edge_width_spin)
        magnetic_layout.addLayout(edge_width_layout)
        
        # Edge Contrast
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Edge Contrast:")
        self.contrast_spin = QSpinBox()
        self.contrast_spin.setRange(1, 255)
        self.contrast_spin.setValue(50)
        self.contrast_spin.valueChanged.connect(self.update_edge_contrast)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_spin)
        magnetic_layout.addLayout(contrast_layout)
        
        # Anchor Spacing
        anchor_layout = QHBoxLayout()
        anchor_label = QLabel("Anchor Spacing:")
        self.anchor_spin = QSpinBox()
        self.anchor_spin.setRange(5, 100)
        self.anchor_spin.setValue(20)
        self.anchor_spin.valueChanged.connect(self.update_anchor_spacing)
        anchor_layout.addWidget(anchor_label)
        anchor_layout.addWidget(self.anchor_spin)
        magnetic_layout.addLayout(anchor_layout)
        
        # Add selection options to magnetic lasso widget
        magnetic_layout.addWidget(selection_widget)
        magnetic_lasso_widget.setLayout(magnetic_layout)
        
        # Add widgets to stacked widget
        self.stacked_widget.addWidget(brush_widget)      # Index 0: Brush/Eraser
        self.stacked_widget.addWidget(selection_widget)  # Index 1: Selection tools
        self.stacked_widget.addWidget(magnetic_lasso_widget)  # Index 2: Magnetic Lasso
        
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
    
    def get_current_tool(self) -> Optional[Tool]:
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
        tool = self.get_current_tool()
        if not tool:
            return

        if tool_name in ['brush', 'eraser']:
            self.stacked_widget.setCurrentIndex(0)  # Show brush options
            if hasattr(tool, 'size'):
                self.size_spin.setValue(int(tool.size))
            if hasattr(tool, 'opacity'):
                self.opacity_spin.setValue(int(tool.opacity * 100))
            if hasattr(tool, 'color'):
                self.current_color = tool.color
                self.update_color_button()
        elif tool_name in ['rectangle_selection', 'ellipse_selection', 'lasso_selection']:
            self.stacked_widget.setCurrentIndex(1)  # Show selection options
            if hasattr(tool, 'feather_radius'):
                self.feather_spin.setValue(tool.feather_radius)
        elif tool_name == 'magnetic_lasso':
            self.stacked_widget.setCurrentIndex(2)  # Show magnetic lasso options
            if isinstance(tool, MagneticLassoSelection):
                self.edge_width_spin.setValue(tool.edge_width)
                self.contrast_spin.setValue(tool.edge_contrast)
                self.anchor_spin.setValue(tool.anchor_spacing)
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
        try:
            tool = self.get_current_tool()
            if not isinstance(tool, SelectionTool):
                print("Current tool is not a selection tool")
                return
                
            if not tool.clipboard_image:
                print("No content to paste")
                return
                
            # Get the canvas
            if not self.main_window or not hasattr(self.main_window, 'canvas_view'):
                print("Canvas not available")
                return
                
            canvas = self.main_window.canvas_view.canvas
            if not canvas:
                print("Invalid canvas")
                return
                
            # Get current cursor position
            cursor_pos = canvas.mapFromGlobal(self.cursor().pos())
            if cursor_pos is None or cursor_pos.isNull():
                print("Could not get cursor position")
                return
                
            # Convert cursor position to QPointF for proper transformation
            cursor_pos_f = QPointF(cursor_pos.x(), cursor_pos.y())
            
            # Get transformed position
            transformed_pos = canvas.get_transformed_pos(cursor_pos_f)
            if transformed_pos is None or transformed_pos.isNull():
                print("Could not transform cursor position")
                return
                
            # Ensure transformed position is within canvas bounds
            if (transformed_pos.x() < 0 or transformed_pos.x() >= canvas.width() or
                transformed_pos.y() < 0 or transformed_pos.y() >= canvas.height()):
                print("Cursor position outside canvas bounds")
                return
                
            # Start floating paste mode
            tool.start_floating_paste(transformed_pos)
            
        except Exception as e:
            print(f"Error in start_paste: {e}")

    def paste_selection(self) -> None:
        """Paste the previously cut content."""
        tool = self.get_current_tool()
        if isinstance(tool, SelectionTool):
            # Start floating paste mode at current mouse position
            self.start_paste()

    def update_edge_width(self, value: int) -> None:
        """Update the magnetic lasso edge width."""
        tool = self.get_current_tool()
        if isinstance(tool, MagneticLassoSelection):
            tool.edge_width = value

    def update_edge_contrast(self, value: int) -> None:
        """Update the magnetic lasso edge contrast."""
        tool = self.get_current_tool()
        if isinstance(tool, MagneticLassoSelection):
            tool.edge_contrast = value

    def update_anchor_spacing(self, value: int) -> None:
        """Update the magnetic lasso anchor spacing."""
        tool = self.get_current_tool()
        if isinstance(tool, MagneticLassoSelection):
            tool.anchor_spacing = value