"""
Zoom controls and navigation helpers for enhanced image navigation.

This module provides widgets and utilities for zooming and navigating
large images, including a zoom slider, minimap navigator, and presets.
"""

import logging
import math
from typing import Optional, Callable, List, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSlider, QComboBox, QSizePolicy, QFrame, QRubberBand,
    QScrollArea
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QMouseEvent, QPaintEvent

logger = logging.getLogger(__name__)


class ZoomLevels:
    """Helper class for managing zoom presets and conversions."""
    
    # Common zoom levels as percentages
    PRESETS = [10, 25, 33, 50, 66, 75, 100, 125, 150, 200, 300, 400, 600, 800, 1200, 1600]
    
    # Zoom bounds
    MIN_ZOOM = 5  # 5%
    MAX_ZOOM = 1600  # 1600%
    
    @staticmethod
    def percent_to_scale(percent: float) -> float:
        """Convert percentage to scale factor (e.g., 100% -> 1.0)."""
        return percent / 100.0
    
    @staticmethod
    def scale_to_percent(scale: float) -> float:
        """Convert scale factor to percentage (e.g., 1.0 -> 100%)."""
        return scale * 100.0
    
    @staticmethod
    def find_nearest_preset(percent: float) -> float:
        """Find the nearest preset zoom level.
        
        Args:
            percent: Zoom percentage to match
            
        Returns:
            The nearest preset zoom level
        """
        if percent <= ZoomLevels.MIN_ZOOM:
            return ZoomLevels.MIN_ZOOM
        if percent >= ZoomLevels.MAX_ZOOM:
            return ZoomLevels.MAX_ZOOM
            
        # Find nearest preset
        nearest = ZoomLevels.PRESETS[0]
        min_diff = abs(percent - nearest)
        
        for preset in ZoomLevels.PRESETS:
            diff = abs(percent - preset)
            if diff < min_diff:
                min_diff = diff
                nearest = preset
                
        return nearest
    
    @staticmethod
    def get_zoom_in_level(current_percent: float) -> float:
        """Get the next zoom level when zooming in.
        
        Args:
            current_percent: Current zoom percentage
            
        Returns:
            Next zoom level percentage
        """
        if current_percent >= ZoomLevels.MAX_ZOOM:
            return ZoomLevels.MAX_ZOOM
            
        # Find next preset
        for preset in ZoomLevels.PRESETS:
            if preset > current_percent:
                return preset
                
        # If no preset is larger, increase by 10%
        next_level = current_percent * 1.1
        return min(next_level, ZoomLevels.MAX_ZOOM)
    
    @staticmethod
    def get_zoom_out_level(current_percent: float) -> float:
        """Get the next zoom level when zooming out.
        
        Args:
            current_percent: Current zoom percentage
            
        Returns:
            Next zoom level percentage
        """
        if current_percent <= ZoomLevels.MIN_ZOOM:
            return ZoomLevels.MIN_ZOOM
            
        # Find previous preset
        for preset in reversed(ZoomLevels.PRESETS):
            if preset < current_percent:
                return preset
                
        # If no preset is smaller, decrease by 10%
        next_level = current_percent * 0.9
        return max(next_level, ZoomLevels.MIN_ZOOM)


class ZoomSlider(QWidget):
    """A slider widget for controlling zoom level."""
    
    zoom_changed = pyqtSignal(float)  # Emitted when zoom percentage changes
    
    def __init__(self, parent=None, initial_zoom=100):
        """Initialize the zoom slider.
        
        Args:
            parent: Parent widget
            initial_zoom: Initial zoom percentage
        """
        super().__init__(parent)
        
        # Current zoom percentage
        self._zoom_percent = initial_zoom
        
        # Set up layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Zoom out button
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(24, 24)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        layout.addWidget(self.zoom_out_button)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self._convert_zoom_to_slider(initial_zoom))
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider)
        
        # Zoom in button
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedSize(24, 24)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        layout.addWidget(self.zoom_in_button)
        
        # Zoom percentage display and combo
        self.zoom_combo = QComboBox()
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setFixedWidth(80)
        
        # Add preset zoom levels
        for preset in ZoomLevels.PRESETS:
            self.zoom_combo.addItem(f"{preset}%")
            
        # Set initial value
        self.zoom_combo.setCurrentText(f"{initial_zoom}%")
        
        # Connect signals
        self.zoom_combo.currentTextChanged.connect(self._on_combo_text_changed)
        self.zoom_combo.activated.connect(self._on_combo_activated)
        
        layout.addWidget(self.zoom_combo)
        
        # Fit button
        self.fit_button = QPushButton("Fit")
        self.fit_button.setFixedWidth(40)
        self.fit_button.clicked.connect(lambda: self.zoom_changed.emit(0))
        layout.addWidget(self.fit_button)
        
        # 100% button
        self.actual_button = QPushButton("100%")
        self.actual_button.setFixedWidth(50)
        self.actual_button.clicked.connect(lambda: self.set_zoom(100))
        layout.addWidget(self.actual_button)
        
        # Update UI
        self._update_ui()
    
    def set_zoom(self, percent: float) -> None:
        """Set the zoom level.
        
        Args:
            percent: Zoom percentage
        """
        # Clamp to valid range
        percent = max(ZoomLevels.MIN_ZOOM, min(ZoomLevels.MAX_ZOOM, percent))
        
        if percent != self._zoom_percent:
            self._zoom_percent = percent
            
            # Update UI without triggering signals
            self._update_ui()
            
            # Emit signal
            self.zoom_changed.emit(percent)
    
    def get_zoom(self) -> float:
        """Get the current zoom percentage."""
        return self._zoom_percent
    
    def zoom_in(self) -> None:
        """Zoom in to the next preset level."""
        next_zoom = ZoomLevels.get_zoom_in_level(self._zoom_percent)
        self.set_zoom(next_zoom)
    
    def zoom_out(self) -> None:
        """Zoom out to the previous preset level."""
        next_zoom = ZoomLevels.get_zoom_out_level(self._zoom_percent)
        self.set_zoom(next_zoom)
    
    def _on_slider_value_changed(self, value: int) -> None:
        """Handle slider value changes."""
        zoom = self._convert_slider_to_zoom(value)
        
        # Only update if the zoom level has actually changed
        if abs(zoom - self._zoom_percent) > 0.1:
            self.set_zoom(zoom)
    
    def _on_combo_text_changed(self, text: str) -> None:
        """Handle combo box text changes."""
        # Parse percentage value
        try:
            text = text.replace("%", "").strip()
            percent = float(text)
            
            # Only update if the zoom level has actually changed
            if abs(percent - self._zoom_percent) > 0.1:
                self.set_zoom(percent)
                
        except ValueError:
            # Restore current value on invalid input
            self.zoom_combo.setCurrentText(f"{self._zoom_percent:.0f}%")
    
    def _on_combo_activated(self, index: int) -> None:
        """Handle combo box item activation."""
        text = self.zoom_combo.currentText()
        
        # Parse percentage value
        try:
            text = text.replace("%", "").strip()
            percent = float(text)
            self.set_zoom(percent)
            
        except ValueError:
            # Restore current value on invalid input
            self.zoom_combo.setCurrentText(f"{self._zoom_percent:.0f}%")
    
    def _update_ui(self) -> None:
        """Update UI components to reflect current zoom level."""
        # Update slider (block signals to prevent feedback loop)
        self.slider.blockSignals(True)
        self.slider.setValue(self._convert_zoom_to_slider(self._zoom_percent))
        self.slider.blockSignals(False)
        
        # Update combo box
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(f"{self._zoom_percent:.0f}%")
        self.zoom_combo.blockSignals(False)
        
        # Update button states
        self.zoom_in_button.setEnabled(self._zoom_percent < ZoomLevels.MAX_ZOOM)
        self.zoom_out_button.setEnabled(self._zoom_percent > ZoomLevels.MIN_ZOOM)
    
    def _convert_zoom_to_slider(self, zoom_percent: float) -> int:
        """Convert zoom percentage to slider value using logarithmic scale.
        
        Args:
            zoom_percent: Zoom percentage
            
        Returns:
            Slider value (0-100)
        """
        # Use logarithmic scale to give more precision to smaller zoom levels
        min_log = math.log(ZoomLevels.MIN_ZOOM)
        max_log = math.log(ZoomLevels.MAX_ZOOM)
        
        zoom_log = math.log(max(ZoomLevels.MIN_ZOOM, zoom_percent))
        
        # Convert to 0-100 range
        slider_value = (zoom_log - min_log) / (max_log - min_log) * 100
        return max(0, min(100, int(slider_value)))
    
    def _convert_slider_to_zoom(self, slider_value: int) -> float:
        """Convert slider value to zoom percentage using logarithmic scale.
        
        Args:
            slider_value: Slider value (0-100)
            
        Returns:
            Zoom percentage
        """
        # Use logarithmic scale to give more precision to smaller zoom levels
        min_log = math.log(ZoomLevels.MIN_ZOOM)
        max_log = math.log(ZoomLevels.MAX_ZOOM)
        
        # Convert from 0-100 range to logarithmic scale
        zoom_log = min_log + (slider_value / 100) * (max_log - min_log)
        
        # Convert to percentage
        zoom_percent = math.exp(zoom_log)
        return zoom_percent


class MinimapNavigator(QWidget):
    """A minimap widget for navigating large images."""
    
    viewport_changed = pyqtSignal(QRectF)  # Emitted when viewport changes
    
    def __init__(self, parent=None):
        """Initialize the minimap navigator.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set fixed size
        self.setMinimumSize(150, 150)
        self.setMaximumSize(200, 200)
        
        # Minimap properties
        self._image = None  # The thumbnail image
        self._image_rect = QRectF(0, 0, 1, 1)  # Image rect in normalized coordinates
        self._viewport_rect = QRectF(0, 0, 1, 1)  # Viewport rect in normalized coordinates
        
        # Interaction state
        self._dragging = False
        self._drag_start = QPoint()
        self._viewport_start = QRectF()
        
        # Set up rubber band for selection visualization
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_band.hide()
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(240, 240, 240))
        self.setPalette(palette)
    
    def set_image(self, pixmap: QPixmap) -> None:
        """Set the image to display in the minimap.
        
        Args:
            pixmap: The image to display
        """
        if pixmap:
            # Create scaled thumbnail
            self._image = pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Calculate image rect in widget coordinates
            img_width = self._image.width()
            img_height = self._image.height()
            
            x = (self.width() - img_width) / 2
            y = (self.height() - img_height) / 2
            
            self._image_rect = QRectF(x, y, img_width, img_height)
        else:
            self._image = None
            self._image_rect = QRectF(0, 0, 1, 1)
            
        self.update()
    
    def set_viewport(self, viewport_rect: QRectF) -> None:
        """Set the current viewport rectangle in normalized coordinates.
        
        Args:
            viewport_rect: Viewport rect with values in range 0-1
        """
        # Ensure rect is in normalized coordinates (0-1)
        self._viewport_rect = QRectF(
            max(0, min(1, viewport_rect.x())),
            max(0, min(1, viewport_rect.y())),
            max(0, min(1, viewport_rect.width())),
            max(0, min(1, viewport_rect.height()))
        )
        
        self._update_rubber_band()
        self.update()
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the minimap."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw border
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        if self._image:
            # Draw the image
            painter.drawPixmap(self._image_rect.toRect(), self._image)
            
            # Draw viewport indicator (handled by rubber band)
            
        else:
            # Draw placeholder
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.setFont(self.font())
            painter.drawText(
                self.rect(), 
                Qt.AlignmentFlag.AlignCenter, 
                "No Image"
            )
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._image and self._image_rect.contains(event.position().toPoint()):
                self._dragging = True
                self._drag_start = event.position().toPoint()
                self._viewport_start = QRectF(self._viewport_rect)
                
                # Update viewport center to clicked point
                self._update_viewport_from_point(event.position().toPoint())
                
                # Update rubber band
                self._update_rubber_band()
                event.accept()
                return
                
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        if self._dragging:
            self._update_viewport_from_point(event.position().toPoint())
            self._update_rubber_band()
            event.accept()
            return
            
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
            
        super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        
        # Rescale image if exists
        if self._image:
            self.set_image(self._image)
            
        # Update rubber band
        self._update_rubber_band()
    
    def _update_viewport_from_point(self, point: QPoint) -> None:
        """Update viewport based on clicked or dragged point.
        
        Args:
            point: The point in widget coordinates
        """
        if not self._image or not self._image_rect.isValid():
            return
            
        # Convert point to normalized coordinates within the image
        norm_x = (point.x() - self._image_rect.x()) / self._image_rect.width()
        norm_y = (point.y() - self._image_rect.y()) / self._image_rect.height()
        
        # Clamp to valid range
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Calculate new viewport center
        new_x = norm_x - self._viewport_rect.width() / 2
        new_y = norm_y - self._viewport_rect.height() / 2
        
        # Ensure viewport stays within image
        new_x = max(0, min(1 - self._viewport_rect.width(), new_x))
        new_y = max(0, min(1 - self._viewport_rect.height(), new_y))
        
        # Update viewport
        self._viewport_rect.moveLeft(new_x)
        self._viewport_rect.moveTop(new_y)
        
        # Emit signal
        self.viewport_changed.emit(self._viewport_rect)
        
        # Update display
        self.update()
    
    def _update_rubber_band(self) -> None:
        """Update the rubber band to show the current viewport."""
        if not self._image or not self._image_rect.isValid():
            self._rubber_band.hide()
            return
            
        # Convert normalized viewport to widget coordinates
        x = self._image_rect.x() + self._viewport_rect.x() * self._image_rect.width()
        y = self._image_rect.y() + self._viewport_rect.y() * self._image_rect.height()
        w = self._viewport_rect.width() * self._image_rect.width()
        h = self._viewport_rect.height() * self._image_rect.height()
        
        # Set rubber band geometry
        self._rubber_band.setGeometry(int(x), int(y), int(w), int(h))
        self._rubber_band.show()


class ZoomPanel(QWidget):
    """A panel with zoom controls and minimap for navigating images."""
    
    zoom_changed = pyqtSignal(float)  # Emitted when zoom percentage changes
    viewport_changed = pyqtSignal(QRectF)  # Emitted when viewport changes
    
    def __init__(self, parent=None):
        """Initialize the zoom panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Header label
        header = QLabel("Navigation")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Zoom slider
        self.zoom_slider = ZoomSlider(self)
        self.zoom_slider.zoom_changed.connect(self.zoom_changed)
        layout.addWidget(self.zoom_slider)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Minimap
        self.minimap = MinimapNavigator(self)
        self.minimap.viewport_changed.connect(self.viewport_changed)
        layout.addWidget(self.minimap, 1)
        
        # Additional navigation controls
        button_layout = QHBoxLayout()
        
        # Center view button
        self.center_button = QPushButton("Center")
        self.center_button.clicked.connect(self._center_view)
        button_layout.addWidget(self.center_button)
        
        # Full view button
        self.full_button = QPushButton("Full")
        self.full_button.clicked.connect(self._full_view)
        button_layout.addWidget(self.full_button)
        
        layout.addLayout(button_layout)
    
    def set_image(self, pixmap: QPixmap) -> None:
        """Set the image for the minimap.
        
        Args:
            pixmap: The image to display
        """
        self.minimap.set_image(pixmap)
    
    def set_zoom(self, percent: float) -> None:
        """Set the zoom level.
        
        Args:
            percent: Zoom percentage
        """
        self.zoom_slider.set_zoom(percent)
    
    def get_zoom(self) -> float:
        """Get the current zoom percentage."""
        return self.zoom_slider.get_zoom()
    
    def set_viewport(self, viewport_rect: QRectF) -> None:
        """Set the current viewport rectangle.
        
        Args:
            viewport_rect: Viewport rect with values in range 0-1
        """
        self.minimap.set_viewport(viewport_rect)
    
    def _center_view(self) -> None:
        """Center the view on the image."""
        # Create a viewport rect centered in the image
        viewport = QRectF(
            self.minimap._viewport_rect.x(),
            self.minimap._viewport_rect.y(),
            self.minimap._viewport_rect.width(),
            self.minimap._viewport_rect.height()
        )
        
        # Center horizontally
        viewport.moveLeft(0.5 - viewport.width() / 2)
        
        # Center vertically
        viewport.moveTop(0.5 - viewport.height() / 2)
        
        # Ensure viewport stays within image
        viewport.moveLeft(max(0, min(1 - viewport.width(), viewport.x())))
        viewport.moveTop(max(0, min(1 - viewport.height(), viewport.y())))
        
        # Update viewport
        self.minimap.set_viewport(viewport)
        
        # Emit signal
        self.viewport_changed.emit(viewport)
    
    def _full_view(self) -> None:
        """Show the full image view."""
        # Create a viewport that shows the entire image
        viewport = QRectF(0, 0, 1, 1)
        
        # Update viewport
        self.minimap.set_viewport(viewport)
        
        # Emit signal
        self.viewport_changed.emit(viewport) 