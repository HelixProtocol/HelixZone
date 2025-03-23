"""Shared interfaces for HelixZone components."""

from __future__ import annotations
from typing import Protocol, Optional, Union, List, Dict, Any, runtime_checkable
from typing_extensions import TypeAlias
from dataclasses import dataclass
from abc import ABC, abstractmethod
from PyQt6.QtCore import QPoint, QPointF, Qt, QObject
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QImage
from PyQt6.QtWidgets import QWidget

# Type aliases
QtPoint: TypeAlias = Union[QPoint, QPointF]
QtImage: TypeAlias = QImage

class CanvasInterface(QWidget):
    """Interface for canvas functionality needed by tools."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
    
    @abstractmethod
    def get_transformed_pos(self, pos: QtPoint) -> QPointF: ...
    
    @abstractmethod
    def width(self) -> int: ...
    
    @abstractmethod
    def height(self) -> int: ...
    
    @abstractmethod
    def update(self) -> None: ...
    
    @abstractmethod
    def setCursor(self, cursor: Qt.CursorShape) -> None: ...

@runtime_checkable
class ToolInterface(Protocol):
    """Interface for tool functionality."""
    name: str
    size: int
    opacity: float
    color: QColor
    feather_radius: int
    points: List[QPointF]
    is_active: bool
    
    def mouse_press(self, event: Optional[QMouseEvent]) -> None: ...
    def mouse_move(self, event: Optional[QMouseEvent]) -> None: ...
    def mouse_release(self, event: Optional[QMouseEvent]) -> None: ...
    def get_cursor(self) -> Qt.CursorShape: ...
    def draw_preview(self, painter: QPainter) -> None: ...

@runtime_checkable
class SelectionToolInterface(ToolInterface, Protocol):
    """Interface for selection tool functionality."""
    def cut_selection(self) -> None: ...
    def copy_selection(self) -> None: ...
    def paste_selection(self) -> None: ...
    def start_floating_paste(self, pos: QPointF) -> None: ...
    def recolor_selection(self, color: QColor) -> None: ...

class ToolManagerInterface(QObject):
    """Interface for tool manager functionality."""
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_current_tool(self) -> ToolInterface: ...
    
    @abstractmethod
    def set_tool(self, tool_name: str) -> None: ...

@runtime_checkable
class LayerInterface(Protocol):
    """Interface for layer functionality."""
    def set_image(self, image: QtImage) -> None: ...
    def get_image(self) -> Optional[QtImage]: ...
    def set_visible(self, visible: bool) -> None: ...
    def set_opacity(self, opacity: float) -> None: ...

@runtime_checkable
class LayerStackInterface(Protocol):
    """Interface for layer stack functionality."""
    def get_active_layer(self) -> Optional[LayerInterface]: ...
    def merge_visible(self) -> Optional[QtImage]: ...
    def add_layer(self, name: str) -> Optional[LayerInterface]: ... 