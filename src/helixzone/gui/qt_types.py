"""Type definitions for Qt-related functionality."""

from typing import (
    TypeVar, Protocol, Any, Callable, overload,
    Union, Optional, Type, cast
)
from typing_extensions import TypeAlias
from PyQt6.QtCore import QObject, QPoint, pyqtSlot as _pyqtSlot
from PyQt6.QtGui import QColor

T = TypeVar('T')

class SlotMethod(Protocol):
    """Protocol for slot methods."""
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class SlotDecorator(Protocol):
    """Protocol for slot decorators."""
    def __call__(self, func: SlotMethod) -> SlotMethod: ...

QObjectT = TypeVar('QObjectT', bound=QObject)
QtColor: TypeAlias = QColor
QtPoint: TypeAlias = QPoint

@overload
def pyqtSlot() -> SlotDecorator:
    """No-argument slot decorator."""
    ...

@overload
def pyqtSlot(type1: Type[Any]) -> SlotDecorator:
    """One-argument slot decorator."""
    ...

@overload
def pyqtSlot(type1: Type[Any], type2: Type[Any]) -> SlotDecorator:
    """Two-argument slot decorator."""
    ...

def pyqtSlot(*types: Type[Any], **kwargs: Any) -> SlotDecorator:
    """Implementation of pyqtSlot that preserves type information."""
    decorator = _pyqtSlot(*types, **kwargs)
    return cast(SlotDecorator, decorator)

__all__ = [
    'pyqtSlot',
    'QtColor',
    'QtPoint',
    'QObjectT',
    'SlotDecorator',
    'SlotMethod'
] 