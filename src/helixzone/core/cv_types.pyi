"""Type stubs for OpenCV types."""
from typing import Any, Dict, List, Optional, Tuple, Union, overload
import numpy as np
from numpy.typing import NDArray

# OpenCV constants
CV_8U: int
CV_8S: int
CV_16U: int
CV_16S: int
CV_32S: int
CV_32F: int
CV_64F: int

# OpenCV flags
IMREAD_UNCHANGED: int
IMREAD_GRAYSCALE: int
IMREAD_COLOR: int
IMREAD_ANYDEPTH: int
IMREAD_ANYCOLOR: int

# OpenCV border types
BORDER_CONSTANT: int
BORDER_REPLICATE: int
BORDER_REFLECT: int
BORDER_WRAP: int
BORDER_REFLECT_101: int
BORDER_TRANSPARENT: int
BORDER_REFLECT101: int
BORDER_DEFAULT: int
BORDER_ISOLATED: int

# OpenCV interpolation flags
INTER_NEAREST: int
INTER_LINEAR: int
INTER_CUBIC: int
INTER_AREA: int
INTER_LANCZOS4: int

# OpenCV Mat class
class Mat:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, size: Tuple[int, int], type: int) -> None: ...
    @overload
    def __init__(self, size: Tuple[int, int], type: int, scalar: Union[float, Tuple[float, ...]]) -> None: ...
    
    def empty(self) -> bool: ...
    def size(self) -> Tuple[int, int]: ...
    def type(self) -> int: ...
    def channels(self) -> int: ...
    def depth(self) -> int: ...
    def total(self) -> int: ...
    def isContinuous(self) -> bool: ...
    def clone(self) -> 'Mat': ...
    def convertTo(self, type: int, alpha: float = 1.0, beta: float = 0.0) -> 'Mat': ...
    def copyTo(self, dst: 'Mat') -> None: ...
    def setTo(self, value: Union[float, Tuple[float, ...]], mask: Optional['Mat'] = None) -> None: ...

# OpenCV functions
@overload
def bilateralFilter(src: NDArray[np.uint8], d: int, sigmaColor: float, sigmaSpace: float, borderType: int = BORDER_DEFAULT) -> NDArray[np.uint8]: ...
@overload
def bilateralFilter(src: NDArray[np.float32], d: int, sigmaColor: float, sigmaSpace: float, borderType: int = BORDER_DEFAULT) -> NDArray[np.float32]: ...

@overload
def Canny(image: NDArray[np.uint8], threshold1: float, threshold2: float, apertureSize: int = 3, L2gradient: bool = False) -> NDArray[np.uint8]: ...
@overload
def Canny(image: NDArray[np.float32], threshold1: float, threshold2: float, apertureSize: int = 3, L2gradient: bool = False) -> NDArray[np.uint8]: ...

@overload
def GaussianBlur(src: NDArray[np.uint8], ksize: Tuple[int, int], sigmaX: float, sigmaY: float = 0, borderType: int = BORDER_DEFAULT) -> NDArray[np.uint8]: ...
@overload
def GaussianBlur(src: NDArray[np.float32], ksize: Tuple[int, int], sigmaX: float, sigmaY: float = 0, borderType: int = BORDER_DEFAULT) -> NDArray[np.float32]: ...

@overload
def Sobel(src: NDArray[np.uint8], ddepth: int, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0, borderType: int = BORDER_DEFAULT) -> NDArray[np.float32]: ...
@overload
def Sobel(src: NDArray[np.float32], ddepth: int, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0, borderType: int = BORDER_DEFAULT) -> NDArray[np.float32]: ...

@overload
def addWeighted(src1: NDArray[np.uint8], alpha: float, src2: NDArray[np.uint8], beta: float, gamma: float, dtype: Optional[int] = None) -> NDArray[np.uint8]: ...
@overload
def addWeighted(src1: NDArray[np.float32], alpha: float, src2: NDArray[np.float32], beta: float, gamma: float, dtype: Optional[int] = None) -> NDArray[np.float32]: ...

@overload
def cvtColor(src: NDArray[np.uint8], code: int, dstCn: Optional[int] = None) -> NDArray[np.uint8]: ...
@overload
def cvtColor(src: NDArray[np.float32], code: int, dstCn: Optional[int] = None) -> NDArray[np.float32]: ...

# OpenCV color conversion codes
COLOR_BGR2GRAY: int
COLOR_RGB2GRAY: int
COLOR_GRAY2BGR: int
COLOR_GRAY2RGB: int
COLOR_BGR2RGB: int
COLOR_RGB2BGR: int
COLOR_BGRA2BGR: int
COLOR_RGBA2RGB: int
COLOR_BGR2BGRA: int
COLOR_RGB2RGBA: int
COLOR_BGRA2GRAY: int
COLOR_RGBA2GRAY: int

# OpenCV error handling
class error(Exception):
    def __init__(self, msg: str) -> None: ...
    def __str__(self) -> str: ...
    def msg(self) -> str: ...
    def code(self) -> int: ...
    def line(self) -> int: ...
    def file(self) -> str: ...
    def func(self) -> str: ... 