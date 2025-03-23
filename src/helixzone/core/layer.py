from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from typing import Optional, TYPE_CHECKING, Dict, Any, Callable, List
import numpy as np
import cv2

if TYPE_CHECKING:
    from ..gui.canvas import Canvas

class Layer(QObject):
    """A single layer in the image editor."""
    
    # Signals for layer changes
    changed = pyqtSignal()  # Emitted when layer content changes
    properties_changed = pyqtSignal()  # Emitted when properties (opacity, visibility) change
    
    def __init__(self, name: str = "Layer", size: tuple[int, int] = (800, 600)):
        super().__init__()
        self.name = name
        self.visible = True
        self.opacity = 1.0  # 0.0 to 1.0
        self.blend_mode = "normal"
        
        # Create transparent image for the layer
        self.image = QImage(size[0], size[1], QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.transparent)
    
    def set_image(self, image: QImage | np.ndarray) -> None:
        """Set the layer's image content."""
        if isinstance(image, QImage):
            self.image = image
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            if len(image.shape) == 2:  # Grayscale
                image = np.stack((image,) * 3, axis=-1)
            bytes_per_line = 3 * width
            self.image = QImage(image.data, width, height,
                              bytes_per_line, QImage.Format.Format_RGB888)
        self.changed.emit()
    
    def get_image(self) -> QImage:
        """Get the layer's image content."""
        return self.image
    
    def set_opacity(self, opacity: float) -> None:
        """Set layer opacity (0.0 to 1.0)."""
        self.opacity = max(0.0, min(1.0, opacity))
        self.properties_changed.emit()
    
    def set_visible(self, visible: bool) -> None:
        """Set layer visibility."""
        self.visible = visible
        self.properties_changed.emit()
    
    def set_blend_mode(self, mode: str) -> None:
        """Set layer blend mode."""
        self.blend_mode = mode
        self.properties_changed.emit()
    
    def clear(self) -> None:
        """Clear the layer to transparency."""
        self.image.fill(Qt.GlobalColor.transparent)
        self.changed.emit()
    
    def resize(self, width: int, height: int) -> None:
        """Resize the layer."""
        self.image = self.image.scaled(width, height, 
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.changed.emit()


class AdjustmentLayer(Layer):
    """A layer that applies non-destructive adjustments to layers below it."""
    
    def __init__(self, name: str = "Adjustment", size: tuple[int, int] = (800, 600), adjustment_type: str = "brightness_contrast"):
        super().__init__(name, size)
        self.adjustment_type = adjustment_type
        self.parameters: Dict[str, Any] = {}
        
        # Set default parameters based on adjustment type
        self._init_default_parameters()
    
    def _init_default_parameters(self) -> None:
        """Initialize default parameters based on adjustment type."""
        if self.adjustment_type == "brightness_contrast":
            self.parameters = {
                "brightness": 0,  # -100 to 100
                "contrast": 0     # -100 to 100
            }
        elif self.adjustment_type == "levels":
            self.parameters = {
                "black_point": 0,      # 0 to 255
                "white_point": 255,    # 0 to 255
                "mid_point": 1.0       # 0.1 to 10.0 (gamma)
            }
        elif self.adjustment_type == "curves":
            # Initialize with identity curve (no change)
            self.parameters = {
                "curve_points": [(0, 0), (255, 255)],  # List of (x, y) points defining the curve
                "curve_type": "rgb"  # rgb, r, g, b
            }
        elif self.adjustment_type == "hue_saturation":
            self.parameters = {
                "hue": 0,          # -180 to 180
                "saturation": 0,   # -100 to 100
                "lightness": 0     # -100 to 100
            }
        elif self.adjustment_type == "color_balance":
            self.parameters = {
                "shadows": [0, 0, 0],      # R, G, B values (-100 to 100)
                "midtones": [0, 0, 0],     # R, G, B values (-100 to 100)
                "highlights": [0, 0, 0]    # R, G, B values (-100 to 100)
            }
        elif self.adjustment_type == "threshold":
            self.parameters = {
                "threshold": 127   # 0 to 255
            }
        else:
            # Generic parameters
            self.parameters = {}
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set an adjustment parameter."""
        if key in self.parameters:
            self.parameters[key] = value
            self.changed.emit()
    
    def apply_adjustment(self, image: np.ndarray) -> np.ndarray:
        """Apply the adjustment to an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Adjusted image as numpy array
        """
        # Skip if not visible
        if not self.visible:
            return image
        
        # Apply the appropriate adjustment
        if self.adjustment_type == "brightness_contrast":
            return self._apply_brightness_contrast(image)
        elif self.adjustment_type == "levels":
            return self._apply_levels(image)
        elif self.adjustment_type == "curves":
            return self._apply_curves(image)
        elif self.adjustment_type == "hue_saturation":
            return self._apply_hue_saturation(image)
        elif self.adjustment_type == "color_balance":
            return self._apply_color_balance(image)
        elif self.adjustment_type == "threshold":
            return self._apply_threshold(image)
        else:
            # No adjustment
            return image
    
    def _apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustment.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        brightness = self.parameters.get("brightness", 0)
        contrast = self.parameters.get("contrast", 0)
        
        # Convert contrast to alpha (multiplicative factor)
        alpha = (contrast + 100) / 100.0
        
        # Convert brightness to beta (additive factor, scaled to 0-255)
        beta = brightness * 2.55
        
        # Apply the adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return adjusted
    
    def _apply_levels(self, image: np.ndarray) -> np.ndarray:
        """Apply levels adjustment.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        black_point = self.parameters.get("black_point", 0)
        white_point = self.parameters.get("white_point", 255)
        mid_point = self.parameters.get("mid_point", 1.0)
        
        # Create a lookup table for the adjustment
        lut = np.zeros((1, 256), dtype=np.uint8)
        
        # Calculate the lookup table values
        for i in range(256):
            if i < black_point:
                lut[0, i] = 0
            elif i > white_point:
                lut[0, i] = 255
            else:
                # Apply gamma correction (mid_point)
                normalized = (i - black_point) / (white_point - black_point)
                gamma_corrected = np.power(normalized, 1.0/mid_point)
                lut[0, i] = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
        
        # Apply the lookup table
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            adjusted = cv2.LUT(image, lut)
        else:
            # Grayscale or other format
            adjusted = cv2.LUT(image[:, :, 0], lut)
            if len(image.shape) == 3:
                adjusted = np.stack([adjusted] * image.shape[2], axis=2)
        
        return adjusted
    
    def _apply_curves(self, image: np.ndarray) -> np.ndarray:
        """Apply curves adjustment.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        curve_points = self.parameters.get("curve_points", [(0, 0), (255, 255)])
        curve_type = self.parameters.get("curve_type", "rgb")
        
        # Create a lookup table for the adjustment
        lut = np.zeros((1, 256), dtype=np.uint8)
        
        # Sort points by x value
        curve_points.sort(key=lambda p: p[0])
        
        # Interpolate between curve points
        for i in range(256):
            # Find the surrounding points for interpolation
            lower_point = None
            upper_point = None
            
            for point in curve_points:
                if point[0] <= i:
                    lower_point = point
                if point[0] >= i and upper_point is None:
                    upper_point = point
            
            if lower_point is None:
                lower_point = curve_points[0]
            if upper_point is None:
                upper_point = curve_points[-1]
            
            # Interpolate
            if lower_point[0] == upper_point[0]:
                # Avoid division by zero
                lut[0, i] = min(255, max(0, lower_point[1]))
            else:
                t = (i - lower_point[0]) / (upper_point[0] - lower_point[0])
                val = lower_point[1] + t * (upper_point[1] - lower_point[1])
                lut[0, i] = min(255, max(0, int(val)))
        
        # Apply the lookup table based on curve type
        if curve_type == "rgb" or len(image.shape) < 3:
            # Apply to all channels
            adjusted = cv2.LUT(image, lut)
        else:
            # Apply to specific channel
            adjusted = image.copy()
            channel_idx = {"r": 0, "g": 1, "b": 2}.get(curve_type, 0)
            adjusted[:, :, channel_idx] = cv2.LUT(image[:, :, channel_idx], lut)
        
        return adjusted
    
    def _apply_hue_saturation(self, image: np.ndarray) -> np.ndarray:
        """Apply hue, saturation, and lightness adjustment.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        hue = self.parameters.get("hue", 0)
        saturation = self.parameters.get("saturation", 0)
        lightness = self.parameters.get("lightness", 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust hue (0-179 in OpenCV HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue / 2) % 180  # Divide by 2 because OpenCV hue range is 0-179
        
        # Adjust saturation (0-255 in OpenCV HSV)
        saturation_factor = (saturation + 100) / 100.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        
        # Adjust lightness (value in HSV, 0-255 in OpenCV HSV)
        if lightness > 0:
            # Increase value (limited to 255)
            factor = 1 + lightness / 100.0
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        else:
            # Decrease value (limited to 0)
            factor = (lightness + 100) / 100.0
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return adjusted
    
    def _apply_color_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply color balance adjustment to shadows, midtones, and highlights.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        shadows = self.parameters.get("shadows", [0, 0, 0])
        midtones = self.parameters.get("midtones", [0, 0, 0])
        highlights = self.parameters.get("highlights", [0, 0, 0])
        
        # Convert to float32 for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Create masks for shadows, midtones, and highlights based on luminance
        luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
        shadow_mask = np.clip(1.0 - luminance * 2, 0, 1)
        highlight_mask = np.clip(luminance * 2 - 1.0, 0, 1)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        
        # Apply color adjustments
        for i in range(3):  # Process each color channel
            # Scale adjustments to [-0.25, 0.25] range
            shadow_adj = shadows[i] / 400.0
            midtone_adj = midtones[i] / 400.0
            highlight_adj = highlights[i] / 400.0
            
            # Apply adjustments with masks
            img_float[:, :, i] += (shadow_adj * shadow_mask + 
                                  midtone_adj * midtone_mask + 
                                  highlight_adj * highlight_mask)
        
        # Clip values to [0, 1] range and convert back to uint8
        adjusted = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def _apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply threshold adjustment.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        threshold = self.parameters.get("threshold", 127)
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert back to RGB if input was RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            adjusted = np.stack([binary] * 3, axis=2)
        else:
            adjusted = binary
        
        return adjusted


class LayerStack:
    """Manages a stack of layers."""
    
    def __init__(self):
        self.layers: list[Layer] = []
        self.active_layer_index: int = -1
        self._canvas: Optional['Canvas'] = None
    
    @property
    def canvas(self) -> Optional['Canvas']:
        """Get the associated canvas."""
        return self._canvas
    
    @canvas.setter
    def canvas(self, value: Optional['Canvas']) -> None:
        """Set the associated canvas."""
        self._canvas = value
    
    def add_layer(self, layer: Optional[Layer] = None, name: Optional[str] = None, size: tuple[int, int] = (800, 600)) -> Layer:
        """Add a new layer to the stack."""
        if layer is None:
            layer = Layer(name or f"Layer {len(self.layers) + 1}", size)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        return layer
    
    def add_adjustment_layer(self, adjustment_type: str, name: Optional[str] = None, size: tuple[int, int] = (800, 600)) -> AdjustmentLayer:
        """Add a new adjustment layer to the stack.
        
        Args:
            adjustment_type: Type of adjustment (brightness_contrast, levels, etc.)
            name: Optional layer name
            size: Layer size
            
        Returns:
            The created adjustment layer
        """
        # Create a descriptive name if none provided
        if name is None:
            name_map = {
                "brightness_contrast": "Brightness/Contrast",
                "levels": "Levels",
                "curves": "Curves",
                "hue_saturation": "Hue/Saturation",
                "color_balance": "Color Balance",
                "threshold": "Threshold"
            }
            name = name_map.get(adjustment_type, f"Adjustment {len(self.layers) + 1}")
        
        # Create the adjustment layer
        layer = AdjustmentLayer(name, size, adjustment_type)
        
        # Add to stack
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        
        return layer
    
    def remove_layer(self, index: int) -> None:
        """Remove a layer from the stack."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self.active_layer_index = min(self.active_layer_index,
                                        len(self.layers) - 1)
    
    def move_layer(self, from_index: int, to_index: int) -> None:
        """Move a layer to a new position in the stack."""
        if 0 <= from_index < len(self.layers) and 0 <= to_index < len(self.layers):
            layer = self.layers.pop(from_index)
            self.layers.insert(to_index, layer)
            if self.active_layer_index == from_index:
                self.active_layer_index = to_index
    
    def get_active_layer(self) -> Optional[Layer]:
        """Get the currently active layer."""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
    
    def set_active_layer(self, index: int) -> None:
        """Set the active layer by index."""
        if 0 <= index < len(self.layers):
            self.active_layer_index = index
    
    def merge_visible(self) -> Optional[QImage]:
        """Merge all visible layers into a single image."""
        if not self.layers:
            return None
            
        # Create a new image with the same size as the first layer
        result = QImage(self.layers[0].image.size(), QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Process layers from bottom to top
        accumulated_image = None
        for i, layer in enumerate(self.layers):
            if not layer.visible:
                continue
                
            if isinstance(layer, AdjustmentLayer):
                # Apply adjustment to accumulated image
                if accumulated_image is not None:
                    # Convert QImage to numpy array
                    width = accumulated_image.width()
                    height = accumulated_image.height()
                    ptr = accumulated_image.constBits()
                    ptr.setsize(height * width * 4)
                    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                    
                    # Apply adjustment
                    adjusted = layer.apply_adjustment(arr)
                    
                    # Convert back to QImage
                    if adjusted.shape[2] == 4:
                        img_format = QImage.Format.Format_RGBA8888
                    else:
                        img_format = QImage.Format.Format_RGB888
                        
                    temp_img = QImage(adjusted.data, width, height, 
                                    adjusted.strides[0], img_format)
                    
                    # Update accumulated image
                    accumulated_image = temp_img.copy()
            else:
                # Regular layer - just draw it
                if accumulated_image is None:
                    # This is the first visible layer
                    accumulated_image = layer.image.copy()
                else:
                    # Draw over accumulated image with opacity
                    temp_img = QImage(accumulated_image.size(), QImage.Format.Format_ARGB32)
                    temp_img.fill(Qt.GlobalColor.transparent)
                    
                    temp_painter = QPainter(temp_img)
                    temp_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    temp_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    
                    # Draw accumulated image
                    temp_painter.drawImage(0, 0, accumulated_image)
                    
                    # Draw current layer with opacity
                    temp_painter.setOpacity(layer.opacity)
                    temp_painter.drawImage(0, 0, layer.image)
                    
                    temp_painter.end()
                    
                    # Update accumulated image
                    accumulated_image = temp_img
        
        # Draw final accumulated image
        if accumulated_image:
            painter.drawImage(0, 0, accumulated_image)
        
        painter.end()
        return result 