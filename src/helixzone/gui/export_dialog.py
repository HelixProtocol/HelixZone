"""
Export dialog with advanced options for image exporting.

This module provides a dialog for configuring export options,
including format selection, quality settings, metadata handling,
and color profile management.
"""

import logging
import os
from typing import Dict, Optional, Tuple, List, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QSpinBox, QCheckBox, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QSlider, QFormLayout,
    QDialogButtonBox, QSizePolicy, QRadioButton, QButtonGroup,
    QScrollArea, QSpacerItem
)
from PyQt6.QtCore import Qt, QSize, QByteArray
from PyQt6.QtGui import QImage, QPixmap

from ..core.file_manager import FileFormat, get_file_manager
from ..core.format_support import get_format_support, ColorProfile

logger = logging.getLogger(__name__)


class ExportOptionsDialog(QDialog):
    """Dialog for configuring advanced export options."""
    
    def __init__(self, parent=None, initial_format=FileFormat.PNG):
        """Initialize the export options dialog.
        
        Args:
            parent: Parent widget
            initial_format: Initial file format
        """
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.resize(600, 500)
        
        # Initialize managers
        self.file_manager = get_file_manager()
        self.format_support = get_format_support()
        
        # Set up layout
        layout = QVBoxLayout(self)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tab pages
        self.create_format_tab()
        self.create_quality_tab()
        self.create_metadata_tab()
        self.create_color_tab()
        
        # Create file name preview
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Output:"))
        self.file_name_preview = QLineEdit()
        self.file_name_preview.setReadOnly(True)
        preview_layout.addWidget(self.file_name_preview)
        layout.addLayout(preview_layout)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Set the initial format
        self.set_format(initial_format)
        
    def create_format_tab(self):
        """Create the format selection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Format selection group
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout(format_group)
        
        # Format combo box
        self.format_combo = QComboBox()
        
        # Add all supported formats
        self.formats = [
            (FileFormat.PNG, "PNG - Portable Network Graphics"),
            (FileFormat.JPEG, "JPEG - Joint Photographic Experts Group"),
            (FileFormat.TIFF, "TIFF - Tagged Image File Format"),
            (FileFormat.WEBP, "WebP - Web Picture Format"),
            (FileFormat.BMP, "BMP - Bitmap Image"),
            (FileFormat.GIF, "GIF - Graphics Interchange Format"),
            (FileFormat.EXR, "EXR - OpenEXR High Dynamic Range"),
            (FileFormat.HDR, "HDR - Radiance HDR"),
            (FileFormat.HEIF, "HEIF - High Efficiency Image Format"),
            (FileFormat.AVIF, "AVIF - AV1 Image File Format")
        ]
        
        for format_enum, format_name in self.formats:
            self.format_combo.addItem(format_name, format_enum)
        
        format_layout.addWidget(self.format_combo)
        
        # Format info label
        self.format_info = QLabel()
        self.format_info.setWordWrap(True)
        self.format_info.setStyleSheet("color: #666;")
        format_layout.addWidget(self.format_info)
        
        layout.addWidget(format_group)
        
        # Size options
        size_group = QGroupBox("Output Size")
        size_layout = QFormLayout(size_group)
        
        size_layout.addRow(QLabel("<b>Size</b>"), QLabel())
        
        # Original size option
        self.original_size_radio = QRadioButton("Original size")
        self.original_size_radio.setChecked(True)
        size_layout.addRow(self.original_size_radio, QLabel())
        
        # Custom size option
        size_row = QHBoxLayout()
        self.custom_size_radio = QRadioButton("Custom size:")
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 16000)
        self.width_spin.setValue(1920)
        self.width_spin.setEnabled(False)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 16000)
        self.height_spin.setValue(1080)
        self.height_spin.setEnabled(False)
        
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_row.addWidget(QLabel("pixels"))
        
        size_layout.addRow(self.custom_size_radio, size_row)
        
        # Connect size radio buttons
        size_group_buttons = QButtonGroup(self)
        size_group_buttons.addButton(self.original_size_radio)
        size_group_buttons.addButton(self.custom_size_radio)
        
        self.original_size_radio.toggled.connect(self._on_size_option_changed)
        self.custom_size_radio.toggled.connect(self._on_size_option_changed)
        
        # Maintain aspect ratio
        self.maintain_aspect_ratio = QCheckBox("Maintain aspect ratio")
        self.maintain_aspect_ratio.setChecked(True)
        self.maintain_aspect_ratio.setEnabled(False)
        size_layout.addRow("", self.maintain_aspect_ratio)
        
        layout.addWidget(size_group)
        
        # Add to tabs
        self.tab_widget.addTab(tab, "Format")
        
        # Connect signals
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        
    def create_quality_tab(self):
        """Create the quality settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Quality settings for file formats
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QVBoxLayout(quality_group)
        
        # JPEG/WebP quality slider
        quality_form = QFormLayout()
        
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(90)
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(10)
        
        self.quality_label = QLabel("90%")
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f"{v}%"))
        
        quality_slider_layout = QHBoxLayout()
        quality_slider_layout.addWidget(self.quality_slider)
        quality_slider_layout.addWidget(self.quality_label)
        
        quality_form.addRow("Compression Quality:", quality_slider_layout)
        
        # Quality info
        quality_info = QLabel("Higher quality results in larger file sizes.")
        quality_info.setWordWrap(True)
        quality_info.setStyleSheet("color: #666;")
        
        quality_layout.addLayout(quality_form)
        quality_layout.addWidget(quality_info)
        
        # Add preset buttons
        presets_layout = QHBoxLayout()
        
        low_button = QPushButton("Low (50%)")
        low_button.clicked.connect(lambda: self.quality_slider.setValue(50))
        
        medium_button = QPushButton("Medium (75%)")
        medium_button.clicked.connect(lambda: self.quality_slider.setValue(75))
        
        high_button = QPushButton("High (90%)")
        high_button.clicked.connect(lambda: self.quality_slider.setValue(90))
        
        maximum_button = QPushButton("Maximum (100%)")
        maximum_button.clicked.connect(lambda: self.quality_slider.setValue(100))
        
        presets_layout.addWidget(low_button)
        presets_layout.addWidget(medium_button)
        presets_layout.addWidget(high_button)
        presets_layout.addWidget(maximum_button)
        
        quality_layout.addLayout(presets_layout)
        layout.addWidget(quality_group)
        
        # Specific format options
        format_options_group = QGroupBox("Format-Specific Options")
        format_options_layout = QVBoxLayout(format_options_group)
        
        # PNG options
        self.png_options_widget = QWidget()
        png_layout = QFormLayout(self.png_options_widget)
        
        self.png_compression_combo = QComboBox()
        self.png_compression_combo.addItems(["Default", "Fast", "Best Compression"])
        png_layout.addRow("Compression:", self.png_compression_combo)
        
        self.png_interlaced = QCheckBox("Interlaced")
        png_layout.addRow("Options:", self.png_interlaced)
        
        # JPEG options
        self.jpeg_options_widget = QWidget()
        jpeg_layout = QFormLayout(self.jpeg_options_widget)
        
        self.jpeg_subsampling_combo = QComboBox()
        self.jpeg_subsampling_combo.addItems(["4:4:4 (Best Quality)", "4:2:2", "4:2:0 (Smallest)"])
        jpeg_layout.addRow("Chroma Subsampling:", self.jpeg_subsampling_combo)
        
        self.jpeg_progressive = QCheckBox("Progressive")
        self.jpeg_optimize = QCheckBox("Optimize")
        self.jpeg_optimize.setChecked(True)
        
        jpeg_options = QHBoxLayout()
        jpeg_options.addWidget(self.jpeg_progressive)
        jpeg_options.addWidget(self.jpeg_optimize)
        jpeg_layout.addRow("Options:", jpeg_options)
        
        # TIFF options
        self.tiff_options_widget = QWidget()
        tiff_layout = QFormLayout(self.tiff_options_widget)
        
        self.tiff_compression_combo = QComboBox()
        self.tiff_compression_combo.addItems(["None", "LZW", "Deflate", "JPEG"])
        tiff_layout.addRow("Compression:", self.tiff_compression_combo)
        
        self.tiff_bit_depth_combo = QComboBox()
        self.tiff_bit_depth_combo.addItems(["8-bit", "16-bit", "32-bit float"])
        tiff_layout.addRow("Bit Depth:", self.tiff_bit_depth_combo)
        
        # WebP options
        self.webp_options_widget = QWidget()
        webp_layout = QFormLayout(self.webp_options_widget)
        
        self.webp_lossless = QCheckBox("Lossless")
        self.webp_lossless.toggled.connect(self._on_webp_lossless_toggled)
        webp_layout.addRow("Mode:", self.webp_lossless)
        
        # HDR options
        self.hdr_options_widget = QWidget()
        hdr_layout = QFormLayout(self.hdr_options_widget)
        
        self.hdr_format_combo = QComboBox()
        self.hdr_format_combo.addItems(["Radiance RGBE", "OpenEXR"])
        hdr_layout.addRow("HDR Format:", self.hdr_format_combo)
        
        # Add all option widgets to the layout (initially hidden)
        format_options_layout.addWidget(self.png_options_widget)
        format_options_layout.addWidget(self.jpeg_options_widget)
        format_options_layout.addWidget(self.tiff_options_widget)
        format_options_layout.addWidget(self.webp_options_widget)
        format_options_layout.addWidget(self.hdr_options_widget)
        
        # Hide all format-specific options initially
        self.png_options_widget.hide()
        self.jpeg_options_widget.hide()
        self.tiff_options_widget.hide()
        self.webp_options_widget.hide()
        self.hdr_options_widget.hide()
        
        layout.addWidget(format_options_group)
        layout.addStretch()
        
        # Add to tabs
        self.tab_widget.addTab(tab, "Quality")
        
    def create_metadata_tab(self):
        """Create the metadata handling tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Metadata group
        metadata_group = QGroupBox("Metadata Options")
        metadata_layout = QVBoxLayout(metadata_group)
        
        # Metadata options
        self.preserve_exif = QCheckBox("Preserve EXIF metadata")
        self.preserve_exif.setChecked(True)
        self.preserve_exif.setToolTip("Preserve camera and photo information")
        
        self.preserve_xmp = QCheckBox("Preserve XMP metadata")
        self.preserve_xmp.setChecked(True)
        self.preserve_xmp.setToolTip("Preserve Adobe XMP data")
        
        self.preserve_iptc = QCheckBox("Preserve IPTC metadata")
        self.preserve_iptc.setChecked(True)
        self.preserve_iptc.setToolTip("Preserve copyright and authorship information")
        
        self.preserve_color_profile = QCheckBox("Preserve ICC color profile")
        self.preserve_color_profile.setChecked(True)
        self.preserve_color_profile.setToolTip("Preserve color management information")
        
        metadata_layout.addWidget(self.preserve_exif)
        metadata_layout.addWidget(self.preserve_xmp)
        metadata_layout.addWidget(self.preserve_iptc)
        metadata_layout.addWidget(self.preserve_color_profile)
        
        # Metadata compatibility note
        compatibility_label = QLabel(
            "Note: Not all formats support all types of metadata. "
            "PNG supports all metadata types. JPEG supports EXIF, XMP, and IPTC. "
            "TIFF supports EXIF and XMP. WebP supports EXIF and XMP."
        )
        compatibility_label.setWordWrap(True)
        compatibility_label.setStyleSheet("color: #666;")
        metadata_layout.addWidget(compatibility_label)
        
        layout.addWidget(metadata_group)
        
        # Copyright group
        copyright_group = QGroupBox("Copyright Information")
        copyright_layout = QFormLayout(copyright_group)
        
        self.copyright_check = QCheckBox("Add copyright information")
        copyright_layout.addRow(self.copyright_check, QLabel())
        
        self.author_edit = QLineEdit()
        self.author_edit.setEnabled(False)
        copyright_layout.addRow("Author:", self.author_edit)
        
        self.copyright_edit = QLineEdit()
        self.copyright_edit.setEnabled(False)
        copyright_layout.addRow("Copyright:", self.copyright_edit)
        
        # Connect checkbox
        self.copyright_check.toggled.connect(self._on_copyright_toggled)
        
        layout.addWidget(copyright_group)
        layout.addStretch()
        
        # Add to tabs
        self.tab_widget.addTab(tab, "Metadata")
        
    def create_color_tab(self):
        """Create the color management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Color profile group
        profile_group = QGroupBox("Color Profile")
        profile_layout = QVBoxLayout(profile_group)
        
        # Color profile options
        self.use_embedded_profile = QRadioButton("Use embedded profile (if available)")
        self.use_embedded_profile.setChecked(True)
        
        self.convert_to_srgb = QRadioButton("Convert to sRGB (standard for web)")
        
        self.custom_profile = QRadioButton("Use custom profile:")
        
        profile_layout.addWidget(self.use_embedded_profile)
        profile_layout.addWidget(self.convert_to_srgb)
        profile_layout.addWidget(self.custom_profile)
        
        # Custom profile selection
        profile_selection = QHBoxLayout()
        
        self.profile_path_edit = QLineEdit()
        self.profile_path_edit.setEnabled(False)
        self.profile_path_edit.setReadOnly(True)
        
        self.browse_profile_button = QPushButton("Browse...")
        self.browse_profile_button.setEnabled(False)
        self.browse_profile_button.clicked.connect(self._browse_profile)
        
        profile_selection.addWidget(self.profile_path_edit)
        profile_selection.addWidget(self.browse_profile_button)
        
        profile_layout.addLayout(profile_selection)
        
        # Connect radio buttons
        profile_group_buttons = QButtonGroup(self)
        profile_group_buttons.addButton(self.use_embedded_profile)
        profile_group_buttons.addButton(self.convert_to_srgb)
        profile_group_buttons.addButton(self.custom_profile)
        
        self.custom_profile.toggled.connect(
            lambda checked: self._enable_custom_profile(checked))
        
        layout.addWidget(profile_group)
        
        # Rendering intent group
        intent_group = QGroupBox("Rendering Intent")
        intent_layout = QVBoxLayout(intent_group)
        
        # Rendering intent options
        self.perceptual_intent = QRadioButton("Perceptual (optimized for photos)")
        self.perceptual_intent.setChecked(True)
        
        self.relative_intent = QRadioButton("Relative Colorimetric (accurate colors)")
        
        self.saturation_intent = QRadioButton("Saturation (optimized for graphics)")
        
        self.absolute_intent = QRadioButton("Absolute Colorimetric (exact match)")
        
        intent_layout.addWidget(self.perceptual_intent)
        intent_layout.addWidget(self.relative_intent)
        intent_layout.addWidget(self.saturation_intent)
        intent_layout.addWidget(self.absolute_intent)
        
        # Description of rendering intent
        intent_description = QLabel(
            "Rendering intent determines how colors are mapped between different color spaces. "
            "Perceptual is recommended for most photo exports."
        )
        intent_description.setWordWrap(True)
        intent_description.setStyleSheet("color: #666;")
        intent_layout.addWidget(intent_description)
        
        # Group the intent radio buttons
        intent_group_buttons = QButtonGroup(self)
        intent_group_buttons.addButton(self.perceptual_intent)
        intent_group_buttons.addButton(self.relative_intent)
        intent_group_buttons.addButton(self.saturation_intent)
        intent_group_buttons.addButton(self.absolute_intent)
        
        layout.addWidget(intent_group)
        layout.addStretch()
        
        # Add to tabs
        self.tab_widget.addTab(tab, "Color")
        
    def _enable_custom_profile(self, enabled):
        """Enable or disable custom profile selection.
        
        Args:
            enabled: Whether to enable custom profile selection
        """
        self.profile_path_edit.setEnabled(enabled)
        self.browse_profile_button.setEnabled(enabled)
    
    def _on_size_option_changed(self):
        """Handle change in size option."""
        use_custom = self.custom_size_radio.isChecked()
        self.width_spin.setEnabled(use_custom)
        self.height_spin.setEnabled(use_custom)
        self.maintain_aspect_ratio.setEnabled(use_custom)
    
    def _on_format_changed(self, index):
        """Handle change in format selection.
        
        Args:
            index: Index of the selected format
        """
        format_enum = self.format_combo.currentData()
        
        # Update format info
        if format_enum == FileFormat.PNG:
            self.format_info.setText(
                "PNG is a lossless format ideal for graphics, illustrations, "
                "and images with transparency. No quality loss, but larger file sizes."
            )
        elif format_enum == FileFormat.JPEG:
            self.format_info.setText(
                "JPEG is a lossy compression format ideal for photographs. "
                "Small file sizes, but quality loss increases with compression."
            )
        elif format_enum == FileFormat.TIFF:
            self.format_info.setText(
                "TIFF is a versatile format supporting various bit depths and compression types. "
                "Often used for print and professional workflows."
            )
        elif format_enum == FileFormat.WEBP:
            self.format_info.setText(
                "WebP is a modern format developed by Google, offering smaller file sizes "
                "than JPEG at equivalent quality. Supports both lossy and lossless compression."
            )
        elif format_enum == FileFormat.BMP:
            self.format_info.setText(
                "BMP is a basic uncompressed format. Simple but creates large files. "
                "Compatible with most software but not recommended for web."
            )
        elif format_enum == FileFormat.GIF:
            self.format_info.setText(
                "GIF supports animations and transparency, but is limited to 256 colors. "
                "Best for simple animations and graphics with solid colors."
            )
        elif format_enum == FileFormat.EXR or format_enum == FileFormat.HDR:
            self.format_info.setText(
                "High Dynamic Range formats that store extended brightness ranges. "
                "Used for professional lighting, VFX, and advanced editing workflows."
            )
        elif format_enum == FileFormat.HEIF or format_enum == FileFormat.AVIF:
            self.format_info.setText(
                "Modern formats offering better compression than JPEG. HEIF is used by Apple devices, "
                "while AVIF is an open format offering excellent quality at small file sizes."
            )
        
        # Show/hide format-specific option widgets
        self.png_options_widget.setVisible(format_enum == FileFormat.PNG)
        self.jpeg_options_widget.setVisible(format_enum == FileFormat.JPEG)
        self.tiff_options_widget.setVisible(format_enum == FileFormat.TIFF)
        self.webp_options_widget.setVisible(format_enum == FileFormat.WEBP)
        self.hdr_options_widget.setVisible(
            format_enum == FileFormat.HDR or format_enum == FileFormat.EXR
        )
        
        # Update quality slider visibility
        use_quality = format_enum in [
            FileFormat.JPEG, FileFormat.WEBP, FileFormat.HEIF, FileFormat.AVIF
        ]
        self.quality_slider.setEnabled(use_quality)
        self.quality_label.setEnabled(use_quality)
        
        # Update file extension in preview
        self._update_file_preview()
    
    def _on_webp_lossless_toggled(self, checked):
        """Handle toggling of WebP lossless option.
        
        Args:
            checked: Whether lossless is checked
        """
        # Disable quality slider for lossless WebP
        self.quality_slider.setEnabled(not checked)
        self.quality_label.setEnabled(not checked)
    
    def _on_copyright_toggled(self, checked):
        """Handle toggling of copyright checkbox.
        
        Args:
            checked: Whether copyright is checked
        """
        self.author_edit.setEnabled(checked)
        self.copyright_edit.setEnabled(checked)
    
    def _browse_profile(self):
        """Open a file dialog to browse for a color profile."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Color Profile", "",
            "Color Profiles (*.icc *.icm);;All Files (*)"
        )
        
        if file_path:
            self.profile_path_edit.setText(file_path)
    
    def _update_file_preview(self):
        """Update the file name preview."""
        format_enum = self.format_combo.currentData()
        extension = FileFormat.get_extension(format_enum)
        
        self.file_name_preview.setText(f"example{extension}")
    
    def set_format(self, format_enum):
        """Set the current format.
        
        Args:
            format_enum: Format to select
        """
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == format_enum:
                self.format_combo.setCurrentIndex(i)
                break
    
    def get_options(self) -> Dict[str, Any]:
        """Get the selected export options.
        
        Returns:
            Dictionary of export options
        """
        options = {}
        
        # Format options
        options['format'] = self.format_combo.currentData()
        
        # Size options
        options['use_original_size'] = self.original_size_radio.isChecked()
        options['width'] = self.width_spin.value()
        options['height'] = self.height_spin.value()
        options['maintain_aspect_ratio'] = self.maintain_aspect_ratio.isChecked()
        
        # Quality options
        options['quality'] = self.quality_slider.value()
        
        # Format-specific options
        format_enum = options['format']
        
        if format_enum == FileFormat.PNG:
            options['png_compression'] = self.png_compression_combo.currentIndex()
            options['png_interlaced'] = self.png_interlaced.isChecked()
        
        elif format_enum == FileFormat.JPEG:
            options['jpeg_subsampling'] = self.jpeg_subsampling_combo.currentIndex()
            options['jpeg_progressive'] = self.jpeg_progressive.isChecked()
            options['jpeg_optimize'] = self.jpeg_optimize.isChecked()
        
        elif format_enum == FileFormat.TIFF:
            options['tiff_compression'] = self.tiff_compression_combo.currentIndex()
            options['tiff_bit_depth'] = self.tiff_bit_depth_combo.currentIndex()
        
        elif format_enum == FileFormat.WEBP:
            options['webp_lossless'] = self.webp_lossless.isChecked()
        
        elif format_enum in [FileFormat.HDR, FileFormat.EXR]:
            options['hdr_format'] = self.hdr_format_combo.currentIndex()
        
        # Metadata options
        options['preserve_exif'] = self.preserve_exif.isChecked()
        options['preserve_xmp'] = self.preserve_xmp.isChecked()
        options['preserve_iptc'] = self.preserve_iptc.isChecked()
        options['preserve_color_profile'] = self.preserve_color_profile.isChecked()
        
        # Copyright options
        options['add_copyright'] = self.copyright_check.isChecked()
        options['author'] = self.author_edit.text() if options['add_copyright'] else ""
        options['copyright'] = self.copyright_edit.text() if options['add_copyright'] else ""
        
        # Color profile options
        if self.use_embedded_profile.isChecked():
            options['color_profile_mode'] = 'embedded'
        elif self.convert_to_srgb.isChecked():
            options['color_profile_mode'] = 'srgb'
        else:
            options['color_profile_mode'] = 'custom'
            options['color_profile_path'] = self.profile_path_edit.text()
        
        # Rendering intent
        if self.perceptual_intent.isChecked():
            options['rendering_intent'] = 'perceptual'
        elif self.relative_intent.isChecked():
            options['rendering_intent'] = 'relative_colorimetric'
        elif self.saturation_intent.isChecked():
            options['rendering_intent'] = 'saturation'
        else:
            options['rendering_intent'] = 'absolute_colorimetric'
        
        return options


def show_export_dialog(parent=None, initial_format=FileFormat.PNG) -> Tuple[bool, Dict[str, Any]]:
    """Show the export options dialog.
    
    Args:
        parent: Parent widget
        initial_format: Initial file format
        
    Returns:
        Tuple of (accepted, options_dict)
    """
    dialog = ExportOptionsDialog(parent, initial_format)
    result = dialog.exec()
    
    if result == QDialog.Accepted:
        return True, dialog.get_options()
    else:
        return False, {} 