"""Integration tests for format support functionality."""

import os
import pytest
import numpy as np
from PyQt6.QtGui import QImage

from src.helixzone.core.format_support import (
    get_format_support,
    RawFormatHandler,
    HdrFormatHandler,
    RawProcessingParams,
    RawProcessingMode
)

# Setup test data paths (assuming there's a test_data directory)
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
SAMPLE_RAW_FILE = os.path.join(TEST_DATA_DIR, "sample.cr2")
SAMPLE_HDR_FILE = os.path.join(TEST_DATA_DIR, "sample.hdr")

# Mark these tests as needing test data
requires_test_data = pytest.mark.skipif(
    not os.path.exists(TEST_DATA_DIR),
    reason="Test data directory not found"
)

# Mark tests that require optional dependencies
requires_rawpy = pytest.mark.skipif(
    not RawFormatHandler.is_raw_file(SAMPLE_RAW_FILE) if os.path.exists(SAMPLE_RAW_FILE) else True,
    reason="RAW support (rawpy) not available or test file missing"
)

requires_hdr = pytest.mark.skipif(
    not HdrFormatHandler.is_hdr_file(SAMPLE_HDR_FILE) if os.path.exists(SAMPLE_HDR_FILE) else True,
    reason="HDR support (OpenImageIO) not available or test file missing"
)


@requires_test_data
class TestFormatSupportIntegration:
    """Integration tests for format support."""

    def setup_method(self):
        """Set up the test environment."""
        self.format_support = get_format_support()
        
        # Create directories if they don't exist
        self.output_dir = os.path.join(TEST_DATA_DIR, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    @requires_rawpy
    def test_raw_metadata_extraction(self):
        """Test extraction of metadata from RAW files."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_RAW_FILE):
            pytest.skip(f"Sample RAW file not found: {SAMPLE_RAW_FILE}")
        
        # Extract metadata
        metadata = self.format_support.get_metadata(SAMPLE_RAW_FILE)
        
        # Verify basic metadata is present
        assert isinstance(metadata, dict)
        assert "width" in metadata
        assert "height" in metadata
        assert isinstance(metadata["width"], int)
        assert isinstance(metadata["height"], int)
        assert metadata["width"] > 0
        assert metadata["height"] > 0
        
        # Check for camera information if available
        if "camera_model" in metadata:
            assert isinstance(metadata["camera_model"], str)
        
        # Verify EXIF data if available
        if "exif" in metadata:
            assert isinstance(metadata["exif"], dict)

    @requires_rawpy
    def test_raw_image_loading(self):
        """Test loading of RAW images with different processing parameters."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_RAW_FILE):
            pytest.skip(f"Sample RAW file not found: {SAMPLE_RAW_FILE}")
        
        # Define different processing parameters to test
        processing_modes = [
            RawProcessingParams(mode=RawProcessingMode.LINEAR),
            RawProcessingParams(mode=RawProcessingMode.SRGB, brightness=1.2),
            RawProcessingParams(
                mode=RawProcessingMode.CUSTOM,
                brightness=1.1,
                contrast=1.2,
                saturation=1.3,
                auto_white_balance=True
            )
        ]
        
        for params in processing_modes:
            # Load the image with these parameters
            options = {
                'raw_processing_mode': params.mode,
                'brightness': params.brightness,
                'contrast': params.contrast,
                'saturation': params.saturation,
                'auto_white_balance': params.auto_white_balance
            }
            
            image = self.format_support.load_image(SAMPLE_RAW_FILE, options)
            
            # Verify the image was loaded successfully
            assert image is not None
            assert isinstance(image, QImage)
            assert not image.isNull()
            assert image.width() > 0
            assert image.height() > 0
            
            # Optionally save the image for visual inspection
            output_path = os.path.join(
                self.output_dir, 
                f"raw_output_{params.mode.name.lower()}.jpg"
            )
            image.save(output_path)

    @requires_hdr
    def test_hdr_metadata_extraction(self):
        """Test extraction of metadata from HDR files."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_HDR_FILE):
            pytest.skip(f"Sample HDR file not found: {SAMPLE_HDR_FILE}")
        
        # Extract metadata
        metadata = self.format_support.get_metadata(SAMPLE_HDR_FILE)
        
        # Verify basic metadata is present
        assert isinstance(metadata, dict)
        assert "width" in metadata
        assert "height" in metadata
        assert isinstance(metadata["width"], int)
        assert isinstance(metadata["height"], int)
        assert metadata["width"] > 0
        assert metadata["height"] > 0

    @requires_hdr
    def test_hdr_image_loading(self):
        """Test loading of HDR images with different tone mapping options."""
        # Skip if the test file doesn't exist
        if not os.path.exists(SAMPLE_HDR_FILE):
            pytest.skip(f"Sample HDR file not found: {SAMPLE_HDR_FILE}")
        
        # Test different tone mapping and exposure options
        test_options = [
            {"tone_map": True, "exposure": 0.0},
            {"tone_map": True, "exposure": 1.0},
            {"tone_map": True, "exposure": -1.0},
            {"tone_map": False, "exposure": 0.0}
        ]
        
        for options in test_options:
            # Load the image with these options
            image = self.format_support.load_image(SAMPLE_HDR_FILE, options)
            
            # Verify the image was loaded successfully
            assert image is not None
            assert isinstance(image, QImage)
            assert not image.isNull()
            assert image.width() > 0
            assert image.height() > 0
            
            # Optionally save the image for visual inspection
            output_path = os.path.join(
                self.output_dir, 
                f"hdr_output_tm{int(options['tone_map'])}_exp{options['exposure']}.jpg"
            )
            image.save(output_path)

    def test_integration_between_components(self):
        """Test integration between format support and other components."""
        # This is a placeholder for testing integration with other components
        # such as file_manager, image_view, etc.
        
        # Example: Test that format_support correctly identifies supported formats
        assert hasattr(self.format_support, "is_supported_format")
        
        # Example: Test integration with metadata handling
        if os.path.exists(SAMPLE_RAW_FILE) and RawFormatHandler.is_raw_file(SAMPLE_RAW_FILE):
            metadata = self.format_support.get_metadata(SAMPLE_RAW_FILE)
            # Verify that metadata could be used by other components
            assert isinstance(metadata, dict)
            assert len(metadata) > 0 