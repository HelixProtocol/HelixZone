# HelixZone Format Support

This document outlines the image formats supported by HelixZone and their capabilities.

## Supported Formats

HelixZone supports a wide range of image formats for both input and output:

### Basic Formats (Always Available)

| Format | Extension | Read | Write | Features |
|--------|-----------|------|-------|----------|
| PNG | .png | ✓ | ✓ | Lossless compression, transparency, metadata |
| JPEG | .jpg, .jpeg | ✓ | ✓ | Lossy compression, quality controls |
| TIFF | .tif, .tiff | ✓ | ✓ | Multiple compression options, multiple bit depths |
| BMP | .bmp | ✓ | ✓ | Uncompressed, simple format |
| GIF | .gif | ✓ | ✓ | Animation support (read only), 8-bit indexed color |
| WebP | .webp | ✓ | ✓ | Lossy or lossless, smaller than JPEG at equivalent quality |

### Advanced Formats (Requires Optional Dependencies)

| Format | Extension | Read | Write | Features | Dependencies |
|--------|-----------|------|-------|----------|-------------|
| RAW | .arw, .cr2, .cr3, .dng, .nef, .orf, .pef, .raf, .rw2, .srw, .x3f | ✓ | ✗ | Camera RAW formats, advanced processing | rawpy, exifread |
| HDR | .hdr | ✓ | ✓ | Radiance HDR format, high dynamic range | OpenImageIO or OpenCV+colour |
| OpenEXR | .exr | ✓ | ✓ | Professional HDR format, multiple layers | OpenImageIO |
| HEIF/HEIC | .heif, .heic | ✓ | ✓ | High efficiency format used by Apple | pillow-heif or OpenImageIO |
| AVIF | .avif | ✓ | ✓ | AV1 Image File Format, excellent compression | pillow-avif or OpenImageIO |

## Installing Optional Dependencies

To enable advanced format support, install the optional dependencies:

```bash
# For RAW support
pip install rawpy exifread

# For HDR support (OpenImageIO is preferred but has complex installation)
pip install OpenImageIO

# Alternative for HDR support
pip install opencv-python-headless colour

# For advanced RAW demosaicing
pip install colour-demosaicing

# For HEIF/HEIC support
pip install pillow-heif

# For AVIF support
pip install pillow-avif
```

### Notes on OpenImageIO Installation

OpenImageIO is a comprehensive library for image handling but can be complex to install:

- **Windows:** Best installed via conda: `conda install -c conda-forge openimageio`
- **macOS:** Can be installed via Homebrew: `brew install openimageio`
- **Linux:** Available through package managers (e.g., `apt install libopenimageio-dev`) or conda

## RAW Processing Options

When opening RAW files, HelixZone offers several processing options:

### Processing Modes

- **Standard (sRGB)** - Default mode with tone mapping for viewing on standard displays
- **Linear** - No tone mapping, preserves linear light values for technical work
- **Custom** - Allows customization of all processing parameters

### White Balance Options

- **Auto White Balance** - Automatically adjust white balance based on image content
- **Camera White Balance** - Use white balance settings from the camera
- **Custom** - Manually set white balance adjustments

### Additional Options

- **Brightness** - Adjust the overall brightness of the image
- **Highlight Recovery** - Recover detail in overexposed areas
- **Demosaicing Algorithm** - Change the algorithm used to convert the RAW Bayer pattern to RGB

## HDR Processing Options

When working with HDR images, HelixZone provides tools to adjust their display:

### Tone Mapping

- **Enable/Disable Tone Mapping** - Convert high dynamic range to standard display range
- **Exposure Adjustment** - Fine-tune the exposure level (+/- 3 EV)
- **Dynamic Range Compression** - Adjust how much the dynamic range is compressed

## Metadata Preservation

HelixZone can preserve various types of metadata during editing and export:

- **EXIF** - Camera information, date/time, exposure settings, etc.
- **XMP** - Adobe's Extensible Metadata Platform for editing history
- **IPTC** - Copyright, authorship, and description information
- **ICC Profiles** - Color space information

### Metadata Compatibility

Not all formats support all types of metadata:

- **Full Support:** PNG, TIFF
- **Partial Support:** JPEG (EXIF, XMP, IPTC), WebP (EXIF, XMP)
- **Limited Support:** GIF, BMP (minimal metadata)
- **Extended Support:** HDR and EXR formats include advanced technical metadata

## Color Management

HelixZone includes comprehensive color management features:

- **ICC Profile Support** - Read and write ICC color profiles
- **Color Space Conversion** - Convert between color spaces (sRGB, Adobe RGB, ProPhoto, etc.)
- **Rendering Intent Options** - Control how colors are mapped between spaces
  - Perceptual - Optimized for natural images
  - Relative Colorimetric - Maintains color accuracy
  - Absolute Colorimetric - Exact color matching
  - Saturation - Optimized for graphics and charts 