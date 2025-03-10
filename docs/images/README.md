# HelixZone Documentation Images

This directory contains all images used in the HelixZone documentation.

## Directory Structure

```
images/
├── interface/          # UI components and layout
├── tools/             # Tool icons and usage examples
├── tutorials/         # Step-by-step tutorial images
├── workflows/         # Workflow diagrams
└── troubleshooting/   # Error messages and solutions
```

## Image Guidelines

### Naming Convention
- Use lowercase with hyphens
- Include category prefix
- Add number for sequences
- Example: `interface-main-window-01.png`

### Image Specifications
- Format: PNG for screenshots, SVG for diagrams
- Resolution: 1920x1080 or higher
- Compression: Optimized for web
- Max file size: 2MB

### Annotation Standards
- Red: Highlights and important areas
- Blue: Navigation elements
- Green: Success states
- Yellow: Warnings
- White: Text annotations
- Font: Arial, 14pt minimum

## Categories

### Interface
- Main window layout
- Panel locations
- Menu structure
- Toolbar components
- Status bar elements

### Tools
- Tool icons
- Tool options
- Usage examples
- Before/after results
- Common settings

### Tutorials
- Step-by-step screenshots
- Expected results
- Common mistakes
- Success indicators
- Progress markers

### Workflows
- Process diagrams
- Decision trees
- Data flow
- User interactions
- System responses

### Troubleshooting
- Error messages
- Solution steps
- Validation screens
- Success indicators
- Common issues

## Usage Instructions

1. **Adding New Images**
   - Place in appropriate subdirectory
   - Follow naming convention
   - Update image registry
   - Optimize before commit

2. **Updating Images**
   - Maintain same filename
   - Update all related docs
   - Archive old versions
   - Update timestamps

3. **Removing Images**
   - Check all references
   - Update documentation
   - Remove from registry
   - Archive if needed

## Image Registry
Maintained in `image-registry.json` with:
- Filename
- Description
- Usage locations
- Last updated
- Version compatibility 