# HelixZone Image Editor

A professional-grade image editing application built with PyQt6 and modern Python technologies.

## Features

- Modern, intuitive user interface with dockable panels
- Layer-based non-destructive editing
- Advanced drawing tools with pressure sensitivity support
- Undo/Redo system for all operations
- Support for various image formats (PNG, JPEG, BMP, GIF)
- GPU acceleration support (optional)

## Requirements

- Python 3.8 or higher
- PyQt6
- OpenCV (for image processing)
- NumPy (for efficient matrix operations)
- GPU with CUDA support (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/helixzone.git
cd helixzone
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

### Basic Operations

- **File Operations**
  - New File: Ctrl+N
  - Open: Ctrl+O
  - Save: Ctrl+S
  - Save As: Ctrl+Shift+S

- **Layer Operations**
  - New Layer: Ctrl+Shift+N
  - Toggle Layer Visibility: Click checkbox
  - Adjust Layer Opacity: Use slider

- **Tools**
  - Brush: Paint on the current layer
  - Eraser: Clear pixels on the current layer
  - Pan: Middle mouse button
  - Zoom: Mouse wheel

- **History**
  - Undo: Ctrl+Z
  - Redo: Ctrl+Y

## Development

- Follow PEP 8 and PEP 257 coding standards
- Run tests: `pytest`
- Format code: `black .`
- Check linting: `flake8`

## License

MIT License - See LICENSE file for details 