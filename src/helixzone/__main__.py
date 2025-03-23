import sys
import os
import logging
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication

from .gui.main_window import MainWindow
from .core.logging_manager import setup_logging
from .core.config import load_config

# Enable high-DPI support - Qt6 handles this automatically
# No need to set any attributes

def check_dependencies():
    """Check for optional dependencies and display information about them."""
    logger = logging.getLogger("helixzone")
    
    # List of optional dependencies and their purpose
    dependencies = [
        ("rawpy", "Camera RAW format support (CR2, NEF, ARW, etc.)"),
        ("exifread", "Advanced EXIF metadata handling"),
        ("OpenImageIO", "HDR image support (EXR, HDR)"),
        ("colour", "Advanced color management and transformations"),
        ("colour_demosaicing", "Advanced RAW demosaicing algorithms")
    ]
    
    available_deps = []
    missing_deps = []
    
    for module_name, purpose in dependencies:
        try:
            __import__(module_name)
            available_deps.append((module_name, purpose))
        except ImportError:
            missing_deps.append((module_name, purpose))
    
    if available_deps:
        logger.info("Optional dependencies available:")
        for name, purpose in available_deps:
            logger.info(f"  - {name}: {purpose}")
    
    if missing_deps:
        logger.info("Optional dependencies not found:")
        for name, purpose in missing_deps:
            logger.info(f"  - {name}: {purpose}")
    
    # Return the results for potential use elsewhere
    return {
        "available": available_deps,
        "missing": missing_deps
    }

def install_missing_dependencies(interactive=True):
    """Attempt to install missing optional dependencies.
    
    Args:
        interactive: Whether to prompt for confirmation
        
    Returns:
        True if dependencies were installed, False otherwise
    """
    logger = logging.getLogger("helixzone")
    
    # Check for missing dependencies
    deps = check_dependencies()
    missing_deps = deps["missing"]
    
    if not missing_deps:
        logger.info("No missing dependencies to install.")
        return True
    
    # Create a list of package names to install
    packages = [name for name, _ in missing_deps]
    install_cmd = f"pip install {' '.join(packages)}"
    
    # If interactive, prompt for confirmation
    if interactive:
        import sys
        logger.info("The following optional dependencies are missing:")
        for name, purpose in missing_deps:
            logger.info(f"  - {name}: {purpose}")
        
        logger.info(f"\nTo install, the following command will be run:")
        logger.info(f"  {install_cmd}")
        
        response = input("\nDo you want to install these dependencies? (y/n): ")
        if response.lower() not in ('y', 'yes'):
            logger.info("Installation cancelled.")
            return False
    
    # Attempt to install dependencies
    logger.info("Installing missing dependencies...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        logger.info("Installation completed successfully.")
        
        # Verify installation
        for name, _ in missing_deps:
            try:
                __import__(name)
                logger.info(f"  - {name}: Successfully installed")
            except ImportError:
                logger.warning(f"  - {name}: Installation may have failed")
                
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HelixZone Image Editor")
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing optional dependencies"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to open"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("helixzone")
    
    logger.info(f"Starting HelixZone")
    
    # Install dependencies if requested
    if args.install_deps:
        if install_missing_dependencies(interactive=True):
            logger.info("Optional dependencies installation completed.")
        else:
            logger.warning("Optional dependencies installation was incomplete.")
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Check dependencies
    check_dependencies()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("HelixZone")
    app.setApplicationVersion("0.1.0")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Open files from command line
    for file_path in args.files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            window.open_document(str(path))
        else:
            logger.warning(f"Could not open file: {file_path}")
    
    # Start event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 