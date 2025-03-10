# HelixZone Installation Guide

This guide provides detailed instructions for installing HelixZone on different operating systems and setting up the required dependencies.

## Table of Contents
- [System Requirements](#system-requirements)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [GPU Setup](#gpu-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- 2GB free disk space
- OpenGL 3.3+ compatible graphics card

### Recommended Requirements
- Python 3.10 or higher
- 16GB RAM
- 4GB free disk space
- NVIDIA GPU with CUDA support (GTX 1060 or better)
- AMD GPU with OpenCL support (RX 580 or better)

## Windows Installation

1. **Install Python**
   - Download Python 3.10+ from [python.org](https://python.org)
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```bash
     python --version
     pip --version
     ```

2. **Install Required Build Tools**
   ```bash
   # Open PowerShell as Administrator
   Set-ExecutionPolicy RemoteSigned
   pip install wheel setuptools
   ```

3. **Install HelixZone**
   ```bash
   # Create and activate virtual environment
   python -m venv helixzone-env
   .\helixzone-env\Scripts\activate

   # Install HelixZone with dependencies
   pip install -r requirements.txt
   ```

## macOS Installation

1. **Install Homebrew and Python**
   ```bash
   # Install Homebrew
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python@3.10
   ```

2. **Install Required Dependencies**
   ```bash
   brew install qt@6
   brew install opencv
   ```

3. **Install HelixZone**
   ```bash
   # Create and activate virtual environment
   python3 -m venv helixzone-env
   source helixzone-env/bin/activate

   # Install HelixZone with dependencies
   pip install -r requirements.txt
   ```

## Linux Installation

1. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y python3-pip python3-venv
   sudo apt install -y qt6-base-dev
   sudo apt install -y python3-opencv
   sudo apt install -y libgl1-mesa-glx

   # Fedora
   sudo dnf install python3-pip python3-virtualenv
   sudo dnf install qt6-qtbase-devel
   sudo dnf install python3-opencv
   sudo dnf install mesa-libGL
   ```

2. **Install HelixZone**
   ```bash
   # Create and activate virtual environment
   python3 -m venv helixzone-env
   source helixzone-env/bin/activate

   # Install HelixZone with dependencies
   pip install -r requirements.txt
   ```

## GPU Setup

### NVIDIA GPU Setup
1. **Install NVIDIA Drivers**
   - Download and install the latest drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
   - Verify installation:
     ```bash
     nvidia-smi
     ```

2. **Install CUDA Toolkit**
   - Download CUDA Toolkit 11.8+ from [NVIDIA's CUDA website](https://developer.nvidia.com/cuda-downloads)
   - Add CUDA to system PATH
   - Verify installation:
     ```bash
     nvcc --version
     ```

3. **Install CuPy**
   ```bash
   pip install cupy-cuda11x  # Replace x with your CUDA version
   ```

### AMD GPU Setup
1. **Install AMD Drivers**
   - Download and install the latest drivers from [AMD's website](https://www.amd.com/en/support)

2. **Install OpenCL**
   - Windows: Included with AMD drivers
   - Linux:
     ```bash
     sudo apt install opencl-headers ocl-icd-opencl-dev  # Ubuntu/Debian
     sudo dnf install opencl-headers ocl-icd-devel       # Fedora
     ```

3. **Install PyOpenCL**
   ```bash
   pip install pyopencl
   ```

## Troubleshooting

### Common Issues

1. **ImportError: DLL load failed while importing cv2**
   ![OpenCV Import Error](images/troubleshooting/installation/troubleshoot-opencv-error-01.png)
   - Solution: Reinstall OpenCV
     ```bash
     pip uninstall opencv-python
     pip install opencv-python-headless
     ```

2. **Qt libraries not found**
   ![Qt Libraries Error](images/troubleshooting/installation/troubleshoot-qt-error-01.png)
   - Solution: Install Qt dependencies
     ```bash
     # Windows
     pip install pyqt6

     # Linux
     sudo apt install python3-pyqt6  # Ubuntu/Debian
     sudo dnf install python3-qt6    # Fedora
     ```

3. **GPU not detected**
   ![GPU Detection Error](images/troubleshooting/installation/troubleshoot-gpu-error-01.png)
   - Verify driver installation
   - Check GPU compatibility
   - Update to latest drivers
   - Ensure correct CUDA/OpenCL version

4. **Memory allocation errors**
   ![Memory Error](images/troubleshooting/installation/troubleshoot-memory-error-01.png)
   - Reduce image size
   - Close other applications
   - Check available system memory
   - Update GPU drivers

### Getting Help
- Check our [FAQ section](FAQ.md)
- Visit our [GitHub Issues](https://github.com/helixzone/issues)
- Join our [Discord community](https://discord.gg/helixzone)

## Verifying Installation

Test your installation by running:
```bash
python -c "import helixzone; print(helixzone.__version__)"
```

You should see the version number without any errors. 

# Create processor with benchmarking
feathering = EnhancedLassoFeathering(
    use_gpu=True,
    use_neural_features=True,
    use_mixed_precision=True,
    distributed=True,
    benchmark=True  # Enable benchmarking
)

# Process image
result = feathering.apply_lasso_feathering(
    image, 
    mask, 
    content_aware=True
)

# Save benchmark results and visualizations
feathering.save_benchmark_results() 