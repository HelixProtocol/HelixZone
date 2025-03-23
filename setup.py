#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

setup(
    name="helixzone",
    version="0.1.0",
    description="Advanced Image Processing Application",
    author="HelixZone Team",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "opencv-contrib-python>=4.5.0",
        "PyQt6>=6.2.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "gpu": [
            "cupy>=10.0.0",
            "pyopencl>=2022.0.0",
        ],
    },
    package_data={
        "helixzone": ["py.typed"],
    },
) 