#!/usr/bin/env python3

"""
Setup script for E.D.D.A.I. framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version from package
def get_version():
    """Get version from __init__.py"""
    init_file = this_directory / "eddai" / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"

setup(
    name="eddai-framework",
    version=get_version(),
    description="Environmental Data-Driven Adaptive Intelligence Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="E.D.D.A.I. Contributors",
    author_email="eddai@openecology.ai",
    url="https://github.com/yourusername/eddai-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Ecology",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "analysis": [
            "scipy>=1.7.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eddai-demo=examples.basic_demo:main",
            "eddai-disturbance=examples.disturbance_response_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
