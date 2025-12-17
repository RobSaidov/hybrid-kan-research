# -*- coding: utf-8 -*-
"""
HybridKAN Package Setup

Installation:
    pip install -e .
    
For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hybridkan",
    version="1.0.0",
    author="Rob",
    author_email="",
    description="Hybrid Kolmogorov-Arnold Networks with Multi-Basis Activation Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybridkan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
