#!/usr/bin/env python3
"""
Setup script for Resonant Learner - Community Edition
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="resonant-learner",
    version="1.0.0",
    author="Damjan Žakelj",
    author_email="zakelj.damjan@gmail.com",
    description="Intelligent early stopping for neural networks using log-periodic resonance analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DamjanZakelj/resonant-learner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "examples": [
            "torchvision>=0.15.0",
            "matplotlib>=3.5.0",
        ],
    },
    keywords="machine-learning deep-learning pytorch early-stopping convergence neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/DamjanZakelj/resonant-learner/issues",
        "Source": "https://github.com/DamjanZakelj/resonant-learner",
        "Documentation": "https://github.com/DamjanZakelj/resonant-learner#readme",
    },
)
