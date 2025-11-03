"""
NexGML setup configuration.
"""
import os
from setuptools import setup, find_packages

# Read version from nexgml/__init__.py
def get_version():
    with open("__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Version not found")

# Read long description from README
def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="nexgml",
    version=get_version(),
    description="NexGML — Next Generation Machine Learning (educational ML utilities)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Nexo-kun",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "numba>=0.56",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/HioDza/HioDzaPlace",
    },
)