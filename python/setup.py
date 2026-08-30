from setuptools import setup, Extension
import os

# Keeps setup.py minimal by letting CMake handle the heavy compilation lifting
setup(
    name="llung_rt",
    version="0.1",
    author="Leigh Gable",
    description="Python bindings for ONNX runtime",
    packages=["llung_rt"],
    zip_safe=False,
)

