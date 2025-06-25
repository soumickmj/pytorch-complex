#!/usr/bin/env python

"""The setup script."""

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    readme = fh.read()

try:
    with open("HISTORY.rst", "r", encoding="utf8") as history_file:
        history = history_file.read()
except FileNotFoundError:
    history = ""

setuptools.setup(
    name="pytorch-complex",
    version="0.1.2",
    author="Soumick Chatterjee",
    author_email="soumick.chatterjee@ovgu.de",
    description="Complex Modules for PyTorch",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    url="https://github.com/soumickmj/pytorch-complex",
    packages=setuptools.find_packages(include=["torchcomplex", "torchcomplex.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
    license="MIT",
    include_package_data=True,
)
