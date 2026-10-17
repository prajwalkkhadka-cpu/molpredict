from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="molpredict",
    version="1.1.0",
    description="Molecular LogP prediction from SMILES with RDKit features and a PyTorch MLP",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0", "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.3.0",
        "scipy>=1.11.0", "rdkit>=2023.9.1", "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    entry_points={
        "console_scripts": [
            "molpredict-train=molpredict.pipeline:main",
            "molpredict-predict=molpredict.predict_cli:main",
        ]
    },
)
