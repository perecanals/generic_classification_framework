#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="generic_classification_framework",
      version="1.0",
      description="Generic classification framework",
      author="Pere Canals",
      author_email="perecanalscanals@gmail.com",
      packages=find_packages(),
      package_data={},
      install_requires=[
        "imbalanced-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
        "setuptools",
        "statsmodels",
        "xgboost"
        ]
)
