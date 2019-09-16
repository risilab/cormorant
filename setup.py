#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='cormorant',
    version='0.0.1',
    packages=find_packages(),
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 'scipy', 'torch'
    ],
    license='TBD'
)
