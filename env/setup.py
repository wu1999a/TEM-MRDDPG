from setuptools import setup
import sys

setup(
    name='mapf',
    py_modules=['mapf'],
    version= '2.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pyyaml',
        'pynput',
        'imageio',
        'pathlib'
    ],
)