# pip install -e .

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os

setup(
    name='mri-radiomics-toolkit',
    version='0.2.0',
    packages=find_packages(),
    url='',
    license='MIT',
    author='ML, Wong',
    author_email="nil",
    description='',
    install_requires=[r.strip() for r in open('requirements.txt').readlines()],
    # entry_points = {
    #     'console_scripts': [
    #         'mnts-train=mnts.scripts.normalization:run_graph_train',
    #         'mnts-infer=mnts.scripts.normalization:run_graph_inference'
    #     ]
    # }
)