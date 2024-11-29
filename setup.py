# setup.py

from setuptools import setup, find_packages

setup(
    name='vv',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'ipywidgets',
        'nonconformist',
        'omikb',
        'discomat'
    ],
    author="Owain Beynon",
    description="Validation and Verification services for OpenModel",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/H2020-OpenModel/OMI-VV/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

)
