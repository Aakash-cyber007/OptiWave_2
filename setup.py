from setuptools import setup, find_packages

setup(
    name="OptiWave",
    version="0.1.0",
    packages=["src"],
    package_dir={"OptiWave": "src"},
    install_requires=[
        'numpy>=1.18.0',  # Example of a dependency
        'scipy>=1.5.0'
    ],
    author="Mathematics Club IITM",
    description="OptiWave - A Python library for image processing around SVD",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Deenabandhan/OptiWave",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

