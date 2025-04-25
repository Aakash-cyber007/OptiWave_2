from setuptools import setup, find_packages

setup(
    name="image_processing_project",  # The name of your project
    version="0.1.0",  # Starting version
    description="A collection of image processing techniques including SVD, DCT, and K-SVD.",
    author="Your Name",  # Add your name
    author_email="your_email@example.com",  # Your email
    url="https://github.com/yourusername/image_processing_project",  # Your GitHub repository URL
    packages=find_packages(),  # Automatically find all packages
    install_requires=[  # List of dependencies
        "numpy",  # Required for numerical operations
        "matplotlib",  # Required for image visualization
        "opencv-python",  # Required for OpenCV functions
        "scipy",  # Required for SVD and sparse matrix operations
    ],
    classifiers=[  # Optional, helps categorize your project on PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    entry_points={  # Optional, define command line scripts if necessary
        'console_scripts': [
            'dct-pipeline=image_processing_project.dct_pipeline:main',
            'ksvd=image_processing_project.ksvd:main',
        ],
    },
)
