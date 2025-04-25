from setuptools import setup, find_packages

setup(
    name='image_compression_project',
    version='0.1.0',
    description='A Python project implementing image compression using SVD, BPSO, DCT, and K-SVD',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/image_compression_project',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'matplotlib>=3.3.0',
        'scipy>=1.7.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)
