"""
SVD and Dictionary Learning Package

This package provides:
1. Block_SVD - Randomized block SVD computation
2. svd_bpso - Image compression using SVD and PSO optimization
3. ksvd - Dictionary learning for image denoising
"""

from .Block_SVD import Block_SVD
from .svd_bpso import svd_bpso
from .ksvd import ksvd

__all__ = ['Block_SVD', 'svd_bpso', 'ksvd']
__version__ = '0.1.0'
