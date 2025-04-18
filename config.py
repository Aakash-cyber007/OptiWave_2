"""
SECURE CONFIGURATION FILE (config.py)
------------------------------------
Public configurations for:
- Block SVD
- SVD-PSO Compression 
- KSVD Learning

Runtime settings are hidden from imports.
"""

import os
from typing import Final

# ======================================================
# 1. PUBLIC PATH CONFIGURATION (visible on import)
# ======================================================
class PATHS:
    """Filesystem paths (auto-created)"""
    BASE_DIR: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT: Final[str] = os.path.join(BASE_DIR, 'data/input')
    OUTPUT: Final[str] = os.path.join(BASE_DIR, 'data/output')
    DICTS: Final[str] = os.path.join(BASE_DIR, 'data/dictionaries')
    
    # Auto-create directories
    os.makedirs(INPUT, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(DICTS, exist_ok=True)

# ======================================================
# 2. PUBLIC ALGORITHM CONFIGS (visible on import)
# ======================================================
SVD_CONFIG = {
    'tolerance': 1e-6,
    'max_iter': 50,
    'power_iter': 1,
    'default_rank': None
}

PSO_CONFIG = {
    'particles': 30,
    'iterations': 100,
    'c1': 2.0,
    'c2': 2.0
}

KSVD_CONFIG = {
    'patch_size': 8,
    'stride': 4,
    'sparsity': 5,
    'max_iter': 20
}

# ======================================================
# 3. HIDDEN RUNTIME CONFIG (not exposed)
# ======================================================
_RUNTIME = {
    'verbose': True,         # Hidden debug flag
    'precision': 'float32',  # Hidden computation dtype  
    'secret_key': "x5F#9@2q", # Completely hidden
    'threads': 1             # Hidden parallel setting
}

# Only expose public configurations
__all__ = ['PATHS', 'SVD_CONFIG', 'PSO_CONFIG', 'KSVD_CONFIG']

# ======================================================
# SECURE VALIDATION (hidden from imports)
# ======================================================
def _validate_secrets():
    """Validates hidden configurations"""
    assert isinstance(_RUNTIME['secret_key'], str), "Invalid secret key"
    
if __name__ == "__main__":
    # Only runs validation when file is executed directly
    _validate_secrets()
