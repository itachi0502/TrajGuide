#!/usr/bin/env python3
"""
NumPy Compatibility Module for MolPilot
======================================

This module provides compatibility fixes for NumPy 1.20+ where deprecated
aliases like np.int, np.float, etc. have been removed.

Key Features:
- Automatic detection of NumPy version
- Backward-compatible aliases for deprecated types
- Safe imports that work across NumPy versions
- Production-ready compatibility layer

Designed specifically for MolPilot's evaluation pipeline compatibility.
"""

import numpy as np
import warnings
from packaging import version

# Get NumPy version
NUMPY_VERSION = version.parse(np.__version__)
NUMPY_1_20_PLUS = NUMPY_VERSION >= version.parse("1.20.0")

def setup_numpy_compatibility():
    """
    Set up NumPy compatibility for deprecated aliases.
    This function should be called early in the application startup.
    """
    if NUMPY_1_20_PLUS:
        # Restore deprecated aliases for backward compatibility
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'complex'):
            np.complex = complex
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'str'):
            np.str = str
        
        # Specific integer types that might be used
        if not hasattr(np, 'int_'):
            np.int_ = np.int64
        if not hasattr(np, 'float_'):
            np.float_ = np.float64
            
        print("✅ NumPy compatibility layer activated for version", np.__version__)
    else:
        print("✅ NumPy version", np.__version__, "- no compatibility fixes needed")

def safe_numpy_int():
    """
    Get the appropriate integer type for the current NumPy version.
    
    Returns:
        The appropriate integer type (int for NumPy 1.20+, np.int for older versions)
    """
    if NUMPY_1_20_PLUS:
        return int
    else:
        return np.int

def safe_numpy_float():
    """
    Get the appropriate float type for the current NumPy version.
    
    Returns:
        The appropriate float type (float for NumPy 1.20+, np.float for older versions)
    """
    if NUMPY_1_20_PLUS:
        return float
    else:
        return np.float

def safe_numpy_array_int(data, **kwargs):
    """
    Create a NumPy array with integer dtype safely across versions.
    
    Args:
        data: Input data for the array
        **kwargs: Additional arguments for np.array
        
    Returns:
        NumPy array with appropriate integer dtype
    """
    if NUMPY_1_20_PLUS:
        return np.array(data, dtype=int, **kwargs)
    else:
        return np.array(data, dtype=np.int, **kwargs)

def safe_numpy_array_float(data, **kwargs):
    """
    Create a NumPy array with float dtype safely across versions.
    
    Args:
        data: Input data for the array
        **kwargs: Additional arguments for np.array
        
    Returns:
        NumPy array with appropriate float dtype
    """
    if NUMPY_1_20_PLUS:
        return np.array(data, dtype=float, **kwargs)
    else:
        return np.array(data, dtype=np.float, **kwargs)

def patch_vina_numpy_compatibility():
    """
    Patch Vina-related modules for NumPy compatibility.
    This function specifically addresses issues in the Vina Python bindings.
    """
    try:
        import vina
        
        # Check if Vina module has NumPy compatibility issues
        if hasattr(vina, '_vina') and NUMPY_1_20_PLUS:
            # Monkey patch the Vina module if needed
            original_vina_init = getattr(vina.Vina, '__init__', None)
            
            if original_vina_init:
                def patched_vina_init(self, *args, **kwargs):
                    # Temporarily suppress NumPy warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
                        warnings.filterwarnings("ignore", message=".*np.int.*deprecated.*")
                        return original_vina_init(self, *args, **kwargs)
                
                vina.Vina.__init__ = patched_vina_init
                print("✅ Vina NumPy compatibility patch applied")
        
    except ImportError:
        print("⚠️  Vina module not available for patching")
    except Exception as e:
        print(f"⚠️  Vina patching failed: {e}")

def suppress_numpy_warnings():
    """
    Suppress NumPy deprecation warnings that are not actionable by the user.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
    warnings.filterwarnings("ignore", message=".*np.int.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np.float.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np.complex.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np.bool.*deprecated.*")

def initialize_numpy_compatibility():
    """
    Initialize all NumPy compatibility fixes.
    This is the main function to call for complete compatibility setup.
    """
    print("🔧 Initializing NumPy compatibility layer...")
    
    # Set up basic compatibility
    setup_numpy_compatibility()
    
    # Suppress warnings
    suppress_numpy_warnings()
    
    # Patch Vina if available
    patch_vina_numpy_compatibility()
    
    print("✅ NumPy compatibility initialization complete")

# Compatibility constants for common use cases
if NUMPY_1_20_PLUS:
    NUMPY_INT_TYPE = int
    NUMPY_FLOAT_TYPE = float
    NUMPY_INT64_TYPE = np.int64
    NUMPY_FLOAT64_TYPE = np.float64
else:
    NUMPY_INT_TYPE = np.int
    NUMPY_FLOAT_TYPE = np.float
    NUMPY_INT64_TYPE = np.int64
    NUMPY_FLOAT64_TYPE = np.float64

# Export commonly used functions
__all__ = [
    'setup_numpy_compatibility',
    'safe_numpy_int',
    'safe_numpy_float', 
    'safe_numpy_array_int',
    'safe_numpy_array_float',
    'patch_vina_numpy_compatibility',
    'suppress_numpy_warnings',
    'initialize_numpy_compatibility',
    'NUMPY_VERSION',
    'NUMPY_1_20_PLUS',
    'NUMPY_INT_TYPE',
    'NUMPY_FLOAT_TYPE',
    'NUMPY_INT64_TYPE',
    'NUMPY_FLOAT64_TYPE'
]

# Auto-initialize if imported directly
if __name__ != "__main__":
    # Only auto-initialize if not being run as a script
    try:
        initialize_numpy_compatibility()
    except Exception as e:
        print(f"⚠️  NumPy compatibility auto-initialization failed: {e}")
        print("   Manual initialization may be required")
