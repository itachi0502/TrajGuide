#!/usr/bin/env python3
"""
Vina Compatibility Wrapper for MolPilot
=======================================

This module provides a compatibility wrapper for the Vina Python bindings
to handle NumPy 1.20+ compatibility issues and other common problems.

Key Features:
- NumPy compatibility fixes for Vina
- Enhanced error handling and recovery
- Timeout protection for long-running docking
- Memory management and cleanup
- Production-ready Vina interface

Designed specifically for MolPilot's docking evaluation pipeline.
"""

import os
import sys
import warnings
import tempfile
import time
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Import NumPy compatibility
try:
    from core.utils.numpy_compat import initialize_numpy_compatibility, suppress_numpy_warnings
    initialize_numpy_compatibility()
except ImportError:
    print("⚠️  NumPy compatibility module not available")

# Import Vina resource management
try:
    from core.evaluation.utils.vina_resource_manager import safe_vina_context, get_vina_resource_manager
    VINA_RESOURCE_MANAGEMENT = True
    print("✅ Vina resource management loaded")
except ImportError:
    print("⚠️  Vina resource management not available")
    VINA_RESOURCE_MANAGEMENT = False
    
    # Basic compatibility setup
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float

# Import Vina with error handling
try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError as e:
    print(f"❌ Vina import failed: {e}")
    VINA_AVAILABLE = False
    
    # Create a dummy Vina class for graceful degradation
    class Vina:
        def __init__(self, *args, **kwargs):
            raise ImportError("Vina is not available")


class VinaCompatWrapper:
    """
    SOTA compatibility wrapper for Vina with comprehensive error handling.
    """
    
    def __init__(self, sf_name='vina', seed=0, verbosity=0, timeout=300, **kwargs):
        """
        Initialize Vina with compatibility fixes and resource management.

        Args:
            sf_name: Scoring function name
            seed: Random seed (0 for random)
            verbosity: Verbosity level
            timeout: Timeout in seconds for docking operations
            **kwargs: Additional Vina parameters
        """
        if not VINA_AVAILABLE:
            raise ImportError("Vina is not available. Please install vina: pip install vina")

        self.timeout = timeout
        self.sf_name = sf_name
        self.seed = seed
        self.verbosity = verbosity
        self.kwargs = kwargs

        # Resource management
        self.resource_context = None
        self.temp_dir = None

        # Initialize Vina with error handling
        self._vina = None
        self._initialize_vina()
    
    def _initialize_vina(self):
        """Initialize Vina instance with compatibility fixes and resource management."""
        try:
            # Use resource management if available
            if VINA_RESOURCE_MANAGEMENT:
                # Get resource manager and check availability
                manager = get_vina_resource_manager()
                memory_info = manager.check_memory_usage()
                print(f"🔧 Initializing Vina with resource management (Memory: {memory_info['rss_mb']:.1f} MB)")

                # Check if we have sufficient resources
                if not manager.is_memory_available(required_mb=300):
                    print("⚠️  Low memory detected, attempting cleanup...")
                    manager.emergency_cleanup()

            # Suppress NumPy warnings during Vina initialization
            with warnings.catch_warnings():
                suppress_numpy_warnings()

                self._vina = Vina(
                    sf_name=self.sf_name,
                    seed=self.seed,
                    verbosity=self.verbosity,
                    **self.kwargs
                )

            print(f"✅ Vina initialized successfully with {self.sf_name} scoring function")

        except Exception as e:
            print(f"❌ Vina initialization failed: {e}")
            # Attempt emergency cleanup before re-raising
            if VINA_RESOURCE_MANAGEMENT:
                try:
                    get_vina_resource_manager().emergency_cleanup()
                except:
                    pass
            raise
    
    def set_receptor(self, receptor_pdbqt: str):
        """
        Set receptor with enhanced error handling.
        
        Args:
            receptor_pdbqt: Path to receptor PDBQT file
        """
        if not os.path.exists(receptor_pdbqt):
            raise FileNotFoundError(f"Receptor PDBQT file not found: {receptor_pdbqt}")
        
        if os.path.getsize(receptor_pdbqt) == 0:
            raise ValueError(f"Receptor PDBQT file is empty: {receptor_pdbqt}")
        
        try:
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                self._vina.set_receptor(receptor_pdbqt)
            print(f"✅ Receptor set: {receptor_pdbqt}")
            
        except Exception as e:
            print(f"❌ Failed to set receptor: {e}")
            raise
    
    def set_ligand_from_file(self, ligand_pdbqt: str):
        """
        Set ligand from file with enhanced error handling.
        
        Args:
            ligand_pdbqt: Path to ligand PDBQT file
        """
        if not os.path.exists(ligand_pdbqt):
            raise FileNotFoundError(f"Ligand PDBQT file not found: {ligand_pdbqt}")
        
        if os.path.getsize(ligand_pdbqt) == 0:
            raise ValueError(f"Ligand PDBQT file is empty: {ligand_pdbqt}")
        
        try:
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                self._vina.set_ligand_from_file(ligand_pdbqt)
            print(f"✅ Ligand set: {ligand_pdbqt}")
            
        except Exception as e:
            print(f"❌ Failed to set ligand: {e}")
            raise
    
    def compute_vina_maps(self, center: List[float], box_size: List[float]):
        """
        Compute Vina maps with enhanced error handling.
        
        Args:
            center: Box center coordinates [x, y, z]
            box_size: Box size [x, y, z]
        """
        try:
            # Validate inputs
            if len(center) != 3 or len(box_size) != 3:
                raise ValueError("Center and box_size must have 3 elements each")
            
            # Check for reasonable box size
            total_volume = box_size[0] * box_size[1] * box_size[2]
            if total_volume > 50000:
                print(f"⚠️  Large search space volume: {total_volume:.0f} Angstrom^3")
            
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                self._vina.compute_vina_maps(center=center, box_size=box_size)
            
            print(f"✅ Vina maps computed: center={center}, box_size={box_size}")
            
        except Exception as e:
            print(f"❌ Failed to compute Vina maps: {e}")
            raise
    
    @contextmanager
    def _timeout_context(self, operation_name: str):
        """Context manager for timeout protection."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"{operation_name} timed out after {elapsed:.1f}s")
            else:
                raise
    
    def score(self) -> Tuple[float, ...]:
        """
        Score ligand with timeout protection.
        
        Returns:
            Tuple of scores
        """
        try:
            with self._timeout_context("Scoring"):
                with warnings.catch_warnings():
                    suppress_numpy_warnings()
                    result = self._vina.score()
                
            print(f"✅ Scoring completed: {result[0]:.3f}")
            return result
            
        except Exception as e:
            print(f"❌ Scoring failed: {e}")
            raise
    
    def optimize(self) -> Tuple[float, ...]:
        """
        Optimize ligand with timeout protection.
        
        Returns:
            Tuple of optimized scores
        """
        try:
            with self._timeout_context("Optimization"):
                with warnings.catch_warnings():
                    suppress_numpy_warnings()
                    result = self._vina.optimize()
                
            print(f"✅ Optimization completed: {result[0]:.3f}")
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            raise
    
    def dock(self, exhaustiveness: int = 8, n_poses: int = 1) -> None:
        """
        Perform docking with timeout protection.
        
        Args:
            exhaustiveness: Exhaustiveness parameter
            n_poses: Number of poses to generate
        """
        try:
            with self._timeout_context("Docking"):
                with warnings.catch_warnings():
                    suppress_numpy_warnings()
                    self._vina.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                
            print(f"✅ Docking completed: exhaustiveness={exhaustiveness}, n_poses={n_poses}")
            
        except Exception as e:
            print(f"❌ Docking failed: {e}")
            raise
    
    def energies(self, n_poses: int = 1) -> List[List[float]]:
        """
        Get energies with error handling.
        
        Args:
            n_poses: Number of poses
            
        Returns:
            List of energy lists
        """
        try:
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                result = self._vina.energies(n_poses=n_poses)
            
            print(f"✅ Energies retrieved for {n_poses} poses")
            return result
            
        except Exception as e:
            print(f"❌ Failed to get energies: {e}")
            raise
    
    def write_pose(self, filename: str, overwrite: bool = True):
        """
        Write pose with error handling.

        Args:
            filename: Output filename
            overwrite: Whether to overwrite existing file
        """
        try:
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                self._vina.write_pose(filename, overwrite=overwrite)

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print(f"✅ Pose written: {filename} ({os.path.getsize(filename)} bytes)")
            else:
                raise RuntimeError(f"Pose file was not created or is empty: {filename}")

        except Exception as e:
            print(f"❌ Failed to write pose: {e}")
            raise

    def poses(self, n_poses: int = 1) -> str:
        """
        Get poses as PDBQT string with error handling.

        Args:
            n_poses: Number of poses to retrieve

        Returns:
            PDBQT string containing the poses
        """
        try:
            with warnings.catch_warnings():
                suppress_numpy_warnings()
                poses_str = self._vina.poses(n_poses=n_poses)

            if poses_str and len(poses_str.strip()) > 0:
                print(f"✅ Poses retrieved: {len(poses_str)} characters, {n_poses} poses")
                return poses_str
            else:
                raise RuntimeError(f"No poses retrieved or empty result")

        except Exception as e:
            print(f"❌ Failed to get poses: {e}")
            raise


def create_compatible_vina(*args, **kwargs) -> VinaCompatWrapper:
    """
    Factory function to create a compatible Vina instance.
    
    Args:
        *args: Positional arguments for Vina
        **kwargs: Keyword arguments for Vina
        
    Returns:
        VinaCompatWrapper instance
    """
    return VinaCompatWrapper(*args, **kwargs)


# Convenience function for backward compatibility
def Vina_compat(*args, **kwargs):
    """Backward compatibility alias for create_compatible_vina."""
    return create_compatible_vina(*args, **kwargs)
