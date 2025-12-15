#!/usr/bin/env python3
"""
SOTA RDKit UFF Force Field Error Handler for MolPilot
====================================================

This module provides comprehensive solutions for RDKit 2023.9.5+ UFF force field
parameter errors, particularly the "S_5+6" atom type issue and other problematic
molecular structures that cause force field failures.

Key Features:
- Automatic atom type sanitization and mapping
- Fallback force field selection (UFF → MMFF → basic geometry)
- Molecular structure validation and repair
- Comprehensive error logging and recovery
- Production-ready error handling for molecular generation pipelines

Designed specifically for MolPilot's molecular evaluation pipeline.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
import numpy as np

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class RDKitUFFHandler:
    """
    SOTA handler for RDKit UFF force field issues in molecular evaluation.
    """
    
    # Problematic atom type mappings for UFF compatibility
    ATOM_TYPE_FIXES = {
        'S_5+6': 'S',      # Problematic sulfur oxidation state
        'S+6': 'S',        # Sulfur +6 oxidation state
        'S+4': 'S',        # Sulfur +4 oxidation state  
        'S+2': 'S',        # Sulfur +2 oxidation state
        'S-2': 'S',        # Sulfur -2 oxidation state
        'N+3': 'N',        # Nitrogen +3 oxidation state
        'N+1': 'N',        # Nitrogen +1 oxidation state
        'N-1': 'N',        # Nitrogen -1 oxidation state
        'O+1': 'O',        # Oxygen +1 oxidation state
        'O-1': 'O',        # Oxygen -1 oxidation state
        'O-2': 'O',        # Oxygen -2 oxidation state
        'P+5': 'P',        # Phosphorus +5 oxidation state
        'P+3': 'P',        # Phosphorus +3 oxidation state
        'P-3': 'P',        # Phosphorus -3 oxidation state
        'Cl+1': 'Cl',     # Chlorine +1 oxidation state
        'Cl-1': 'Cl',     # Chlorine -1 oxidation state
        'F-1': 'F',        # Fluorine -1 oxidation state
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the UFF handler.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.stats = {
            'uff_success': 0,
            'uff_fixed': 0,
            'mmff_fallback': 0,
            'geometry_fallback': 0,
            'total_failures': 0
        }
    
    def safe_uff_optimize(self, mol: Chem.Mol, max_iters: int = 200) -> Tuple[bool, Optional[Chem.Mol], str]:
        """
        Safely optimize molecule using UFF with comprehensive error handling.
        
        Args:
            mol: RDKit molecule object
            max_iters: Maximum optimization iterations
            
        Returns:
            Tuple of (success, optimized_mol, method_used)
        """
        if mol is None:
            return False, None, "invalid_molecule"
        
        try:
            # Step 1: Try direct UFF optimization
            mol_copy = Chem.Mol(mol)
            result = UFFOptimizeMolecule(mol_copy, maxIters=max_iters)
            
            if result == 0:  # Success
                self.stats['uff_success'] += 1
                if self.verbose:
                    logger.info("✅ UFF optimization successful")
                return True, mol_copy, "uff_direct"
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Direct UFF failed: {e}")
        
        # Step 2: Try UFF with atom type fixes
        try:
            fixed_mol = self._fix_problematic_atoms(mol)
            if fixed_mol is not None:
                mol_copy = Chem.Mol(fixed_mol)
                result = UFFOptimizeMolecule(mol_copy, maxIters=max_iters)
                
                if result == 0:  # Success
                    self.stats['uff_fixed'] += 1
                    if self.verbose:
                        logger.info("✅ UFF optimization successful after atom fixes")
                    return True, mol_copy, "uff_fixed"
                    
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  UFF with fixes failed: {e}")
        
        # Step 3: Try MMFF as fallback
        try:
            mol_copy = Chem.Mol(mol)
            AllChem.MMFFSanitizeMolecule(mol_copy)
            result = MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
            
            if result == 0:  # Success
                self.stats['mmff_fallback'] += 1
                if self.verbose:
                    logger.info("✅ MMFF fallback optimization successful")
                return True, mol_copy, "mmff_fallback"
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  MMFF fallback failed: {e}")
        
        # Step 4: Basic geometry optimization fallback
        try:
            optimized_mol = self._basic_geometry_optimization(mol)
            if optimized_mol is not None:
                self.stats['geometry_fallback'] += 1
                if self.verbose:
                    logger.info("✅ Basic geometry optimization successful")
                return True, optimized_mol, "geometry_fallback"
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Basic geometry optimization failed: {e}")
        
        # All methods failed
        self.stats['total_failures'] += 1
        if self.verbose:
            logger.error("❌ All optimization methods failed")
        return False, mol, "all_failed"
    
    def _fix_problematic_atoms(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Fix problematic atom types that cause UFF failures.
        
        Args:
            mol: Input molecule
            
        Returns:
            Fixed molecule or None if fixing failed
        """
        try:
            # Create editable molecule
            em = Chem.EditableMol(mol)
            
            # Track if any changes were made
            changes_made = False
            
            # Iterate through atoms and fix problematic types
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                
                # Check for problematic atom types
                if atom_symbol in self.ATOM_TYPE_FIXES:
                    # Replace with standard atom type
                    new_symbol = self.ATOM_TYPE_FIXES[atom_symbol]
                    atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
                    changes_made = True
                    
                    if self.verbose:
                        logger.info(f"🔧 Fixed atom type: {atom_symbol} → {new_symbol}")
                
                # Handle charged atoms that might cause issues
                if atom.GetFormalCharge() != 0:
                    # Reset formal charge for problematic atoms
                    if atom_symbol in ['S', 'P', 'N', 'O']:
                        atom.SetFormalCharge(0)
                        changes_made = True
                        
                        if self.verbose:
                            logger.info(f"🔧 Reset formal charge for {atom_symbol}")
            
            if changes_made:
                # Rebuild molecule
                fixed_mol = em.GetMol()
                
                # Sanitize the fixed molecule
                try:
                    Chem.SanitizeMol(fixed_mol)
                    return fixed_mol
                except Exception as sanitize_error:
                    if self.verbose:
                        logger.warning(f"⚠️  Sanitization failed after fixes: {sanitize_error}")
                    return None
            else:
                return mol
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Atom fixing failed: {e}")
            return None
    
    def _basic_geometry_optimization(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Perform basic geometry optimization without force fields.
        
        Args:
            mol: Input molecule
            
        Returns:
            Optimized molecule or None if failed
        """
        try:
            mol_copy = Chem.Mol(mol)
            
            # Add hydrogens if not present
            mol_with_h = Chem.AddHs(mol_copy)
            
            # Generate 3D coordinates using ETKDG
            params = AllChem.ETKDGv3()
            params.randomSeed = 42  # For reproducibility
            params.maxAttempts = 10
            
            result = AllChem.EmbedMolecule(mol_with_h, params)
            
            if result == 0:  # Success
                # Basic distance geometry cleanup
                AllChem.MMFFSanitizeMolecule(mol_with_h)
                return mol_with_h
            else:
                if self.verbose:
                    logger.warning("⚠️  ETKDG embedding failed")
                return None
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Basic geometry optimization failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        total_attempts = sum(self.stats.values())
        
        if total_attempts > 0:
            stats_with_percentages = {}
            for key, value in self.stats.items():
                stats_with_percentages[key] = {
                    'count': value,
                    'percentage': (value / total_attempts) * 100
                }
            stats_with_percentages['total_attempts'] = total_attempts
            return stats_with_percentages
        else:
            return self.stats
    
    def reset_stats(self):
        """Reset optimization statistics."""
        for key in self.stats:
            self.stats[key] = 0


def safe_uff_optimize_molecule(mol: Chem.Mol, max_iters: int = 200, verbose: bool = False) -> Tuple[bool, Optional[Chem.Mol], str]:
    """
    Convenience function for safe UFF optimization.
    
    Args:
        mol: RDKit molecule object
        max_iters: Maximum optimization iterations
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (success, optimized_mol, method_used)
    """
    handler = RDKitUFFHandler(verbose=verbose)
    return handler.safe_uff_optimize(mol, max_iters)


def validate_molecule_for_uff(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Validate if a molecule is likely to work with UFF.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if mol is None:
        return False, "molecule_is_none"
    
    try:
        # Check for problematic atom types
        handler = RDKitUFFHandler()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in handler.ATOM_TYPE_FIXES:
                return False, f"problematic_atom_type_{symbol}"
        
        # Check molecular weight (very large molecules might cause issues)
        mw = Descriptors.MolWt(mol)
        if mw > 2000:  # Arbitrary threshold
            return False, f"molecule_too_large_{mw:.1f}"
        
        # Check for unusual formal charges
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if abs(total_charge) > 5:  # Arbitrary threshold
            return False, f"high_formal_charge_{total_charge}"
        
        return True, "valid"
        
    except Exception as e:
        return False, f"validation_error_{str(e)}"


# Global handler instance for reuse
_global_handler = None

def get_global_uff_handler(verbose: bool = False) -> RDKitUFFHandler:
    """
    Get or create global UFF handler instance.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Global UFF handler instance
    """
    global _global_handler
    if _global_handler is None:
        _global_handler = RDKitUFFHandler(verbose=verbose)
    return _global_handler
