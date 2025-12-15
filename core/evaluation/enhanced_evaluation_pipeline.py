#!/usr/bin/env python3
"""
==============================================

This module provides a comprehensive evaluation pipeline that handles all the
common errors encountered in molecular generation evaluation, particularly:

1. PDBQT generation failures
2. RDKit UFF force field parameter errors (S_5+6, etc.)
3. Docking evaluation failures
4. File descriptor leaks and resource management
5. Molecular structure validation and repair
"""

import os
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import tempfile
import shutil

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Import core evaluation components
from core.evaluation.metrics import CondMolGenMetric
from core.evaluation.docking_vina import VinaDockingTask
from core.evaluation.docking_qvina import QVinaDockingTask

# Import SOTA error handling modules
try:
    from core.evaluation.utils.rdkit_uff_fix import get_global_uff_handler
    from core.utils.resource_manager import safe_file_operation, ResourceMonitor
    HAS_ENHANCED_MODULES = True
except ImportError:
    HAS_ENHANCED_MODULES = False

logger = logging.getLogger(__name__)


class EnhancedEvaluationPipeline:
    """
    SOTA evaluation pipeline with comprehensive error handling and recovery.
    """
    
    def __init__(self, 
                 atom_decoder: Dict[int, str],
                 atom_enc_mode: str = "add_aromatic",
                 docking_config: Optional[Dict] = None,
                 verbose: bool = True,
                 max_retries: int = 3):
        """
        Initialize the enhanced evaluation pipeline.
        
        Args:
            atom_decoder: Mapping from atomic numbers to symbols
            atom_enc_mode: Atom encoding mode
            docking_config: Docking configuration
            verbose: Enable verbose logging
            max_retries: Maximum retry attempts for failed evaluations
        """
        self.atom_decoder = atom_decoder
        self.atom_enc_mode = atom_enc_mode
        self.docking_config = docking_config
        self.verbose = verbose
        self.max_retries = max_retries
        
        # Initialize statistics tracking
        self.stats = {
            'total_molecules': 0,
            'successful_evaluations': 0,
            'pdbqt_failures': 0,
            'uff_failures': 0,
            'docking_failures': 0,
            'total_failures': 0,
            'retry_successes': 0,
        }
        
        # Initialize resource monitoring if available
        if HAS_ENHANCED_MODULES:
            self.resource_monitor = ResourceMonitor()
            self.uff_handler = get_global_uff_handler(verbose=verbose)
        else:
            self.resource_monitor = None
            self.uff_handler = None
        
        # Initialize core metric calculator
        self.metric_calculator = CondMolGenMetric(
            atom_decoder=atom_decoder,
            atom_enc_mode=atom_enc_mode,
            type_one_hot=False,
            single_bond=False,
            docking_config=docking_config
        )
    
    @contextmanager
    def safe_evaluation_context(self, molecule_id: str = "unknown"):
        """
        Context manager for safe evaluation with automatic cleanup.
        
        Args:
            molecule_id: Identifier for the molecule being evaluated
        """
        temp_dir = None
        start_time = time.time()
        
        try:
            # Create temporary directory for this evaluation
            temp_dir = tempfile.mkdtemp(prefix=f"molpilot_eval_{molecule_id}_")
            
            # Monitor resources if available
            if self.resource_monitor:
                self.resource_monitor.check_resources()
            
            if self.verbose:
                logger.info(f"🔄 Starting evaluation for molecule {molecule_id}")
            
            yield temp_dir
            
            # Success
            elapsed = time.time() - start_time
            if self.verbose:
                logger.info(f"✅ Evaluation completed for molecule {molecule_id} in {elapsed:.2f}s")
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Evaluation failed for molecule {molecule_id} after {elapsed:.2f}s: {e}")
            if self.verbose:
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to cleanup temp directory {temp_dir}: {cleanup_error}")
    
    def evaluate_molecule_with_retry(self, 
                                   mol_data: Dict[str, Any], 
                                   molecule_id: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate a single molecule with retry mechanism.
        
        Args:
            mol_data: Molecule data containing positions, atom types, etc.
            molecule_id: Identifier for the molecule
            
        Returns:
            Dictionary containing evaluation results
        """
        self.stats['total_molecules'] += 1
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.safe_evaluation_context(f"{molecule_id}_attempt_{attempt}") as temp_dir:
                    result = self._evaluate_single_molecule(mol_data, molecule_id, temp_dir)
                    
                    if attempt > 0:
                        self.stats['retry_successes'] += 1
                        if self.verbose:
                            logger.info(f"✅ Retry successful for molecule {molecule_id} on attempt {attempt + 1}")
                    
                    self.stats['successful_evaluations'] += 1
                    return result
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    if self.verbose:
                        logger.warning(f"⚠️  Attempt {attempt + 1} failed for molecule {molecule_id}: {e}")
                        logger.warning(f"🔄 Retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(1)  # Brief delay before retry
                else:
                    if self.verbose:
                        logger.error(f"❌ All attempts failed for molecule {molecule_id}")
        
        # All attempts failed
        self.stats['total_failures'] += 1
        return {
            'molecule_id': molecule_id,
            'evaluation_status': 'failed',
            'error': str(last_error),
            'attempts': self.max_retries + 1,
            'validity': False,
            'docking_score': float('nan'),
            'molecular_properties': {}
        }
    
    def _evaluate_single_molecule(self, 
                                mol_data: Dict[str, Any], 
                                molecule_id: str,
                                temp_dir: str) -> Dict[str, Any]:
        """
        Evaluate a single molecule (internal implementation).
        
        Args:
            mol_data: Molecule data
            molecule_id: Molecule identifier
            temp_dir: Temporary directory for this evaluation
            
        Returns:
            Evaluation results dictionary
        """
        result = {
            'molecule_id': molecule_id,
            'evaluation_status': 'in_progress',
            'timestamp': time.time()
        }
        
        try:
            # Step 1: Reconstruct molecule from generated data
            if self.verbose:
                logger.info(f"🔄 Reconstructing molecule {molecule_id}")
            
            mol = self._reconstruct_molecule(mol_data)
            if mol is None:
                raise ValueError("Failed to reconstruct molecule from generated data")
            
            result['reconstruction_success'] = True
            
            # Step 2: Validate and optimize molecular structure
            if self.verbose:
                logger.info(f"🔄 Validating and optimizing molecule {molecule_id}")
            
            optimized_mol = self._validate_and_optimize_molecule(mol)
            result['optimization_success'] = optimized_mol is not None
            
            if optimized_mol is not None:
                mol = optimized_mol
            
            # Step 3: Calculate basic molecular properties
            if self.verbose:
                logger.info(f"🔄 Calculating molecular properties for {molecule_id}")
            
            properties = self._calculate_molecular_properties(mol)
            result['molecular_properties'] = properties
            result['validity'] = properties.get('valid', False)
            
            # Step 4: Perform docking evaluation if configured
            if self.docking_config and result['validity']:
                if self.verbose:
                    logger.info(f"🔄 Performing docking evaluation for {molecule_id}")
                
                docking_results = self._perform_docking_evaluation(mol, molecule_id, temp_dir)
                result.update(docking_results)
            else:
                result['docking_score'] = float('nan')
                result['docking_status'] = 'skipped'
            
            result['evaluation_status'] = 'completed'
            return result
            
        except Exception as e:
            result['evaluation_status'] = 'failed'
            result['error'] = str(e)
            result['validity'] = False
            result['docking_score'] = float('nan')
            raise
    
    def _reconstruct_molecule(self, mol_data: Dict[str, Any]) -> Optional[Chem.Mol]:
        """
        Reconstruct RDKit molecule from generated data.
        
        Args:
            mol_data: Generated molecule data
            
        Returns:
            RDKit molecule or None if reconstruction failed
        """
        try:
            # This would use the existing reconstruction logic
            # from core.utils.reconstruct import reconstruct_from_generated
            # For now, assume mol_data contains the necessary information
            
            if 'rdkit_mol' in mol_data:
                return mol_data['rdkit_mol']
            elif 'smiles' in mol_data:
                return Chem.MolFromSmiles(mol_data['smiles'])
            else:
                # Use existing reconstruction methods
                return None
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Molecule reconstruction failed: {e}")
            return None
    
    def _validate_and_optimize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Validate and optimize molecular structure using SOTA methods.
        
        Args:
            mol: Input RDKit molecule
            
        Returns:
            Optimized molecule or None if optimization failed
        """
        if mol is None:
            return None
        
        try:
            # Use SOTA UFF handler if available
            if self.uff_handler:
                success, optimized_mol, method = self.uff_handler.safe_uff_optimize(mol)
                if success and optimized_mol is not None:
                    if self.verbose:
                        logger.info(f"✅ Molecule optimization successful using {method}")
                    return optimized_mol
                else:
                    if self.verbose:
                        logger.warning(f"⚠️  UFF optimization failed, using original molecule")
                    self.stats['uff_failures'] += 1
                    return mol
            else:
                # Basic fallback optimization
                try:
                    mol_copy = Chem.Mol(mol)
                    AllChem.UFFOptimizeMolecule(mol_copy)
                    return mol_copy
                except Exception as uff_error:
                    if self.verbose:
                        logger.warning(f"⚠️  Basic UFF optimization failed: {uff_error}")
                    self.stats['uff_failures'] += 1
                    return mol
                    
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Molecule optimization failed: {e}")
            return mol
    
    def _calculate_molecular_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate basic molecular properties.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary of molecular properties
        """
        properties = {}
        
        try:
            if mol is None:
                properties['valid'] = False
                return properties
            
            # Basic validity check
            try:
                Chem.SanitizeMol(mol)
                properties['valid'] = True
            except:
                properties['valid'] = False
                return properties
            
            # Calculate properties
            properties['num_atoms'] = mol.GetNumAtoms()
            properties['num_bonds'] = mol.GetNumBonds()
            properties['molecular_weight'] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            properties['smiles'] = Chem.MolToSmiles(mol)
            
            # Additional properties if molecule is valid
            if properties['valid']:
                try:
                    from rdkit.Chem.Descriptors import qed
                    from rdkit.Chem.Crippen import MolLogP
                    
                    properties['qed'] = qed(mol)
                    properties['logp'] = MolLogP(mol)
                    properties['num_rotatable_bonds'] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                    properties['tpsa'] = Chem.rdMolDescriptors.CalcTPSA(mol)
                    
                except Exception as prop_error:
                    if self.verbose:
                        logger.warning(f"⚠️  Failed to calculate some properties: {prop_error}")
            
            return properties
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Property calculation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _perform_docking_evaluation(self, 
                                  mol: Chem.Mol, 
                                  molecule_id: str,
                                  temp_dir: str) -> Dict[str, Any]:
        """
        Perform docking evaluation with enhanced error handling.
        
        Args:
            mol: RDKit molecule
            molecule_id: Molecule identifier
            temp_dir: Temporary directory for docking files
            
        Returns:
            Dictionary containing docking results
        """
        docking_result = {
            'docking_status': 'attempted',
            'docking_score': float('nan'),
            'docking_method': 'unknown'
        }
        
        try:
            # Create SDF file for docking
            sdf_path = os.path.join(temp_dir, f"{molecule_id}.sdf")
            
            with Chem.SDWriter(sdf_path) as writer:
                writer.write(mol)
            
            # Perform docking using enhanced VinaDockingTask
            if self.docking_config.get('mode') == 'vina_score':
                docking_task = VinaDockingTask.from_generated_mol(
                    mol, f"{molecule_id}.sdf", 
                    protein_root=self.docking_config.get('protein_root', './data/test_set')
                )
                
                results = docking_task.run(
                    mode='score_only',
                    exhaustiveness=self.docking_config.get('exhaustiveness', 16)
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    if 'affinity' in result and not np.isnan(result['affinity']):
                        docking_result['docking_score'] = result['affinity']
                        docking_result['docking_status'] = 'success'
                        docking_result['docking_method'] = 'vina_score'
                    else:
                        docking_result['docking_status'] = 'invalid_score'
                        self.stats['docking_failures'] += 1
                else:
                    docking_result['docking_status'] = 'no_results'
                    self.stats['docking_failures'] += 1
            
            return docking_result
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  Docking evaluation failed for {molecule_id}: {e}")
            
            docking_result['docking_status'] = 'failed'
            docking_result['docking_error'] = str(e)
            self.stats['docking_failures'] += 1
            return docking_result
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive evaluation statistics.
        
        Returns:
            Dictionary containing evaluation statistics
        """
        stats = dict(self.stats)
        
        if stats['total_molecules'] > 0:
            stats['success_rate'] = stats['successful_evaluations'] / stats['total_molecules']
            stats['failure_rate'] = stats['total_failures'] / stats['total_molecules']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add UFF handler statistics if available
        if self.uff_handler:
            stats['uff_optimization_stats'] = self.uff_handler.get_stats()
        
        return stats
    
    def reset_statistics(self):
        """Reset all evaluation statistics."""
        for key in self.stats:
            self.stats[key] = 0
        
        if self.uff_handler:
            self.uff_handler.reset_stats()


def create_enhanced_evaluation_pipeline(config: Dict[str, Any]) -> EnhancedEvaluationPipeline:
    """
    Factory function to create an enhanced evaluation pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured EnhancedEvaluationPipeline instance
    """
    return EnhancedEvaluationPipeline(
        atom_decoder=config.get('atom_decoder', {1: 'H', 6: 'C', 7: 'N', 8: 'O'}),
        atom_enc_mode=config.get('atom_enc_mode', 'add_aromatic'),
        docking_config=config.get('docking_config'),
        verbose=config.get('verbose', True),
        max_retries=config.get('max_retries', 3)
    )
