import torch
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


try:
    from core.utils.molecular_condition_processor import get_global_condition_processor
    CONDITION_PROCESSOR_AVAILABLE = True
except ImportError:
    CONDITION_PROCESSOR_AVAILABLE = False


class AddMolecularConditions:

    def __init__(self, condition_config=None):
        self.condition_config = condition_config

        # 兼容Struct对象和字典对象
        if hasattr(condition_config, 'enabled'):
            # Struct对象
            self.enabled = getattr(condition_config, 'enabled', False)
            self.use_probability = getattr(condition_config, 'use_probability', 0.7)
            self.noise_std = getattr(condition_config, 'noise_std', 0.05)
            self.training_mode = getattr(condition_config, 'training_mode', True)
            self.dataset_name = getattr(condition_config, 'dataset_name', 'default')
        else:
            # 字典对象或None
            self.enabled = condition_config.get('enabled', False) if condition_config else False
            self.use_probability = condition_config.get('use_probability', 0.7) if condition_config else 0.7
            self.noise_std = condition_config.get('noise_std', 0.05) if condition_config else 0.05
            self.training_mode = condition_config.get('training_mode', True) if condition_config else True
            self.dataset_name = condition_config.get('dataset_name', 'default') if condition_config else 'default'

        # SOTA: 初始化条件处理器（使用正确的dataset_name）
        self.condition_processor = None
        if self.enabled and CONDITION_PROCESSOR_AVAILABLE:
            try:
                self.condition_processor = get_global_condition_processor(dataset_name=self.dataset_name)
                precomputed_count = len(self.condition_processor.dataset_conditions)
            except Exception as e:
                self.condition_processor = None

    
    def __call__(self, data):
        default_conditions = torch.tensor([0.5, 0.5], dtype=torch.float32)

        data.conditions = default_conditions.clone()

        if not self.enabled:
            return data

        try:
            if self.condition_processor is not None:
                return self._process_with_sota_processor(data)
            else:
                return self._process_with_basic_method(data)

        except Exception as e:
            return data

    def _process_with_sota_processor(self, data):
        try:
            if hasattr(data, 'ligand_filename') and data.ligand_filename:
                sample_id = data.ligand_filename

                has_precomputed = sample_id in self.condition_processor.dataset_conditions

                if has_precomputed:
                    normalized_conditions = self.condition_processor.get_conditions_by_sample_id(sample_id)

                    if self.training_mode and random.random() < self.use_probability:
                        if random.random() < 0.3:  
                            noise = torch.randn_like(normalized_conditions) * self.noise_std
                            data.conditions = torch.clamp(normalized_conditions + noise, -2.0, 2.0)
                        else:
                            data.conditions = normalized_conditions.clone()
                    else:
                        data.conditions = torch.zeros(2, dtype=torch.float32)

                    return data

            if hasattr(data, 'ligand_smiles') and data.ligand_smiles is not None:
                smiles = data.ligand_smiles

                normalized_conditions = self.condition_processor.get_normalized_conditions(smiles)

                if self.training_mode and random.random() < self.use_probability:
                    if random.random() < 0.3:  
                        noise = torch.randn_like(normalized_conditions) * self.noise_std
                        data.conditions = torch.clamp(normalized_conditions + noise, -2.0, 2.0)
                    else:
                        data.conditions = normalized_conditions.clone()
                else:
                    data.conditions = torch.zeros(2, dtype=torch.float32)

                return data

            return self._estimate_conditions_from_graph(data)

        except Exception as e:
            return self._process_with_basic_method(data)

    def _estimate_conditions_from_graph(self, data):

        try:
            if hasattr(data, 'ligand_element'):
                elements = data.ligand_element
                num_atoms = len(elements)

                qed_estimate = max(0.1, min(0.9, 0.8 - (num_atoms - 20) * 0.01))
                sa_estimate = min(0.8, num_atoms / 60.0) + 0.1 * np.random.random()

                estimated_conditions = torch.tensor([
                    qed_estimate, sa_estimate
                ], dtype=torch.float32)

                if self.training_mode and random.random() < self.use_probability:
                    data.conditions = estimated_conditions.clone()
                else:
                    data.conditions = torch.zeros(2, dtype=torch.float32)

                return data

        except Exception as e:
            print(f"failed: {e}")

        return self._process_with_basic_method(data)

    def _process_with_basic_method(self, data):
        default_conditions = torch.tensor([0.5, 0.5], dtype=torch.float32)

        try:
            ligand_mol = self._extract_ligand_mol(data)

            if ligand_mol is not None:
                molecular_conditions = self._extract_molecular_conditions(ligand_mol)


                if self.training_mode and random.random() < self.use_probability:
                    data.conditions = molecular_conditions.clone()
                    if random.random() < 0.3:
                        noise = torch.randn_like(data.conditions) * self.noise_std
                        data.conditions += noise

                        data.conditions = torch.clamp(data.conditions, -2.0, 2.0)
                else:
                    data.conditions = torch.zeros(2, dtype=torch.float32)
            else:

                data.conditions = torch.zeros(2, dtype=torch.float32)  

        except Exception as e:
            data.conditions = torch.zeros(2, dtype=torch.float32)

        if not isinstance(data.conditions, torch.Tensor):
            data.conditions = torch.tensor(data.conditions, dtype=torch.float32)

        if data.conditions.shape != (2,):
            data.conditions = default_conditions.clone()

        return data
    
    def _extract_ligand_mol(self, data):
        """从数据中提取配体分子"""
        try:
            if hasattr(data, 'ligand_smiles') and data.ligand_smiles:
                return Chem.MolFromSmiles(data.ligand_smiles)
            

            if hasattr(data, 'ligand_mol') and data.ligand_mol:
                return data.ligand_mol
            
            if hasattr(data, 'ligand_filename'):
                pass
            

            
            return None
            
        except Exception as e:
            print(f"failed: {e}")
            return None
    
    def _extract_molecular_conditions(self, mol):
        try:
            try:
                from rdkit.Contrib.SA_Score import sascorer
                sa_score = min(sascorer.calculateScore(mol), 10.0) / 10.0
            except ImportError:
                sa_score = 0.5
            
            conditions = [
                QED.qed(mol),                                    
                sa_score                                         
            ]
            return torch.tensor(conditions, dtype=torch.float32)
        except Exception as e:
            return torch.tensor([0.5, 0.5], dtype=torch.float32)


class ConditionalDataAugmentation:
    
    def __init__(self, augment_prob=0.1):
        self.augment_prob = augment_prob
    
    def __call__(self, data):
        if not hasattr(data, 'conditions') or data.conditions is None:
            return data
        
        if random.random() < self.augment_prob:
            noise = torch.randn_like(data.conditions) * 0.02
            data.conditions += noise
            data.conditions = torch.clamp(data.conditions, -2.0, 2.0)
        
        return data


def create_condition_aware_transform(base_transform, condition_config):
    from torch_geometric.transforms import Compose
    
    if isinstance(base_transform, Compose):
        transform_list = base_transform.transforms.copy()
    else:
        transform_list = [base_transform] if base_transform else []
    
    condition_transform = AddMolecularConditions(condition_config)
    transform_list.append(condition_transform)
    
    if hasattr(condition_config, 'augment_prob'):
        augment_prob = getattr(condition_config, 'augment_prob', 0.0)
        if augment_prob > 0:
            augment_transform = ConditionalDataAugmentation(augment_prob)
            transform_list.append(augment_transform)
    
    return Compose(transform_list)


def add_condition_awareness_to_transform(transform, condition_config):
    return create_condition_aware_transform(transform, condition_config)
