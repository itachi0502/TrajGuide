"""
SOTA级模型兼容性检查和修复工具
解决模型架构不匹配、维度错误等问题

作者：SOTA级优化版本
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class ModelCompatibilityChecker:
    """SOTA级模型兼容性检查器"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        
    def diagnose_model_architecture(self) -> Dict[str, Any]:
        """全面诊断模型架构"""
        diagnosis = {
            'model_type': type(self.model).__name__,
            'dynamics_type': type(self.model.dynamics).__name__ if hasattr(self.model, 'dynamics') else None,
            'parameters': {},
            'methods': [],
            'embedding_layers': {},
            'issues': [],
            'recommendations': []
        }
        
        # 检查dynamics模块
        if hasattr(self.model, 'dynamics'):
            dynamics = self.model.dynamics
            
            # 基本参数
            diagnosis['parameters']['num_classes'] = getattr(dynamics, 'num_classes', None)
            diagnosis['parameters']['num_bond_classes'] = getattr(dynamics, 'num_bond_classes', None)
            diagnosis['parameters']['time_emb_dim'] = getattr(dynamics, 'time_emb_dim', None)
            diagnosis['parameters']['bond_bfn'] = getattr(dynamics, 'bond_bfn', None)
            
            # 检查嵌入层
            if hasattr(dynamics, 'ligand_atom_emb'):
                emb_weight = dynamics.ligand_atom_emb.weight
                diagnosis['embedding_layers']['ligand_atom_emb'] = {
                    'shape': emb_weight.shape,
                    'input_dim': emb_weight.shape[1],
                    'output_dim': emb_weight.shape[0]
                }
            
            if hasattr(dynamics, 'protein_atom_emb'):
                emb_weight = dynamics.protein_atom_emb.weight
                diagnosis['embedding_layers']['protein_atom_emb'] = {
                    'shape': emb_weight.shape,
                    'input_dim': emb_weight.shape[1],
                    'output_dim': emb_weight.shape[0]
                }
            
            # 检查时间嵌入
            if hasattr(dynamics, 'time_emb_layer'):
                diagnosis['parameters']['has_time_emb_layer'] = True
                if hasattr(dynamics.time_emb_layer, 'time_emb_dim'):
                    diagnosis['parameters']['time_emb_layer_dim'] = dynamics.time_emb_layer.time_emb_dim
            else:
                diagnosis['parameters']['has_time_emb_layer'] = False
            
            # 检查可用方法
            methods = ['forward', 'interdependency_modeling', 'loss_one_step', 'sample']
            for method in methods:
                if hasattr(dynamics, method):
                    diagnosis['methods'].append(method)
        
        # 检查训练相关方法
        if hasattr(self.model, 'training_step'):
            diagnosis['methods'].append('training_step')
        
        return diagnosis
    
    def check_input_compatibility(self, batch) -> Dict[str, Any]:
        """检查输入数据与模型的兼容性"""
        compatibility = {
            'ligand_data': {},
            'protein_data': {},
            'issues': [],
            'fixes': []
        }
        
        # 检查配体数据
        if hasattr(batch, 'ligand_pos'):
            compatibility['ligand_data']['pos_shape'] = batch.ligand_pos.shape
        if hasattr(batch, 'ligand_atom_feature_full'):
            compatibility['ligand_data']['feature_shape'] = batch.ligand_atom_feature_full.shape
        if hasattr(batch, 'ligand_element'):
            compatibility['ligand_data']['element_shape'] = batch.ligand_element.shape
        
        # 检查蛋白质数据
        if hasattr(batch, 'protein_pos'):
            compatibility['protein_data']['pos_shape'] = batch.protein_pos.shape
        if hasattr(batch, 'protein_atom_feature'):
            compatibility['protein_data']['feature_shape'] = batch.protein_atom_feature.shape
        
        # 检查维度兼容性
        if hasattr(self.model, 'dynamics') and hasattr(self.model.dynamics, 'ligand_atom_emb'):
            expected_ligand_dim = self.model.dynamics.ligand_atom_emb.weight.shape[1]
            
            if hasattr(batch, 'ligand_atom_feature_full'):
                actual_ligand_dim = batch.ligand_atom_feature_full.shape[1]
                time_dim = getattr(self.model.dynamics, 'time_emb_dim', 1)
                
                total_expected = expected_ligand_dim
                total_actual = actual_ligand_dim + time_dim
                
                if total_actual != total_expected:
                    compatibility['issues'].append({
                        'type': 'dimension_mismatch',
                        'component': 'ligand_features',
                        'expected': total_expected,
                        'actual': total_actual,
                        'difference': total_actual - total_expected
                    })
                    
                    if total_actual > total_expected:
                        compatibility['fixes'].append({
                            'type': 'truncate',
                            'component': 'ligand_features',
                            'action': f'截断到 {total_expected - time_dim} 维'
                        })
                    else:
                        compatibility['fixes'].append({
                            'type': 'pad',
                            'component': 'ligand_features',
                            'action': f'填充到 {total_expected - time_dim} 维'
                        })
        
        return compatibility
    
    def fix_input_dimensions(self, batch, t_tensor) -> Tuple[Any, torch.Tensor]:
        """修复输入维度不匹配"""
        fixed_batch = batch
        
        if hasattr(self.model, 'dynamics') and hasattr(self.model.dynamics, 'ligand_atom_emb'):
            expected_ligand_dim = self.model.dynamics.ligand_atom_emb.weight.shape[1]
            time_dim = t_tensor.shape[1]
            
            if hasattr(batch, 'ligand_atom_feature_full'):
                ligand_v = batch.ligand_atom_feature_full
                current_dim = ligand_v.shape[1]
                target_dim = expected_ligand_dim - time_dim
                
                if current_dim != target_dim:
                    if current_dim > target_dim:
                        # 截断
                        ligand_v_fixed = ligand_v[:, :target_dim]
                        print(f"🔧 截断配体特征: {ligand_v.shape} -> {ligand_v_fixed.shape}")
                    else:
                        # 填充
                        padding_dim = target_dim - current_dim
                        padding = torch.zeros(ligand_v.shape[0], padding_dim, 
                                            device=ligand_v.device, dtype=ligand_v.dtype)
                        ligand_v_fixed = torch.cat([ligand_v, padding], dim=1)
                        print(f"🔧 填充配体特征: {ligand_v.shape} -> {ligand_v_fixed.shape}")
                    
                    # 更新batch中的特征
                    fixed_batch.ligand_atom_feature_full = ligand_v_fixed
        
        return fixed_batch, t_tensor
    
    def suggest_compatible_method(self) -> str:
        """建议最兼容的调用方法"""
        diagnosis = self.diagnose_model_architecture()
        
        # 按优先级排序
        if 'loss_one_step' in diagnosis['methods']:
            return 'loss_one_step'
        elif 'training_step' in diagnosis['methods']:
            return 'training_step'
        elif 'interdependency_modeling' in diagnosis['methods']:
            return 'interdependency_modeling'
        elif 'forward' in diagnosis['methods']:
            return 'forward'
        else:
            return 'none'
    
    def create_compatible_call_kwargs(self, batch, t_tensor, conditions=None) -> Dict[str, Any]:
        """创建兼容的调用参数"""
        method = self.suggest_compatible_method()
        
        if method == 'loss_one_step':
            return self._create_loss_one_step_kwargs(batch, t_tensor, conditions)
        elif method == 'training_step':
            return self._create_training_step_kwargs(batch, conditions)
        elif method == 'interdependency_modeling':
            return self._create_interdependency_kwargs(batch, t_tensor, conditions)
        else:
            return {}
    
    def _create_loss_one_step_kwargs(self, batch, t_tensor, conditions=None) -> Dict[str, Any]:
        """创建loss_one_step调用参数"""
        kwargs = {
            't': t_tensor,
            'protein_pos': getattr(batch, 'protein_pos', None),
            'protein_v': getattr(batch, 'protein_atom_feature', None),
            'batch_protein': getattr(batch, 'protein_element_batch', None),
            'ligand_pos': getattr(batch, 'ligand_pos', None),
            'ligand_v': getattr(batch, 'ligand_atom_feature_full', None),
            'batch_ligand': getattr(batch, 'ligand_element_batch', None),
            'ligand_bond_type': getattr(batch, 'ligand_fc_bond_type', None),
            'ligand_bond_index': getattr(batch, 'ligand_fc_bond_index', None),
            'batch_ligand_bond': getattr(batch, 'ligand_fc_bond_type_batch', None),
            'include_protein': True,
            't_pos': t_tensor,
        }
        
        if conditions is not None:
            kwargs['conditions'] = conditions
        
        return kwargs
    
    def _create_training_step_kwargs(self, batch, conditions=None) -> Dict[str, Any]:
        """创建training_step调用参数"""
        return {'batch': batch, 'batch_idx': 0}
    
    def _create_interdependency_kwargs(self, batch, t_tensor, conditions=None) -> Dict[str, Any]:
        """创建interdependency_modeling调用参数"""
        # 这里需要根据具体模型实现
        return {}


def create_compatibility_checker(model, device: str = 'cuda') -> ModelCompatibilityChecker:
    """创建模型兼容性检查器"""
    return ModelCompatibilityChecker(model, device)


def diagnose_model_compatibility(model, batch, device: str = 'cuda') -> Dict[str, Any]:
    """便捷函数：诊断模型兼容性"""
    checker = ModelCompatibilityChecker(model, device)
    
    model_diagnosis = checker.diagnose_model_architecture()
    input_compatibility = checker.check_input_compatibility(batch)
    
    return {
        'model': model_diagnosis,
        'input': input_compatibility,
        'suggested_method': checker.suggest_compatible_method()
    }
