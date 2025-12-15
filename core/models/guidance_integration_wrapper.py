import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from types import SimpleNamespace


class GuidanceIntegrationWrapper:
    
    def __init__(
        self,
        base_integrator,  # 现有的GeometricGuidanceIntegrator
        enable_booster: bool = True,
        booster_config: Optional[Dict] = None
    ):

        self.base_integrator = base_integrator
        self.enable_booster = enable_booster
        
        if enable_booster:
            from core.models.enhanced_guidance_booster import create_enhanced_guidance_booster
            
            booster_config = booster_config or {}
            self.booster = create_enhanced_guidance_booster(**booster_config)
        else:
            self.booster = None
        
        # 统计信息
        self.stats = {
            'total_calls': 0,
            'boosted_calls': 0,
            'avg_amplification': 0.0,
            'max_amplification': 0.0
        }
    
    def apply_multiplicative_guidance(
        self,
        theta_prime: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_conditions: torch.Tensor,
        guidance_scale: float = 1.0,
        alpha_h: float = None
    ) -> dict:

        self.stats['total_calls'] += 1
        
        if not self.enable_booster or self.booster is None:
            return self.base_integrator.apply_multiplicative_guidance(
                theta_prime, pos_t, t, batch_ligand, target_conditions,
                guidance_scale, alpha_h
            )
        
        try:
            current_properties = self._estimate_current_properties(
                theta_prime, pos_t, batch_ligand
            )
        except:
            current_properties = torch.tensor([[0.5, 0.5]], device=theta_prime.device)
        
        if t.dim() == 0:
            current_time = t.item()
        elif t.dim() == 1:
            current_time = t[0].item()
        else:
            current_time = t[0, 0].item()
        
        boost_result = self.booster.boost_guidance(
            theta_t=theta_prime,
            current_properties=current_properties,
            target_properties=target_conditions,
            current_time=current_time,
            original_guidance_scale=guidance_scale,
            original_guidance_logits=None  # 稍后从基础集成器获取
        )
        
        amplified_scale = boost_result.get('amplified_scale', guidance_scale)
        time_weight = boost_result.get('time_weight', 1.0)
        preference_logits = boost_result.get('preference_logits', None)
        
        # 更新统计信息
        self.stats['boosted_calls'] += 1
        amplification_ratio = amplified_scale / guidance_scale
        self.stats['avg_amplification'] = (
            (self.stats['avg_amplification'] * (self.stats['boosted_calls'] - 1) + amplification_ratio)
            / self.stats['boosted_calls']
        )
        self.stats['max_amplification'] = max(self.stats['max_amplification'], amplification_ratio)
        
        # 5. 调用基础集成器（使用增强后的参数）
        base_result = self.base_integrator.apply_multiplicative_guidance(
            theta_prime, pos_t, t, batch_ligand, target_conditions,
            amplified_scale,  # 使用放大后的引导强度
            alpha_h
        )
        
        if preference_logits is not None and 'guidance_probability' in base_result:
            guidance_prob = base_result['guidance_probability']
            
            if guidance_prob is not None:
                preference_adjustment = torch.softmax(preference_logits, dim=-1)
                

                mixing_ratio = min(time_weight * 0.3, 0.5)  # 最多50%的偏好引导
                
                adjusted_prob = (1 - mixing_ratio) * guidance_prob + mixing_ratio * preference_adjustment
                
                # 重新归一化
                adjusted_prob = adjusted_prob / (adjusted_prob.sum(dim=-1, keepdim=True) + 1e-8)
                
                base_result['guidance_probability'] = adjusted_prob
                base_result['preference_applied'] = True
                base_result['mixing_ratio'] = mixing_ratio
        
        # 7. 添加增强信息到结果
        base_result['boosted'] = True
        base_result['amplification_ratio'] = amplification_ratio
        base_result['time_weight'] = time_weight
        base_result['original_scale'] = guidance_scale
        base_result['amplified_scale'] = amplified_scale
        
        return base_result
    
    def apply_guidance(self, *args, **kwargs):
        if not self.enable_booster or self.booster is None:
            return self.base_integrator.apply_guidance(*args, **kwargs)
        
        # 否则，尝试使用增强逻辑
        # 这里需要根据参数判断调用哪个方法
        
        # 检查是否是乘性引导调用
        if 'theta_prime' in kwargs or (len(args) > 0 and isinstance(args[0], torch.Tensor)):
            # 尝试调用乘性引导
            try:
                return self.apply_multiplicative_guidance(*args, **kwargs)
            except:
                # 如果失败，回退到基础方法
                return self.base_integrator.apply_guidance(*args, **kwargs)
        else:
            # 其他情况，直接调用基础方法
            return self.base_integrator.apply_guidance(*args, **kwargs)
    
    def _estimate_current_properties(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        batch_ligand: torch.Tensor
    ) -> torch.Tensor:
        device = theta_t.device
        batch_size = batch_ligand.max().item() + 1
        
        properties = []
        
        for b in range(batch_size):
            mask = (batch_ligand == b)
            theta_b = theta_t[mask]  # [N_b, K]
            
            # 统计原子类型分布
            # 假设前8个维度是基本原子类型：H, C, N, O, F, P, S, Cl
            if theta_b.size(-1) >= 8:
                atom_probs = theta_b[:, :8].mean(dim=0)  # [8]
                
                # QED估算：O, N含量高 -> QED高
                qed_estimate = 0.5 + 0.3 * (atom_probs[3] + atom_probs[2])  # O + N
                qed_estimate = torch.clamp(qed_estimate, 0.3, 0.9)
                
                # SA估算：C含量高，复杂原子少 -> SA高
                sa_estimate = 0.5 + 0.3 * atom_probs[1] - 0.2 * (atom_probs[5] + atom_probs[6] + atom_probs[7])  # C - (P + S + Cl)
                sa_estimate = torch.clamp(sa_estimate, 0.2, 0.9)
            else:
                qed_estimate = torch.tensor(0.5, device=device)
                sa_estimate = torch.tensor(0.5, device=device)
            
            properties.append(torch.stack([qed_estimate, sa_estimate]))
        
        return torch.stack(properties)  # [B, 2]
    
    def get_stats(self) -> Dict:
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats = {
            'total_calls': 0,
            'boosted_calls': 0,
            'avg_amplification': 0.0,
            'max_amplification': 0.0
        }


def create_wrapped_guidance_integrator(
    guidance_model_path: str,
    device: str = 'cuda',
    enable_booster: bool = True,
    booster_config: Optional[Dict] = None,
    **base_integrator_kwargs
) -> GuidanceIntegrationWrapper:

    from core.models.geometric_guidance_integration import create_geometric_guidance_integrator
    
    base_integrator = create_geometric_guidance_integrator(
        guidance_model_path=guidance_model_path,
        device=device,
        **base_integrator_kwargs
    )
    
    if base_integrator is None:
        return None
    
    # 2. 包装增强器
    wrapped_integrator = GuidanceIntegrationWrapper(
        base_integrator=base_integrator,
        enable_booster=enable_booster,
        booster_config=booster_config
    )
    
    
    return wrapped_integrator

def create_booster_config(
    amplification_level: str = 'moderate',  # 'conservative', 'moderate', 'aggressive'
    enable_early_intervention: bool = True,
    enable_atom_preference: bool = True,
    enable_synergy: bool = True
) -> Dict:

    amplification_params = {
        'conservative': {'max_amplification': 3.0, 'gap_sensitivity': 2.0},
        'moderate': {'max_amplification': 5.0, 'gap_sensitivity': 3.0},
        'aggressive': {'max_amplification': 8.0, 'gap_sensitivity': 4.0}
    }
    
    params = amplification_params.get(amplification_level, amplification_params['moderate'])
    
    return {
        'enable_amplifier': True,
        'enable_early_intervention': enable_early_intervention,
        'enable_atom_preference': enable_atom_preference,
        'enable_synergy': enable_synergy,
        **params
    }

