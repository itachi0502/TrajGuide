import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class AdaptiveGuidanceAmplifier:

    
    def __init__(
        self,
        base_scale: float = 2.0,
        max_amplification: float = 5.0,
        gap_sensitivity: float = 3.0
    ):
        self.base_scale = base_scale
        self.max_amplification = max_amplification
        self.gap_sensitivity = gap_sensitivity
    
    def compute_amplified_scale(
        self,
        current_properties: torch.Tensor,  # [B, 2] 当前预测的QED, SA
        target_properties: torch.Tensor,   # [B, 2] 目标QED, SA
        current_time: float,               # [0, 1] 当前时间步
        original_scale: float              # 原始guidance_scale
    ) -> float:

        gaps = torch.abs(target_properties - current_properties)  # [B, 2]
        qed_gap = gaps[:, 0].mean().item()
        sa_gap = gaps[:, 1].mean().item()

        gap_factor = 1.0 + self.gap_sensitivity * max(qed_gap, sa_gap)
        

        if current_time < 0.3:
            time_factor = 1.0
        elif current_time < 0.7:
            time_factor = 1.0 + 2.5 * (current_time - 0.3) / 0.4
        else:
            time_factor = 2.5 + 1.5 * (current_time - 0.7) / 0.3
        
        # 计算最终放大后的强度
        amplified_scale = original_scale * gap_factor * time_factor
        
        # 限制最大值，避免过度引导破坏结构
        amplified_scale = min(amplified_scale, self.max_amplification * original_scale)
        
        return amplified_scale


class EarlyInterventionScheduler:

    def __init__(
        self,
        early_start_ratio: float = 0.5,  # 早期引导强度比例
        mid_boost_ratio: float = 1.2,    # 中期引导增强比例
        late_boost_ratio: float = 1.5    # 后期引导增强比例
    ):
        self.early_start_ratio = early_start_ratio
        self.mid_boost_ratio = mid_boost_ratio
        self.late_boost_ratio = late_boost_ratio
    
    def get_time_weight(self, current_time: float) -> float:

        if current_time < 0.3:
            return self.early_start_ratio
        elif current_time < 0.7:
            progress = (current_time - 0.3) / 0.4
            return self.early_start_ratio + (self.mid_boost_ratio - self.early_start_ratio) * progress
        else:
            progress = (current_time - 0.7) / 0.3
            return self.mid_boost_ratio + (self.late_boost_ratio - self.mid_boost_ratio) * progress


class AtomTypePreferenceInjector:

    
    def __init__(self):
        self.atom_indices = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3,
            'F': 4, 'P': 5, 'S': 6, 'Cl': 7
        }
        
        # QED偏好：O, N有利于药物性
        self.qed_preference = torch.zeros(8)
        self.qed_preference[self.atom_indices['O']] = 0.3   # 氧原子增强
        self.qed_preference[self.atom_indices['N']] = 0.25  # 氮原子增强
        self.qed_preference[self.atom_indices['C']] = 0.1   # 碳原子轻微增强
        self.qed_preference[self.atom_indices['F']] = -0.1  # 氟原子轻微抑制
        self.qed_preference[self.atom_indices['P']] = -0.2  # 磷原子抑制
        self.qed_preference[self.atom_indices['S']] = -0.15 # 硫原子抑制
        self.qed_preference[self.atom_indices['Cl']] = -0.2 # 氯原子抑制
        
        # SA偏好：C, H有利于合成，复杂原子不利
        self.sa_preference = torch.zeros(8)
        self.sa_preference[self.atom_indices['C']] = 0.35   # 碳原子强增强
        self.sa_preference[self.atom_indices['H']] = 0.2    # 氢原子增强
        self.sa_preference[self.atom_indices['N']] = 0.1    # 氮原子轻微增强
        self.sa_preference[self.atom_indices['O']] = 0.15   # 氧原子增强
        self.sa_preference[self.atom_indices['P']] = -0.4   # 磷原子强抑制
        self.sa_preference[self.atom_indices['S']] = -0.3   # 硫原子抑制
        self.sa_preference[self.atom_indices['Cl']] = -0.35 # 氯原子强抑制
        self.sa_preference[self.atom_indices['F']] = -0.2   # 氟原子抑制
    
    def inject_preference(
        self,
        theta_t: torch.Tensor,           # [N, K] 当前原子类型分布
        target_qed: float,               # 目标QED
        target_sa: float,                # 目标SA
        current_qed: float,              # 当前QED
        current_sa: float,               # 当前SA
        injection_strength: float = 1.0  # 注入强度
    ) -> torch.Tensor:

        device = theta_t.device
        N, K = theta_t.shape
        
        if K != 8:
            K_effective = min(K, 8)
        else:
            K_effective = K
        
        qed_gap = target_qed - current_qed
        sa_gap = target_sa - current_sa
        
        preference = torch.zeros(K, device=device)
        
        if qed_gap > 0:
            preference[:K_effective] += qed_gap * self.qed_preference[:K_effective].to(device)
        
        if sa_gap > 0:
            preference[:K_effective] += sa_gap * self.sa_preference[:K_effective].to(device)
        
        preference = preference * injection_strength
        
        preference_logits = preference.unsqueeze(0).expand(N, -1)  # [N, K]
        
        return preference_logits


class MultiObjectiveSynergyOptimizer:

    
    def __init__(
        self,
        qed_weight: float = 1.0,
        sa_weight: float = 1.0,
        synergy_bonus: float = 0.3  # 协同奖励
    ):
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        self.synergy_bonus = synergy_bonus
    
    def compute_balanced_guidance(
        self,
        qed_guidance: torch.Tensor,  # [N, K] QED引导方向
        sa_guidance: torch.Tensor,   # [N, K] SA引导方向
        qed_gap: float,              # QED差距
        sa_gap: float                # SA差距
    ) -> torch.Tensor:

        if qed_gap > 0 and sa_gap > 0:

            qed_w = self.qed_weight * (1.0 + self.synergy_bonus)
            sa_w = self.sa_weight * (1.0 + self.synergy_bonus)
        else:
            total_gap = abs(qed_gap) + abs(sa_gap) + 1e-6
            qed_w = self.qed_weight * abs(qed_gap) / total_gap
            sa_w = self.sa_weight * abs(sa_gap) / total_gap
        

        balanced_guidance = qed_w * qed_guidance + sa_w * sa_guidance
        
        return balanced_guidance


class EnhancedGuidanceBooster:


    def __init__(
        self,
        base_guidance_scale: float = 2.0,
        enable_amplifier: bool = True,
        enable_early_intervention: bool = True,
        enable_atom_preference: bool = True,
        enable_synergy: bool = True,
        max_amplification: float = 5.0,
        gap_sensitivity: float = 3.0
    ):
        self.base_guidance_scale = base_guidance_scale

        # 初始化各个模块
        self.amplifier = AdaptiveGuidanceAmplifier(
            base_scale=base_guidance_scale,
            max_amplification=max_amplification,
            gap_sensitivity=gap_sensitivity
        ) if enable_amplifier else None
        self.early_scheduler = EarlyInterventionScheduler() if enable_early_intervention else None
        self.atom_injector = AtomTypePreferenceInjector() if enable_atom_preference else None
        self.synergy_optimizer = MultiObjectiveSynergyOptimizer() if enable_synergy else None
        

    
    def boost_guidance(
        self,
        theta_t: torch.Tensor,              # [N, K] 当前原子类型分布
        current_properties: torch.Tensor,   # [B, 2] 当前QED, SA
        target_properties: torch.Tensor,    # [B, 2] 目标QED, SA
        current_time: float,                # [0, 1] 当前时间步
        original_guidance_scale: float,     # 原始引导强度
        original_guidance_logits: Optional[torch.Tensor] = None  # [N, K] 原始引导logits
    ) -> Dict[str, torch.Tensor]:

        result = {}

        if self.amplifier is not None:
            amplified_scale = self.amplifier.compute_amplified_scale(
                current_properties, target_properties, current_time, original_guidance_scale
            )
            result['amplified_scale'] = amplified_scale
        else:
            amplified_scale = original_guidance_scale
            result['amplified_scale'] = amplified_scale
        
        if self.early_scheduler is not None:
            time_weight = self.early_scheduler.get_time_weight(current_time)
            result['time_weight'] = time_weight
        else:
            time_weight = 1.0
            result['time_weight'] = time_weight
        
        preference_logits = None
        if self.atom_injector is not None and theta_t is not None:
            current_qed = current_properties[0, 0].item()
            current_sa = current_properties[0, 1].item()
            target_qed = target_properties[0, 0].item()
            target_sa = target_properties[0, 1].item()
            
            preference_logits = self.atom_injector.inject_preference(
                theta_t, target_qed, target_sa, current_qed, current_sa,
                injection_strength=amplified_scale * time_weight
            )
            result['preference_logits'] = preference_logits
        
        if original_guidance_logits is not None:
            final_logits = original_guidance_logits * amplified_scale * time_weight
            if preference_logits is not None:
                final_logits = final_logits + preference_logits
            result['final_guidance_logits'] = final_logits
        elif preference_logits is not None:
            result['final_guidance_logits'] = preference_logits
        
        return result


def create_enhanced_guidance_booster(
    base_guidance_scale: float = 2.0,
    **kwargs
) -> EnhancedGuidanceBooster:
    return EnhancedGuidanceBooster(base_guidance_scale=base_guidance_scale, **kwargs)

