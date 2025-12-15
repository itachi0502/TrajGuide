import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DirectLogitsGuidance:

    
    def __init__(self):

        self.atom_indices = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3,
            'F': 4, 'P': 5, 'S': 6, 'Cl': 7
        }
        

        self.qed_guidance_weights = torch.tensor([
            -5.0,   
            -3.0,   
            +10.0,  
            +10.0,  
            +3.0,   
            -5.0,   
            -5.0, 
            -3.0   
        ])
        

        self.sa_guidance_weights = torch.tensor([
            +5.0, 
            +10.0,  
            -5.0, 
            -5.0,   
            -8.0,   
            -10.0, 
            -10.0,  
            -8.0 
        ])
    
    def compute_guidance_logits(
        self,
        theta_t: torch.Tensor,              # [N, K] 
        current_properties: torch.Tensor,   # [B, 2] 
        target_properties: torch.Tensor,    # [B, 2] 
        batch_ligand: torch.Tensor,         # [N] 
        guidance_scale: float = 2.0,
        current_time: float = 0.5
    ) -> torch.Tensor:
        """
        计算引导logits
        
        Returns:
            guidance_logits: [N, K] 要添加到log(theta)的logits
        """
        device = theta_t.device
        N, K = theta_t.shape
        batch_size = torch.unique(batch_ligand).numel()
        
        # 确保K=8
        if K != 8:
            raise ValueError(f"Expected K=8, got K={K}")
        
        # 计算性质差距
        qed_gap = target_properties[:, 0] - current_properties[:, 0]  # [B]
        sa_gap = target_properties[:, 1] - current_properties[:, 1]    # [B]
        

        qed_gap = torch.relu(qed_gap)
        sa_gap = torch.relu(sa_gap)   
        
  

        qed_amplification = torch.exp(5.0 * qed_gap) - 1.0  # [B]
        sa_amplification = torch.exp(5.0 * sa_gap) - 1.0    # [B]
        

        if current_time < 0.3:
            time_factor = 2.0  # 早期：强引导
        elif current_time < 0.7:
            time_factor = 1.5  # 中期：中等引导
        else:
            time_factor = 1.0  # 后期：维持引导
        
        # 构造引导logits
        guidance_logits = torch.zeros(N, K, device=device)
        
        for b in range(batch_size):
            mask = (batch_ligand == b)
            
            # QED引导
            qed_contribution = (
                qed_gap[b].item() * 
                qed_amplification[b].item() * 
                self.qed_guidance_weights.to(device)
            )
            
            # SA引导
            sa_contribution = (
                sa_gap[b].item() * 
                sa_amplification[b].item() * 
                self.sa_guidance_weights.to(device)
            )
            

            total_guidance = qed_contribution + sa_contribution
            
            # 应用到该分子的所有原子
            guidance_logits[mask] = total_guidance.unsqueeze(0)
        
        # 应用guidance_scale和time_factor
        guidance_logits = guidance_logits * guidance_scale * time_factor
        
        return guidance_logits
    
    def apply_direct_guidance(
        self,
        theta_prime: torch.Tensor,          # [N, K] 未引导的信念状态
        current_properties: torch.Tensor,   # [B, 2] 当前预测的QED, SA
        target_properties: torch.Tensor,    # [B, 2] 目标QED, SA
        batch_ligand: torch.Tensor,         # [N] 批次索引
        guidance_scale: float = 2.0,
        current_time: float = 0.5
    ) -> Dict:
        """
        应用直接logits引导
        
        Returns:
            dict包含:
                - guided_theta: [N, K] 引导后的信念状态
                - guidance_logits: [N, K] 应用的引导logits
                - guidance_strength: float 实际引导强度
        """
        # 计算引导logits
        guidance_logits = self.compute_guidance_logits(
            theta_prime, current_properties, target_properties,
            batch_ligand, guidance_scale, current_time
        )
        
        # 🔥 关键：直接修改logits
        log_theta_prime = torch.log(theta_prime + 1e-10)
        log_guided = log_theta_prime + guidance_logits
        
        # 归一化
        guided_theta = F.softmax(log_guided, dim=-1)
        
        # 计算统计信息
        guidance_strength = torch.abs(guidance_logits).mean().item()
        max_guidance = torch.abs(guidance_logits).max().item()
        
        # 计算KL散度
        kl_div = torch.sum(
            guided_theta * (torch.log(guided_theta + 1e-10) - torch.log(theta_prime + 1e-10)),
            dim=-1
        ).mean().item()
        
        return {
            'guided_theta': guided_theta,
            'guidance_logits': guidance_logits,
            'guidance_strength': guidance_strength,
            'max_guidance': max_guidance,
            'kl_divergence': kl_div
        }


class HybridGuidanceIntegrator:

    
    def __init__(
        self,
        base_integrator,  
        use_direct_guidance: bool = True,
        direct_guidance_weight: float = 0.7  
    ):
        self.base_integrator = base_integrator
        self.use_direct_guidance = use_direct_guidance
        self.direct_guidance_weight = direct_guidance_weight
        
        if use_direct_guidance:
            self.direct_guidance = DirectLogitsGuidance()
        else:
            self.direct_guidance = None

    
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
        # 1. 调用基础几何引导
        base_result = self.base_integrator.apply_multiplicative_guidance(
            theta_prime, pos_t, t, batch_ligand, target_conditions,
            guidance_scale, alpha_h
        )
        
        if not self.use_direct_guidance or self.direct_guidance is None:
            return base_result
        
        # 2. 估算当前性质
        try:
            # 使用基础集成器的引导模型预测当前性质
            device = theta_prime.device
            batch_size = torch.unique(batch_ligand).numel()
            
            if t.numel() == 1:
                t_batch = t.expand(batch_size)
            elif t.size(0) == theta_prime.size(0):
                t_batch = torch.zeros(batch_size, device=device, dtype=t.dtype)
                for b in range(batch_size):
                    mask = (batch_ligand == b)
                    if mask.any():
                        t_batch[b] = t[mask][0]
            else:
                t_batch = t
            
            with torch.no_grad():
                pred_mu, pred_sigma = self.base_integrator.guidance_model(
                    theta_t=theta_prime,
                    pos_t=pos_t,
                    t=t_batch,
                    batch=batch_ligand
                )
            
            current_properties = pred_mu
        except:
            # 如果失败，使用默认值
            batch_size = torch.unique(batch_ligand).numel()
            current_properties = torch.tensor(
                [[0.5, 0.5]] * batch_size,
                device=theta_prime.device
            )
        
        # 3. 应用直接logits引导
        current_time = t[0].item() if t.numel() > 0 else 0.5
        
        direct_result = self.direct_guidance.apply_direct_guidance(
            theta_prime, current_properties, target_conditions,
            batch_ligand, guidance_scale, current_time
        )
        
        # 4. 混合两种引导结果
        # 使用加权平均（在概率空间）
        base_theta = base_result['guided_theta']
        direct_theta = direct_result['guided_theta']
        
        w_direct = self.direct_guidance_weight
        w_base = 1.0 - w_direct
        
        mixed_theta = w_base * base_theta + w_direct * direct_theta
        
        # 重新归一化
        mixed_theta = mixed_theta / (mixed_theta.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 5. 更新结果
        base_result['guided_theta'] = mixed_theta
        base_result['hybrid_guidance'] = True
        base_result['direct_guidance_strength'] = direct_result['guidance_strength']
        base_result['direct_guidance_kl'] = direct_result['kl_divergence']
        
        return base_result


def create_hybrid_guidance_integrator(
    guidance_model_path: str,
    device: str = 'cuda',
    use_direct_guidance: bool = True,
    direct_guidance_weight: float = 0.7,
    **kwargs
):
    # 1. 创建基础几何引导集成器
    from core.models.geometric_guidance_integration import create_geometric_guidance_integrator
    
    base_integrator = create_geometric_guidance_integrator(
        guidance_model_path=guidance_model_path,
        device=device,
        **kwargs
    )
    
    if base_integrator is None:
        return None
    
    # 2. 包装为混合引导集成器
    hybrid_integrator = HybridGuidanceIntegrator(
        base_integrator=base_integrator,
        use_direct_guidance=use_direct_guidance,
        direct_guidance_weight=direct_guidance_weight
    )
    
    
    return hybrid_integrator

