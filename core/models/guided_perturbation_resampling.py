import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer


def sa_norm_from_rdkit(sa_raw: float) -> float:

    v = float(sa_raw)
    v = max(1.0, min(10.0, v))
    return (10.0 - v) / 9.0


class GuidedPerturbation:

    
    def __init__(
        self,
        guidance_model,
        perturbation_strength: float = 0.15,
        guided_ratio: float = 0.7,
        random_ratio: float = 0.3,
        adaptive_strength: bool = True,
        verbose: bool = True,
    ):

        self.guidance_model = guidance_model
        self.perturbation_strength = perturbation_strength
        self.guided_ratio = guided_ratio
        self.random_ratio = random_ratio
        self.adaptive_strength = adaptive_strength
        self.verbose = verbose
        
        # 归一化权重
        total_ratio = guided_ratio + random_ratio
        self.guided_ratio = guided_ratio / total_ratio
        self.random_ratio = random_ratio / total_ratio
    
    def compute_guided_direction(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_qed: float,
        target_sa: float,
        t: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:

        device = theta_t.device
        batch_size = batch_ligand.max().item() + 1
        
        # 准备时间张量
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        
        # 准备目标条件
        target_conditions = torch.tensor(
            [[target_qed, target_sa]] * batch_size,
            device=device,
            dtype=torch.float32
        )  # [B, 2]
        
   
        with torch.no_grad():
            pred_mu, pred_sigma = self.guidance_model(
                theta_t=theta_t,
                pos_t=pos_t,
                t=t_tensor,
                batch=batch_ligand,
            )

        

        delta_qed = target_conditions[:, 0] - pred_mu[:, 0]  # [B]
        delta_sa = target_conditions[:, 1] - pred_mu[:, 1]   # [B]
       
        eps = 1e-6
        grad_qed = delta_qed / (pred_sigma[:, 0] ** 2 + eps)  # [B]
        grad_sa = delta_sa / (pred_sigma[:, 1] ** 2 + eps)    # [B]
        
        # 合并QED和SA的梯度
        delta_log_prob = grad_qed + grad_sa  # [B]
        
        # 分配到每个原子
        delta_log_prob_per_atom = delta_log_prob[batch_ligand]  # [N]
        
        # 扩展到原子类型维度
        guided_direction = delta_log_prob_per_atom.unsqueeze(-1)  # [N, 1]
        
        # 诊断信息
        info = {
            'pred_qed': pred_mu[:, 0].mean().item(),
            'pred_sa': pred_mu[:, 1].mean().item(),
            'delta_qed': delta_qed.mean().item(),
            'delta_sa': delta_sa.mean().item(),
            'guidance_strength': delta_log_prob.abs().mean().item(),
        }
        
        return guided_direction, info
    
    def perturb_theta(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_qed: float,
        target_sa: float,
        current_qed: Optional[float] = None,
        current_sa: Optional[float] = None,
        t: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:

        # 1. 计算引导方向
        guided_direction, guidance_info = self.compute_guided_direction(
            theta_t=theta_t,
            pos_t=pos_t,
            batch_ligand=batch_ligand,
            target_qed=target_qed,
            target_sa=target_sa,
            t=t,
        )
        
        # 2. 自适应调整扰动强度
        strength = self.perturbation_strength
        if self.adaptive_strength and current_qed is not None and current_sa is not None:
            # 根据偏差大小调整强度
            qed_deviation = abs(current_qed - target_qed)
            sa_deviation = abs(current_sa - target_sa)
            total_deviation = qed_deviation + sa_deviation
            
            # 偏差越大，扰动强度越大（但有上限）
            strength = self.perturbation_strength * (1.0 + total_deviation)
            strength = min(strength, self.perturbation_strength * 3.0)  # 最多3倍
            
            if self.verbose:
                print(f"   🎯 自适应扰动强度: {strength:.4f} (基础={self.perturbation_strength:.4f}, 偏差={total_deviation:.4f})")
        
        # 3. 在log空间进行扰动
        eps = 1e-10
        log_theta = torch.log(theta_t + eps)
        
        # 3.1 引导性扰动
        guided_perturbation = strength * self.guided_ratio * guided_direction
        
        # 3.2 随机噪声（探索性）
        random_noise = torch.randn_like(log_theta) * (strength * self.random_ratio)
        
        # 3.3 混合
        log_theta_perturbed = log_theta + guided_perturbation + random_noise
        
        # 4. 归一化
        theta_perturbed = F.softmax(log_theta_perturbed, dim=-1)
        
        # 5. 诊断信息
        info = {
            **guidance_info,
            'perturbation_strength': strength,
            'guided_ratio': self.guided_ratio,
            'random_ratio': self.random_ratio,
            'theta_change': (theta_perturbed - theta_t).abs().mean().item(),
        }
        

        
        return theta_perturbed, info


class PartialResampler:

    
    def __init__(
        self,
        resample_last_steps: int = 20,
        max_attempts: int = 3,
        verbose: bool = True,
    ):

        self.resample_last_steps = resample_last_steps
        self.max_attempts = max_attempts
        self.verbose = verbose
    
    def resample(
        self,
        dynamics_model,
        theta_chain: List[torch.Tensor],
        pos_chain: List[torch.Tensor],
        perturbation_calculator: GuidedPerturbation,
        batch_ligand: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        target_qed: float,
        target_sa: float,
        current_qed: float,
        current_sa: float,
        guidance_scale: float = 2.0,
        **sampling_kwargs
    ) -> Tuple[Optional[Chem.Mol], Dict]:

        return None, {'status': 'not_implemented'}

