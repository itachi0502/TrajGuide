import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer


def sa_norm_from_rdkit(sa_raw: float) -> float:

    v = float(sa_raw)
    v = max(1.0, min(10.0, v))  # 裁剪到 [1, 10]
    return (10.0 - v) / 9.0


def compute_qed_sa_from_mol(mol) -> Optional[Tuple[float, float]]:

    if mol is None:
        return None
    
    try:
        qed_value = float(QED.qed(mol))
        

        sa_raw = float(sascorer.calculateScore(mol))
        sa_normalized = sa_norm_from_rdkit(sa_raw)
        
        return qed_value, sa_normalized
    except Exception as e:
        return None


class TerminalFilteringResampler:

    
    def __init__(
        self,
        target_qed: float = 0.56,
        target_sa: float = 0.78,
        qed_threshold: float = 0.50,  # QED最低阈值
        sa_threshold: float = 0.70,   # SA最低阈值
        max_resample_attempts: int = 3,  # 最大重采样次数
        perturbation_strength: float = 0.1,  # 扰动强度
        resample_last_steps: int = 20,  # 重采样最后N步
        acceptance_tolerance: float = 0.05,  # 接受容差
        verbose: bool = True,
    ):

        self.target_qed = target_qed
        self.target_sa = target_sa
        self.qed_threshold = qed_threshold
        self.sa_threshold = sa_threshold
        self.max_resample_attempts = max_resample_attempts
        self.perturbation_strength = perturbation_strength
        self.resample_last_steps = resample_last_steps
        self.acceptance_tolerance = acceptance_tolerance
        self.verbose = verbose
        
        # 统计信息
        self.stats = {
            'total_molecules': 0,
            'accepted_first_try': 0,
            'accepted_after_resample': 0,
            'rejected': 0,
            'qed_improvements': [],
            'sa_improvements': [],
        }
    
    def compute_acceptance_score(self, qed: float, sa: float) -> float:
        qed_diff = abs(qed - self.target_qed)
        sa_diff = abs(sa - self.target_sa)
        score = 1.0 - (qed_diff + sa_diff) / 2.0
        return max(0.0, score)
    
    def should_accept(self, qed: float, sa: float) -> bool:
        if qed < self.qed_threshold or sa < self.sa_threshold:
            return False
        
  
        qed_close = abs(qed - self.target_qed) <= self.acceptance_tolerance
        sa_close = abs(sa - self.target_sa) <= self.acceptance_tolerance
        

        if qed_close or sa_close:
            return True
        

        qed_reasonable = abs(qed - self.target_qed) <= 2 * self.acceptance_tolerance
        sa_reasonable = abs(sa - self.target_sa) <= 2 * self.acceptance_tolerance
        
        return qed_reasonable and sa_reasonable
    
    def perturb_theta(self, theta: torch.Tensor, strength: float = None) -> torch.Tensor:
        if strength is None:
            strength = self.perturbation_strength
        
        # 在log空间扰动
        eps = 1e-10
        log_theta = torch.log(theta + eps)
        noise = torch.randn_like(log_theta) * strength
        log_theta_perturbed = log_theta + noise
        
        # 重新归一化
        theta_perturbed = F.softmax(log_theta_perturbed, dim=-1)
        
        return theta_perturbed
    
    def filter_and_resample(
        self,
        molist: List[Optional[Chem.Mol]],
        sample_chain: List[Tuple],
        theta_chain: List[Tuple],
        dynamics_model,
        batch_ligand: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        **sampling_kwargs
    ) -> Tuple[List[Optional[Chem.Mol]], Dict]:

        self.stats['total_molecules'] = len(molist)
        filtered_molist = []
        
        for mol_idx, mol in enumerate(molist):
            qed_sa = compute_qed_sa_from_mol(mol)
            
            if qed_sa is None:
                self.stats['rejected'] += 1
                filtered_molist.append(None)
                continue
            
            qed, sa = qed_sa
            

            if self.should_accept(qed, sa):
                self.stats['accepted_first_try'] += 1
                filtered_molist.append(mol)
                continue


            
            best_mol = mol
            best_qed, best_sa = qed, sa
            best_score = self.compute_acceptance_score(qed, sa)
            

            if self.should_accept(best_qed, best_sa):
                self.stats['accepted_after_resample'] += 1
                self.stats['qed_improvements'].append(best_qed - qed)
                self.stats['sa_improvements'].append(best_sa - sa)
                filtered_molist.append(best_mol)
            else:
                self.stats['rejected'] += 1
                filtered_molist.append(None)
        
        return filtered_molist, self.get_stats()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_molecules']
        if total == 0:
            return self.stats
        
        stats_summary = {
            **self.stats,
            'acceptance_rate': (self.stats['accepted_first_try'] + self.stats['accepted_after_resample']) / total,
            'first_try_rate': self.stats['accepted_first_try'] / total,
            'resample_success_rate': self.stats['accepted_after_resample'] / max(1, total - self.stats['accepted_first_try']),
            'rejection_rate': self.stats['rejected'] / total,
        }
        
        if self.stats['qed_improvements']:
            stats_summary['avg_qed_improvement'] = np.mean(self.stats['qed_improvements'])
            stats_summary['avg_sa_improvement'] = np.mean(self.stats['sa_improvements'])
        
        return stats_summary
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_molecules': 0,
            'accepted_first_try': 0,
            'accepted_after_resample': 0,
            'rejected': 0,
            'qed_improvements': [],
            'sa_improvements': [],
        }

