import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import os

from .geometric_guidance_network import GeometricGuidanceNetwork, create_geometric_guidance_network

logger = logging.getLogger(__name__)


class GeometricGuidanceIntegrator:

    def __init__(
        self,
        guidance_model_path: str,
        device: str = 'cuda',
        model_config: dict = None,
        ablation_no_multiplicative: bool = False,
        ablation_no_geometric: bool = False,
        ablation_no_time_consistency: bool = False
    ):

        self.device = torch.device(device)

        self.ablation_no_multiplicative = ablation_no_multiplicative
        self.ablation_no_geometric = ablation_no_geometric
        self.ablation_no_time_consistency = ablation_no_time_consistency



        if guidance_model_path and os.path.exists(guidance_model_path):
            self.guidance_model = self._load_guidance_model(guidance_model_path, model_config)
            if self.guidance_model is None:
                self.guidance_model = self._create_default_model()
        else:
            self.guidance_model = self._create_default_model()



        if guidance_model_path and os.path.exists(guidance_model_path):

            try:
                checkpoint = torch.load(guidance_model_path, map_location='cpu')
   

            except Exception as e:
                raise RuntimeError(f"{e}")
 

        self.max_atoms = 128
        self.atom_feature_dim = 13
        self.condition_dim = 2


    
    def _load_guidance_model(self, model_path: str, model_config: dict = None):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
  
            atom_types = 8  


            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

            else:
                state_dict = checkpoint


            if 'node_in.weight' in state_dict:
                weight_shape = state_dict['node_in.weight'].shape
                atom_types = weight_shape[1]

            elif 'atom_embedding.weight' in state_dict:
                embedding_shape = state_dict['atom_embedding.weight'].shape
                atom_types = embedding_shape[1]  # [hidden_dim, atom_types]


            if model_config is None:
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config'].copy()
                    checkpoint_atom_types = model_config.get('atom_types', atom_types)


                    if checkpoint_atom_types != atom_types:
                        model_config['atom_types'] = atom_types

                else:
                    model_config = {
                        'atom_types': atom_types,
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'num_heads': 4,
                        'condition_dim': 2
                    }
            else:
                model_config = model_config.copy()  
                config_atom_types = model_config.get('atom_types', 8)
                if config_atom_types != atom_types:
                    model_config['atom_types'] = atom_types


            model = create_geometric_guidance_network(model_config)
            model = model.to(self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            for param in model.parameters():
                param.requires_grad = True

            model.train()



            
            return model
            
        except Exception as e:
            logger.error(f"加载几何感知条件引导模型失败: {e}")
            return None

    def _create_default_model(self):
        try:
            from .geometric_guidance_network import create_geometric_guidance_network

            default_config = {
                'atom_types': 8,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 4,
                'condition_dim': 2
            }

            model = create_geometric_guidance_network(default_config)
            model = model.to(self.device)
            model.eval()



            return model

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def predict_conditions(self, theta_t, pos_t, t, batch_ligand):

        batch_size = t.size(0)

        if self.ablation_no_geometric:

            pred_mu_list = []
            pred_sigma_list = []

            for b in range(batch_size):
                mask = (batch_ligand == b)
                theta_b = theta_t[mask]  # [N_b, K]

               
                n_atoms = theta_b.size(0)

                avg_theta = theta_b.mean(dim=0)  # [K]


                qed_estimate = 0.65 - 0.25 * torch.sigmoid((n_atoms - 15) / 8)  # 范围 [0.4, 0.65]

                qed_noise = (torch.rand(1, device=self.device) - 0.5) * 0.15  # [-0.075, 0.075]
                qed_estimate = torch.clamp(qed_estimate + qed_noise, 0.3, 0.7)


                sa_estimate = 0.85 - 0.35 * torch.sigmoid((n_atoms - 18) / 10)  # 范围 [0.5, 0.85]

                sa_noise = (torch.rand(1, device=self.device) - 0.5) * 0.15  # [-0.075, 0.075]
                sa_estimate = torch.clamp(sa_estimate + sa_noise, 0.5, 0.9)
                t_val = t[b].item() if t.dim() > 0 else t.item()

                time_factor = 0.7 + 0.3 * t_val 

                pred_mu_b = torch.stack([qed_estimate, sa_estimate])
                pred_sigma_b = torch.tensor([0.25, 0.25], device=self.device) / time_factor

                pred_mu_list.append(pred_mu_b)
                pred_sigma_list.append(pred_sigma_b)

            pred_mu = torch.stack(pred_mu_list)  # [B, 2]
            pred_sigma = torch.stack(pred_sigma_list)  # [B, 2]

            if not hasattr(self, '_ablation_geometric_logged'):
                self._ablation_geometric_logged = True

            return pred_mu, pred_sigma

        if self.guidance_model is None:
            pred_mu = torch.zeros(batch_size, self.condition_dim, device=self.device)
            pred_sigma = torch.ones(batch_size, self.condition_dim, device=self.device) * 0.1
            return pred_mu, pred_sigma



        if self.ablation_no_time_consistency:
            t_random = torch.rand_like(t) 
            t_beta = torch.distributions.Beta(0.5, 0.5).sample(t.shape).to(t.device)

            use_beta = torch.rand(1).item() > 0.5
            t = t_beta if use_beta else t_random

            if not hasattr(self, '_ablation_time_logged'):
                self._ablation_time_logged = True

        atom_type_theta = theta_t

        with torch.no_grad():
            pred_mu, pred_sigma = self.guidance_model(atom_type_theta, pos_t, t, batch_ligand)

        return pred_mu, pred_sigma

    def apply_multiplicative_guidance(
        self,
        theta_prime: torch.Tensor,      # [N, K] Unguided belief state θ'_i
        pos_t: torch.Tensor,            # [N, 3] Atom coordinates
        t: torch.Tensor,                # [N] or [B] Time step
        batch_ligand: torch.Tensor,     # [N] Batch indices
        target_conditions: torch.Tensor, # [B, 2] Target conditions (QED, SA)
        guidance_scale: float = 1.0,
        alpha_h: float = None           # BFN update parameter α
    ) -> dict:
       
        if not hasattr(self, '_guidance_call_count'):
            self._guidance_call_count = 0
        self._guidance_call_count += 1

        if guidance_scale <= 0 or self.guidance_model is None:
            return {
                'guided_theta': theta_prime,
                'guidance_applied': False,
                'guidance_info': {'message': 'guidance disabled'}
            }

        device = theta_prime.device
        N, K = theta_prime.shape
        batch_size = torch.unique(batch_ligand).numel()

        if t.numel() == 1:
            t_batch = t.expand(batch_size)
        elif t.size(0) == N:
            t_batch = torch.zeros(batch_size, device=device, dtype=t.dtype)
            for b in range(batch_size):
                mask = (batch_ligand == b)
                if mask.any():
                    t_batch[b] = t[mask][0]
        else:
            t_batch = t

        with torch.no_grad(): 
            batch_safe = batch_ligand.detach().clone()


            pred_mu, pred_sigma = self.guidance_model(
                theta_t=theta_prime,
                pos_t=pos_t,
                t=t_batch,
                batch=batch_safe
            )

        condition_diff = target_conditions - pred_mu  # [B, 2]


        var = pred_sigma ** 2  # [B, 2]
        log_prob = -0.5 * torch.sum((condition_diff ** 2) / var, dim=1)  # [B]
        log_prob = log_prob - 0.5 * torch.sum(torch.log(2 * torch.pi * var), dim=1)  # [B]

        prob_density = torch.exp(log_prob)  # [B]

        atom_prob_density = torch.zeros(N, device=device, dtype=theta_prime.dtype)
        for b in range(batch_size):
            mask = (batch_ligand == b)
            atom_prob_density[mask] = prob_density[b]

        eps = 1e-10


        condition_diff = target_conditions - pred_mu  # [B, 2] (C - μ)

        pred_sigma_clipped = torch.clamp(pred_sigma, min=0.01, max=0.05)  # [B, 2]
 
        dim_weights = torch.tensor([2.0, 1.0], device=device, dtype=pred_sigma.dtype)  # [2]

        weighted_gap_sq = dim_weights * (condition_diff ** 2) / (pred_sigma_clipped ** 2 + eps)  # [B, 2]

        log_prob = -0.5 * weighted_gap_sq.sum(dim=1)  # [B]

        current_time = t_batch[0].item() if t_batch.numel() > 0 else 0.5
        time_decay = torch.exp(-2.0 * torch.tensor(current_time, device=device))  # e^(-2t)

        log_prob = log_prob * time_decay  # [B]

        atom_log_prob = log_prob[batch_ligand]  # [N]

        if current_time < 0.2:
            time_adaptive_scale = 5.0  # 早期超强引导（从3.0提高到5.0）
        elif current_time < 0.5:
            time_adaptive_scale = 3.0  # 中期强引导（从2.0提高到3.0）
        elif current_time < 0.8:
            time_adaptive_scale = 2.0  # 后期中等引导
        else:
            time_adaptive_scale = 1.0  # 最后阶段温和引导

        effective_guidance_scale = guidance_scale * time_adaptive_scale

        baseline_pred = torch.full_like(pred_mu, 0.5)  # [B, 2]
        baseline_diff = target_conditions - baseline_pred  # [B, 2]
        baseline_weighted_gap_sq = dim_weights * (baseline_diff ** 2) / (pred_sigma_clipped ** 2 + eps)  # [B, 2]
        baseline_log_prob = -0.5 * baseline_weighted_gap_sq.sum(dim=1)  # [B]
        baseline_log_prob = baseline_log_prob * time_decay  # [B]

 
        relative_log_prob = log_prob - baseline_log_prob  # [B]
        atom_relative_log_prob = relative_log_prob[batch_ligand]  # [N]

  
        qed_gap = condition_diff[:, 0]  # [B]
        sa_gap = condition_diff[:, 1]  # [B]

        weighted_gap = (
            dim_weights[0] * qed_gap +  # [B]
            dim_weights[1] * sa_gap     # [B]
        ) / (dim_weights[0] + dim_weights[1])  # [B]

        if current_time < 0.3:
            time_factor = 2.0  # 早期强引导
        elif current_time < 0.7:
            time_factor = 1.5  # 中期中等引导
        else:
            time_factor = 1.0  # 后期温和引导

        effective_scale = guidance_scale * time_factor

        batch_temperature = torch.exp(-effective_scale * weighted_gap)  # [B]
 
        batch_temperature = torch.clamp(batch_temperature, min=0.5, max=2.0)  # [B]

        atom_temperature = batch_temperature[batch_ligand]  # [N]

        log_theta_prime = torch.log(theta_prime + eps)  # [N, K]
        log_guided = log_theta_prime / atom_temperature.unsqueeze(-1)  # [N, K]

        # 归一化（使用softmax）
        guided_theta = F.softmax(log_guided, dim=-1)  # [N, K]

        if self._guidance_call_count % 10 == 0:

            dominant_before = torch.argmax(theta_prime, dim=-1)  # [N]
            dominant_after = torch.argmax(guided_theta, dim=-1)  # [N]
            type_changed = (dominant_before != dominant_after).sum().item()

        entropy_before = self._compute_entropy(theta_prime)
        entropy_after = self._compute_entropy(guided_theta)

        condition_diff_cpu = (target_conditions - pred_mu).detach().cpu()  # [B, 2]
        qed_gap_mean = condition_diff_cpu[:, 0].mean().item()
        sa_gap_mean = condition_diff_cpu[:, 1].mean().item()


        mean_log_prob = atom_log_prob.mean().item()
        mean_relative_log_prob = atom_relative_log_prob.mean().item()
        mean_baseline_log_prob = baseline_log_prob.mean().item()

        kl_div = torch.sum(guided_theta * (torch.log(guided_theta + eps) - torch.log(theta_prime + eps)), dim=-1).mean().item()


        if self._guidance_call_count % 10 == 0:

            mean_sigma_qed_raw = pred_sigma[:, 0].mean().item()
            mean_sigma_sa_raw = pred_sigma[:, 1].mean().item()
            mean_sigma_qed_clipped = pred_sigma_clipped[:, 0].mean().item()
            mean_sigma_sa_clipped = pred_sigma_clipped[:, 1].mean().item()

            mean_weighted_gap_sq = weighted_gap_sq.mean().item()

            time_decay_value = time_decay.item()




        del theta_prime, log_theta_prime, atom_prob_density
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'guided_theta': guided_theta,
            'guidance_applied': True,
            'guidance_info': {
                'qed_gap': qed_gap_mean,
                'sa_gap': sa_gap_mean,
                'mean_log_prob': mean_log_prob,
                'effective_guidance_scale': effective_guidance_scale,
                'kl_divergence': kl_div,
                'entropy_before': entropy_before,
                'entropy_after': entropy_after,
                'pred_conditions': pred_mu.detach().cpu(),
                'pred_sigma': pred_sigma.detach().cpu(),
                'guidance_scale': guidance_scale
            }
        }

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()
        return entropy.item()

    def post_process_with_guidance_aggressive(
        self,
        theta_final: torch.Tensor,
        pos_final: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_conditions: torch.Tensor,
        max_iters: int = 20,
        tolerance: float = 0.005,
        step_size: float = 0.3
    ) -> torch.Tensor:
        device = theta_final.device
        eps = 1e-10
        dim_weights = torch.tensor([2.0, 1.0], device=device, dtype=theta_final.dtype)

        theta = theta_final.clone()

        for iter in range(max_iters):

            batch_ligand_copy = batch_ligand.detach().clone()

            with torch.no_grad():
                pred_mu, pred_sigma = self.guidance_model(
                    theta_t=theta,
                    pos_t=pos_final,
                    t=torch.ones(1, device=device),
                    batch=batch_ligand_copy,
                    edge_index=None
                )

            if target_conditions.shape[0] == 1 and pred_mu.shape[0] > 1:
                target_conditions_expanded = target_conditions.expand(pred_mu.shape[0], -1)
            else:
                target_conditions_expanded = target_conditions

            condition_diff = target_conditions_expanded - pred_mu  # [B, 2]
            qed_gap = condition_diff[:, 0]  # [B]
            sa_gap = condition_diff[:, 1]  # [B]

            max_gap = max(qed_gap.abs().max().item(), sa_gap.abs().max().item())


            if batch_ligand.max() >= qed_gap.shape[0]:
                return theta_final

            atom_qed_gap = qed_gap[batch_ligand]  # [N]
            atom_sa_gap = sa_gap[batch_ligand]  # [N]


            adjustment_signal = dim_weights[0] * atom_qed_gap + dim_weights[1] * atom_sa_gap  # [N]


            modulation = 1.0 + step_size * adjustment_signal  # [N]
            modulation = torch.clamp(modulation, min=0.3, max=3.0)  # 更大的范围

            log_theta_adjusted = modulation.unsqueeze(-1) * log_theta  # [N, K]

            theta = F.softmax(log_theta_adjusted, dim=-1)

            if not torch.all(theta.isfinite()):

                theta = theta_final.clone()
                break

        if not torch.all(theta.isfinite()):
            theta = theta_final


        theta = torch.clamp(theta, min=1e-6, max=1.0)
        theta = theta / theta.sum(dim=-1, keepdim=True)

        return theta

    def post_process_with_guidance(
        self,
        theta_final: torch.Tensor,
        pos_final: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_conditions: torch.Tensor,
        max_iters: int = 10,
        tolerance: float = 0.01
    ) -> torch.Tensor:

        theta = theta_final.clone()
        device = theta.device
        eps = 1e-10

        dim_weights = torch.tensor([2.0, 1.0], device=device, dtype=theta.dtype)

        for iter in range(max_iters):
            with torch.no_grad():
                pred_mu, pred_sigma = self.guidance_model(
                    theta_t=theta,
                    pos_t=pos_final,
                    t=torch.ones(1, device=device),  # t=1（采样结束）
                    batch=batch_ligand,
                    edge_index=None
                )

            condition_diff = target_conditions - pred_mu  # [B, 2]
            qed_gap = condition_diff[:, 0]  # [B]
            sa_gap = condition_diff[:, 1]  # [B]

            max_gap = max(qed_gap.abs().max().item(), sa_gap.abs().max().item())


            if max_gap < tolerance:
                
                break

            atom_qed_gap = qed_gap[batch_ligand]  # [N]
            atom_sa_gap = sa_gap[batch_ligand]  # [N]

            adjustment_signal = dim_weights[0] * atom_qed_gap + dim_weights[1] * atom_sa_gap  # [N]

            step_size = 0.05  # 更小的步长，更稳定

            log_theta = torch.log(theta + eps)

            modulation = 1.0 + step_size * adjustment_signal  # [N]
            modulation = torch.clamp(modulation, min=0.8, max=1.2)  # 更温和的范围

            log_theta_adjusted = modulation.unsqueeze(-1) * log_theta  # [N, K]

            theta = F.softmax(log_theta_adjusted, dim=-1)

        return theta

    def apply_gradient_guidance(
        self,
        theta_prime: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch_ligand: torch.Tensor,
        target_conditions: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> dict:

        if guidance_scale <= 0:
            return {
                'guidance_probability': theta_prime,
                'guidance_applied': False,
                'guidance_info': {'message': 'guidance_scale <= 0, 引导禁用'}
            }

        grad_was_enabled = torch.is_grad_enabled()

        torch.set_grad_enabled(True)
        torch.enable_grad()

        theta_prime_grad = theta_prime + 0.0
        theta_prime_grad.requires_grad_(True)

        batch_size = batch_ligand.max().item() + 1

        if t.dim() == 1 and t.size(0) == theta_prime.size(0):
            t_batch = torch.zeros(batch_size, device=t.device, dtype=t.dtype)
            for b in range(batch_size):
                mask = batch_ligand == b
                if mask.any():
                    t_batch[b] = t[mask][0]
        elif t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t_value = t.item() if t.dim() == 0 else t[0].item()
            t_batch = torch.full((batch_size,), t_value, device=t.device, dtype=t.dtype)
        else:
            t_batch = t


        batch_ligand_safe = batch_ligand.detach().clone()


        was_training = self.guidance_model.training
        self.guidance_model.train()

        original_requires_grad = {}
        for name, param in self.guidance_model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        for module in self.guidance_model.modules():
            if hasattr(module, 'training'):
                module.train()



        if hasattr(self.guidance_model, 'node_in'):
            node_in_weight = self.guidance_model.node_in.weight
        if hasattr(self.guidance_model, 'mu_head'):
            mu_head_weight = list(self.guidance_model.mu_head.parameters())[0]


        pred_mu, pred_sigma = self.guidance_model(
            theta_prime_grad, pos_t, t_batch, batch_ligand_safe
        )  # pred_mu: [B, 2], pred_sigma: [B, 2]
        test_loss = pred_mu.sum()


        diff = (target_conditions - pred_mu) / (pred_sigma + 1e-8)
        log_prob = -0.5 * torch.sum(diff ** 2) - torch.sum(torch.log(pred_sigma + 1e-8))


        try:
            guidance_grad = torch.autograd.grad(
                log_prob,
                theta_prime_grad,
                retain_graph=False,
                create_graph=False
            )[0]  # [N, K]
        except Exception as e:
            torch.set_grad_enabled(grad_was_enabled)
            return {
                'guidance_probability': theta_prime,
                'guidance_applied': False,
                'guidance_info': {'message': f'梯度计算失败: {e}'}
            }

        guidance_grad = torch.clamp(guidance_grad, min=-10.0, max=10.0)

        grad_mean = guidance_grad.mean().item()
        grad_std = guidance_grad.std().item()
        grad_max = guidance_grad.abs().max().item()

        if t_batch.dim() == 0:
            t_val = t_batch.item()
        else:
            t_val = t_batch.mean().item()

        time_factor = 4.0 * t_val * (1 - t_val)  
        time_factor = max(time_factor, 0.1)

        effective_guidance_scale = guidance_scale * time_factor

        log_theta = torch.log(torch.clamp(theta_prime, min=1e-8))
        log_theta_guided = log_theta + effective_guidance_scale * guidance_grad

        theta_guided = F.softmax(log_theta_guided, dim=-1)

        adjustment = torch.abs(theta_guided - theta_prime).mean().item()
        max_adjustment = torch.abs(theta_guided - theta_prime).max().item()

        guidance_info = {
            'pred_qed': pred_mu[:, 0].mean().item(),
            'pred_sa': pred_mu[:, 1].mean().item(),
            'target_qed': target_conditions[:, 0].mean().item(),
            'target_sa': target_conditions[:, 1].mean().item(),
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_max': grad_max,
            'time_factor': time_factor,
            'effective_guidance_scale': effective_guidance_scale,
            'adjustment': adjustment,
            'max_adjustment': max_adjustment
        }


        if not was_training:
            self.guidance_model.eval()
        for name, param in self.guidance_model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

        torch.set_grad_enabled(grad_was_enabled)

        return {
            'guidance_probability': theta_guided,
            'guidance_applied': True,
            'guidance_info': guidance_info
        }

    def apply_guidance(
        self,
        *args,
        batch=None,
        current_time=None,
        target_conditions=None,
        guidance_scale=1.0,
        theta_t=None,
        pos_t=None,
        t=None,
        batch_ligand=None,
        adaptive_guidance=True,
        **kwargs
    ):


        if self.ablation_no_geometric:
            if not hasattr(self, '_ablation_no_geometric_logged'):
                self._ablation_no_geometric_logged = True

            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'ablation_no_geometric',
                'ablation_mode': 'no_geometric'
            }

        if guidance_scale == 0:
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'guidance_scale_zero',
                'guidance_scale': guidance_scale
            }

        try:
            return self._apply_sota_conditional_guidance(
                args, batch, current_time, target_conditions, guidance_scale,
                theta_t, pos_t, t, batch_ligand, adaptive_guidance, **kwargs
            )
        except Exception as e:
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'sota_guidance_error',
                'error': str(e)
            }

    def _apply_sota_conditional_guidance(
        self, args, batch, current_time, target_conditions, guidance_scale,
        theta_t, pos_t, t, batch_ligand, adaptive_guidance, **kwargs
    ):

        if guidance_scale == 0:
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'guidance_scale_zero_in_core',
                'guidance_scale': guidance_scale
            }


        if len(args) >= 3:
            # 位置参数调用：apply_guidance(batch, current_time, target_conditions, guidance_scale)
            batch = args[0]
            current_time = args[1]
            target_conditions = args[2]
            if len(args) >= 4:
                guidance_scale = args[3]
        else:
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'unrecognized_call_pattern'
            }

        time_str = f"{current_time:.3f}" if current_time is not None else "None"



        if current_time is not None and current_time < 0.01:
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'early_protection',
                'progress': current_time
            }
        if batch is not None:

            if hasattr(batch, 'theta_h_t_for_guidance') and batch.theta_h_t_for_guidance is not None:
                theta_t = batch.theta_h_t_for_guidance
            else:
                return {'guidance_probability': None, 'guidance_applied': False, 'reason': 'missing_theta'}

            if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                pos_t = batch.ligand_pos
            else:
                return {'guidance_probability': None, 'guidance_applied': False, 'reason': 'missing_pos'}

            if hasattr(batch, 'batch_ligand') and batch.batch_ligand is not None:
                batch_ligand = batch.batch_ligand
            else:
                batch_ligand = torch.zeros(theta_t.size(0), dtype=torch.long, device=theta_t.device)

            return self._compute_sota_conditional_guidance(
                theta_t=theta_t,
                pos_t=pos_t,
                target_conditions=target_conditions,
                current_time=current_time,
                guidance_scale=guidance_scale
            )

        else:

            if theta_t is None or pos_t is None:
                return {'guidance_probability': None, 'guidance_applied': False, 'reason': 'missing_inputs'}

            return self._compute_sota_conditional_guidance(
                theta_t=theta_t,
                pos_t=pos_t,
                target_conditions=target_conditions,
                current_time=current_time,
                guidance_scale=guidance_scale
            )

    def _compute_sota_conditional_guidance(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        target_conditions: torch.Tensor,
        current_time: float,
        guidance_scale: float
    ):

        try:
            theta_prime_i = theta_t

            alpha_i = torch.tensor([1.0 - current_time], dtype=torch.float32, device=theta_t.device)  # [1]

            predicted_properties, predicted_sigma = self._predict_conditional_posterior(
                theta_prime_i, pos_t, alpha_i
            )  # [2], [2]

            if target_conditions.dim() > 1:
                target_conditions = target_conditions[0]  # 取第一个样本

            likelihood_factor = self._compute_likelihood_factor(
                predicted_properties, predicted_sigma, target_conditions, guidance_scale
            )

            N, K = theta_prime_i.shape

            pred_qed, pred_sa = predicted_properties[0].item(), predicted_properties[1].item()
            target_qed, target_sa = target_conditions[0].item(), target_conditions[1].item()

            qed_gap = target_qed - pred_qed 
            sa_gap = target_sa - pred_sa 

            total_gap = abs(qed_gap) + abs(sa_gap)

            base_strength = guidance_scale * total_gap * 20.0

            if current_time < 0.5:
                time_adaptive_factor = 1.5  
            elif current_time < 0.8:
                time_adaptive_factor = 2.0 
            else:
                time_adaptive_factor = 2.5 

            guidance_strength = base_strength * time_adaptive_factor

            guidance_strength = torch.clamp(torch.tensor(guidance_strength), min=0.2, max=50.0).item()


            qed_influence = torch.tensor([0.0, 0.50, 1.00, 0.80, 0.10, -0.30, -0.20, -0.50, -0.70, -0.90, 0.0, 0.0, 0.0], device=theta_prime_i.device)
            sa_influence = torch.tensor([0.0, 0.50, 0.70, 0.60, -0.30, -0.70, -0.40, -0.60, -0.80, -1.00, 0.0, 0.0, 0.0], device=theta_prime_i.device)

            if len(qed_influence) != K:
                qed_influence = qed_influence[:K]
                sa_influence = sa_influence[:K]


            guidance_direction = qed_gap * qed_influence + sa_gap * sa_influence  # [K]

            log_theta_prime = torch.log(torch.clamp(theta_prime_i, min=1e-8))  # [N, K]

            guidance_adjustment = guidance_strength * guidance_direction.unsqueeze(0)  # [1, K] -> [N, K]

            entropy = -torch.sum(theta_prime_i * torch.log(torch.clamp(theta_prime_i, min=1e-8)), dim=-1)  # [N]

            log_theta_i = log_theta_prime + guidance_adjustment  # [N, K]


            theta_i_unnormalized = torch.exp(log_theta_i)  # [N, K]
            theta_i = theta_i_unnormalized / torch.sum(theta_i_unnormalized, dim=-1, keepdim=True)  # [N, K]

            guidance_adjustment = theta_i - theta_prime_i  # [N, K]

            max_adjustment = torch.max(torch.abs(guidance_adjustment)).item()
            mean_adjustment = torch.mean(torch.abs(guidance_adjustment)).item()

            min_adjustment_threshold = 1e-8 
            if mean_adjustment < min_adjustment_threshold:
                return {
                    'guidance_probability': None,
                    'guidance_applied': False,
                    'reason': 'adjustment_too_small',
                    'likelihood_factor': likelihood_factor.item()
                }

            return {
                'guidance_probability': theta_i,
                'guidance_applied': True,
                'likelihood_factor': likelihood_factor.item(),
                'predicted_properties': predicted_properties,
                'target_conditions': target_conditions,
                'adjustment_magnitude': mean_adjustment,
                'guidance_method': 'multiplicative_reweighting'
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'guidance_probability': None,
                'guidance_applied': False,
                'reason': 'computation_error',
                'error': str(e)
            }

    def _compute_geometry_importance(self, pos_t: torch.Tensor) -> torch.Tensor:

        N = pos_t.size(0)

        dist_matrix = torch.cdist(pos_t, pos_t)  # [N, N]


        neighbor_counts = torch.sum(dist_matrix < 4.0, dim=-1) - 1 


        density_importance = neighbor_counts.float() / neighbor_counts.max()

        centroid = torch.mean(pos_t, dim=0)  # [3]
        distances_to_center = torch.norm(pos_t - centroid, dim=-1)  # [N]
        centrality_importance = 1.0 - (distances_to_center / distances_to_center.max())


        importance = 0.6 * density_importance + 0.4 * centrality_importance

        return importance  # [N]

    def _predict_conditional_posterior(self, theta_i: torch.Tensor, pos_t: torch.Tensor, alpha_i: torch.Tensor) -> tuple:
  
        try:
            if self.ablation_no_geometric:
                num_atoms = theta_i.size(0)

                qed_estimate = 0.65 - 0.25 * torch.sigmoid((num_atoms - 15) / 8)  # 范围 [0.4, 0.65]
                qed_noise = (torch.rand(1, device=self.device) - 0.5) * 0.15  # [-0.075, 0.075]
                qed_estimate = torch.clamp(qed_estimate + qed_noise, 0.3, 0.7)

                sa_estimate = 0.85 - 0.35 * torch.sigmoid((num_atoms - 18) / 10)  # 范围 [0.5, 0.85]
                sa_noise = (torch.rand(1, device=self.device) - 0.5) * 0.15  # [-0.075, 0.075]
                sa_estimate = torch.clamp(sa_estimate + sa_noise, 0.5, 0.9)

                pred_mu = torch.stack([qed_estimate, sa_estimate]).squeeze()  # [2]
                pred_sigma = torch.tensor([0.25, 0.25], device=self.device)  # [2] 高方差

                if not hasattr(self, '_ablation_posterior_logged'):
                    self._ablation_posterior_logged = True


                return pred_mu, pred_sigma

            num_atoms = theta_i.size(0)
            batch_ligand = torch.zeros(num_atoms, dtype=torch.long, device=theta_i.device)

            t_dummy = alpha_i.expand(1)  # [1]

            with torch.no_grad():
                pred_mu, pred_sigma = self.guidance_model(theta_i, pos_t, t_dummy, batch_ligand)

            if pred_mu.dim() > 1:
                pred_mu = pred_mu[0]  # [2]
                pred_sigma = pred_sigma[0]  # [2]

            return pred_mu, pred_sigma

        except Exception as e:
            return torch.tensor([0.5, 0.5], device=theta_i.device), torch.tensor([0.1, 0.1], device=theta_i.device)

    def _compute_likelihood_factor(self, pred_mu: torch.Tensor, pred_sigma: torch.Tensor,
                                 target_conditions: torch.Tensor, guidance_scale: float) -> torch.Tensor:

        pred_sigma_safe = torch.clamp(pred_sigma, min=1e-6)

        diff = target_conditions - pred_mu  # [2]
        normalized_diff = diff / pred_sigma_safe  # [2]


        qed_gap = abs(normalized_diff[0].item())
        sa_gap = abs(normalized_diff[1].item())


        adaptive_scale = guidance_scale * (1.0 + qed_gap + sa_gap)

        total_gap = torch.sum(normalized_diff ** 2) 

        likelihood_factor = torch.exp(-adaptive_scale * total_gap)

        likelihood_factor = torch.clamp(likelihood_factor, min=min_factor)

        return likelihood_factor.unsqueeze(0)  # [1]

    def _predict_current_properties(self, theta_t: torch.Tensor, pos_t: torch.Tensor, current_time: float = None) -> torch.Tensor:
        if self.guidance_model is None:
            return self._heuristic_property_prediction(theta_t)

        try:

            batch_ligand = torch.zeros(theta_t.size(0), dtype=torch.long, device=theta_t.device)


            if current_time is not None:
                t_dummy = torch.tensor([current_time], device=theta_t.device)
                sampling_progress = current_time 
            else:
                t_dummy = torch.tensor([0.5], device=theta_t.device)  
                sampling_progress = 0.5

            try:
                first_layer = self.guidance_model.node_in
                expected_dim = first_layer.in_features if hasattr(first_layer, 'in_features') else 8

                if theta_t.size(-1) != expected_dim:
                    if expected_dim == 8 and theta_t.size(-1) == 13:
                        atom_type_theta = theta_t[:, :8] 

                    else:
                        atom_type_theta = theta_t
                else:
                    atom_type_theta = theta_t

            except Exception as e:
                if theta_t.size(-1) > 8:
                    atom_type_theta = theta_t[:, :8]
                else:
                    atom_type_theta = theta_t

            theta_sums = atom_type_theta.sum(dim=-1)

            with torch.no_grad():
                pred_mu, pred_sigma = self.guidance_model(atom_type_theta, pos_t, t_dummy, batch_ligand)
    
            if pred_mu is None or pred_mu.numel() == 0:
                return self._heuristic_property_prediction(theta_t)


            discount_factor = sampling_progress
            neutral_values = torch.tensor([0.5, 0.5], device=theta_t.device)

            if pred_mu.dim() > 1:
                raw_pred = pred_mu[0]  # [condition_dim]
            else:
                raw_pred = pred_mu  # [condition_dim]

            if raw_pred.numel() < 2:
                return self._heuristic_property_prediction(theta_t)

            adjusted_pred = discount_factor * raw_pred + (1 - discount_factor) * neutral_values

            adjusted_pred = torch.clamp(adjusted_pred, 0.1, 0.9)

            qed_raw, sa_raw = raw_pred[0].item(), raw_pred[1].item()

            if sa_raw > 0.7:
                sa_desc = "易合成"
            elif sa_raw > 0.4:
                sa_desc = "中等"
            else:
                sa_desc = "难合成"

            return adjusted_pred

        except Exception as e:
            return self._heuristic_property_prediction(theta_t)

    def _heuristic_property_prediction(self, theta_t: torch.Tensor) -> torch.Tensor:
        atom_probs = torch.mean(theta_t, dim=0)  # [K] 平均原子类型分布

        if theta_t.size(-1) >= 4:
            c_prob = atom_probs[0] if theta_t.size(-1) > 0 else 0.0  # C
            n_prob = atom_probs[1] if theta_t.size(-1) > 1 else 0.0  # N
            o_prob = atom_probs[2] if theta_t.size(-1) > 2 else 0.0  # O
            f_prob = atom_probs[3] if theta_t.size(-1) > 3 else 0.0  # F

            hetero_ratio = n_prob + o_prob + f_prob
            carbon_backbone = c_prob

            qed_pred = 0.45 + 0.25 * hetero_ratio + 0.15 * carbon_backbone
            qed_pred = torch.clamp(qed_pred, 0.1, 0.9)
        else:
            qed_pred = 0.5

        simple_atoms = c_prob + n_prob + o_prob  
        complex_atoms = 1.0 - simple_atoms

        sa_pred = 0.6 + 0.2 * simple_atoms - 0.3 * complex_atoms
        sa_pred = torch.clamp(sa_pred, 0.2, 0.9)

        return torch.tensor([qed_pred, sa_pred], device=theta_t.device)

    def _compute_time_weight(self, current_time: float) -> float:
        if current_time is None:
            return 0.0 

        if current_time < 0.5:
            return 0.1
        elif current_time < 0.8:
            progress = (current_time - 0.5) / 0.3
            return 0.1 + 0.7 * progress
        else:
            progress = (current_time - 0.8) / 0.2
            return 0.8 + 0.2 * progress

    def _compute_geometry_constrained_transfer(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        importance_scores: torch.Tensor,
        qed_gap: float,
        sa_gap: float,
        base_strength: float
    ) -> torch.Tensor:
        N, K = theta_t.shape
        guidance_logits = torch.zeros_like(theta_t)  # [N, K]

        effective_strength = base_strength

        protection_mask = importance_scores > 0.98

        dominant_atoms = torch.max(theta_t, dim=-1)[0] > 0.99

        modifiable_atoms = importance_scores < 0.9

        modifiable_mask = modifiable_atoms & (~dominant_atoms) & (~protection_mask)  # [N]

        modifiable_count = modifiable_mask.sum().item()

        if modifiable_count == 0:
            return guidance_logits

        atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'F': 3,
            'P': 4, 'S': 5, 'Cl': 6, 'Br': 7
        }

        for i in range(N):
            if not modifiable_mask[i]:
                continue

            current_probs = theta_t[i]  # [K]
            entropy = -torch.sum(current_probs * torch.log(current_probs.clamp(min=1e-8)))
            entropy_weight = torch.clamp(entropy / torch.log(torch.tensor(K, dtype=torch.float)), 0.1, 1.0)

            importance_weight = 1.0 - importance_scores[i]
            atom_final_strength = effective_strength * entropy_weight * importance_weight

            if qed_gap > 0.01 and K >= 4: 
                if current_probs[atom_types['C']] > 0.1:  # 进一步降低到0.1
                    guidance_logits[i, atom_types['N']] += atom_final_strength * qed_gap * 5.0  # 进一步提高到5.0
                    guidance_logits[i, atom_types['O']] += atom_final_strength * qed_gap * 5.0  # 进一步提高到5.0
                    guidance_logits[i, atom_types['C']] -= atom_final_strength * qed_gap * 3.0  # 进一步提高到3.0

            if sa_gap > 0.02 and K >= 4:  # 进一步降低到0.02
                guidance_logits[i, atom_types['C']] += atom_final_strength * sa_gap * 8.0  # 进一步提高到8.0
                guidance_logits[i, atom_types['N']] += atom_final_strength * sa_gap * 5.0  # 进一步提高到5.0
                if K > 4:
                    guidance_logits[i, atom_types.get('P', 4)] -= atom_final_strength * sa_gap * 3.0  # 进一步提高到3.0
                    guidance_logits[i, atom_types.get('S', 5)] -= atom_final_strength * sa_gap * 2.0  # 进一步提高到2.0

        return guidance_logits

    def _apply_zero_mean_constraint(self, guidance_logits: torch.Tensor) -> torch.Tensor:
        N, K = guidance_logits.shape

        max_guidance = torch.max(torch.abs(guidance_logits))
        if max_guidance < 1e-10: 
            return guidance_logits


        guidance_strength = torch.sum(torch.abs(guidance_logits), dim=-1, keepdim=True)  # [N, 1]

        significant_guidance = guidance_strength > 1e-5  # [N, 1]

        constrained_logits = guidance_logits.clone()

        for i in range(N):
            if not significant_guidance[i]:
                continue

            atom_logits = guidance_logits[i]  # [K]
  
            positive_mask = atom_logits > 0
            negative_mask = atom_logits < 0

            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                positive_sum = atom_logits[positive_mask].sum()
                negative_sum = atom_logits[negative_mask].sum()

                if positive_sum > -negative_sum:
                    scale_factor = -negative_sum / positive_sum * 0.8  # 80%平衡
                    constrained_logits[i][positive_mask] *= scale_factor
                else:
                    scale_factor = positive_sum / (-negative_sum) * 0.8  # 80%平衡
                    constrained_logits[i][negative_mask] *= scale_factor

        final_mean = torch.mean(constrained_logits, dim=-1, keepdim=True)
        constrained_logits = constrained_logits - final_mean

        max_allowed_guidance = 0.1
        constrained_logits = torch.clamp(constrained_logits, -max_allowed_guidance, max_allowed_guidance)


        return constrained_logits

    def _logits_to_probability_adjustment(
        self,
        theta_t: torch.Tensor,
        guidance_logits: torch.Tensor
    ) -> torch.Tensor:
        N, K = theta_t.shape

        max_guidance = torch.max(torch.abs(guidance_logits))
        if max_guidance < 1e-8:
            return theta_t

        if torch.any(theta_t <= 0):
            theta_t = theta_t.clamp(min=1e-8)

        current_logits = torch.log(theta_t)  # [N, K]

        max_probs = torch.max(theta_t, dim=-1)[0]  # [N]
        certainty_factor = 1.0 - max_probs
        certainty_factor = certainty_factor.unsqueeze(-1)  # [N, 1]

        adaptive_guidance = guidance_logits * certainty_factor  # [N, K]

        temperature = 10.0
        guided_logits = current_logits + adaptive_guidance / temperature  # [N, K]

        max_logits = torch.max(guided_logits, dim=-1, keepdim=True)[0]
        stable_logits = guided_logits - max_logits

        guided_probs = torch.softmax(stable_logits, dim=-1)  # [N, K]

        prob_change = torch.norm(guided_probs - theta_t, dim=-1)  # [N]
        max_change = torch.max(prob_change)
        avg_change = torch.mean(prob_change)

        if max_guidance > 1e-6:
        
            if max_change > 0.5: 
                scale_factor = 0.2 / max_change
                conservative_guidance = adaptive_guidance * scale_factor
                guided_logits = current_logits + conservative_guidance / temperature
                max_logits = torch.max(guided_logits, dim=-1, keepdim=True)[0]
                stable_logits = guided_logits - max_logits
                guided_probs = torch.softmax(stable_logits, dim=-1)

                final_change = torch.max(torch.norm(guided_probs - theta_t, dim=-1))

        return guided_probs


def create_geometric_guidance_integrator(
    guidance_model_path: str,
    device: str = 'cuda',
    model_config: dict = None,
    ablation_no_multiplicative: bool = False,
    ablation_no_geometric: bool = False,
    ablation_no_time_consistency: bool = False,
    **kwargs 
):

    return GeometricGuidanceIntegrator(
        guidance_model_path=guidance_model_path,
        device=device,
        model_config=model_config,
        ablation_no_multiplicative=ablation_no_multiplicative,
        ablation_no_geometric=ablation_no_geometric,
        ablation_no_time_consistency=ablation_no_time_consistency
    )


