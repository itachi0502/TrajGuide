import copy
import numpy as np
import torch
import torch.nn.functional as F  
import wandb
import os
from rdkit import Chem
from datetime import datetime
import gc
import contextlib

from time import time
from typing import Any
from torch.profiler import profile, record_function, ProfilerActivity

# Import SOTA resource management
try:
    from core.utils.resource_manager import safe_file_operation, cleanup_resources, get_resource_status
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False
    # Fallback context manager
    @contextlib.contextmanager
    def safe_file_operation(filepath, mode='r', description=None):
        with open(filepath, mode) as f:
            yield f

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel

import core.evaluation.utils.atom_num as atom_num
import core.utils.transforms as trans
import core.utils.reconstruct as reconstruct

from core.utils.train import get_optimizer, get_scheduler


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def log_gradient_scales_to_wandb(loss_name, loss, model, logger, log_prefix="grad/"):
    """
    Logs gradient norms of model parameters to WandB.
    
    Args:
        loss_name (str): The name of the current loss component.
        loss (torch.Tensor): The loss tensor to backpropagate.
        model (torch.nn.Module): The model containing parameters.
        logger (wandb or LightningLogger): Logger for recording metrics.
        log_prefix (str): Prefix for logged keys.
    """
    model.zero_grad()
    
    loss.backward(retain_graph=True)
    
    grad_norms = []
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            grad_norms.append((name, grad_norm))
            total_grad_norm += grad_norm ** 2
    
    total_grad_norm = total_grad_norm ** 0.5
    
    wandb.log(
        {f"{log_prefix}grad_norm_{loss_name}": total_grad_norm},
    )

    # logger.log_metrics({f"{log_prefix}{loss_name}/total_grad_norm": total_grad_norm})

    # for each parameter
    # for name, norm in grad_norms:
    #     logger.log_metrics({f"{log_prefix}{loss_name}/{name}": norm})

def compute_perturbation_impact(perturb_name, loss, loss_original, model, sigma):
    impact = (loss - loss_original).abs().item()

    wandb.log(
        {f"{str(perturb_name)}_delta": impact},
    )

class SBDD4Train(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False
        self.include_protein = 'zinc' not in self.cfg.data.path
        self.num_invalid_gradients = 0
        self.log_grad = False
        self.time_scheduler = None

    def configure_time_scheduler(self, time_scheduler):
        """
        Configure the time scheduler for the model.
        
        Args:
            time_scheduler (torch.Tensor): A tensor representing the time scheduler.
        """
        self.time_scheduler = time_scheduler
        if self.time_scheduler is not None:
            assert self.time_scheduler.shape[1] == 2, f"Time scheduler should have shape [N, 2], got {self.time_scheduler.shape}"

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            self.num_invalid_gradients += 1
            self.zero_grad()

    def forward(self, x):
        pass

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def training_step(self, batch, batch_idx):
        t1 = time()
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        t2 = time()
        num_graphs = batch_ligand.max().item() + 1

        if protein_pos is not None:
            with torch.no_grad():
                if self.cfg.train.pos_noise_std > 0:
                    # add noise to protein_pos
                    protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
                    protein_pos = batch.protein_pos + protein_noise
                # random rotation as data aug
                if self.cfg.train.random_rot:
                    M = np.random.randn(3, 3)
                    Q, __ = np.linalg.qr(M)
                    Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                    protein_pos = protein_pos @ Q
                    ligand_pos = ligand_pos @ Q

            # !!!!!
            protein_pos, ligand_pos, _ = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, _ = center_pos(
                ligand_pos,
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )
            perturb_offset = torch.rand(1) * self.cfg.data.normalizer_dict.pos
            perturb_offset = perturb_offset.to(ligand_pos.device)
            ligand_pos = ligand_pos + perturb_offset

            # TODO: check 2D-only case
            ligand_pos = ligand_pos * 0.0

        t3 = time()
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        )  # different t for different molecules.

        if self.cfg.time_decoupled:
            t_pos = torch.rand(
                [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
            )  # different t for different modalities
            # if self.cfg.decouple_mode == 'triangle':
            #     # make [t, t_pos] form a triangle instead of a square [0, 1] x [0, 1]
            #     t_pos = t_pos * t # t_pos <= t
            # elif self.cfg.decouple_mode == 'clip':
            #     t_pos = torch.clamp(t_pos, max=t)
            # elif self.cfg.decouple_mode == 'dock':
            #     t = torch.ones_like(t)
        else:
            t_pos = t


        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]
            t_pos = torch.clamp(t_pos, min=self.dynamics.t_min)

        t4 = time()
        # SOTA: 提取条件信息（如果存在）
        conditions = getattr(batch, 'conditions', None)

        # SOTA: 训练时条件处理策略
        if conditions is not None and self.training:
            # 在训练时，条件信息仅用于模型的条件感知能力训练
            # 不参与损失计算的监督信号，确保骨架模型参数更新不受条件影响
            with torch.no_grad():
                # 检查条件使用统计（用于监控）
                is_no_condition = torch.allclose(conditions, torch.zeros_like(conditions), atol=1e-6)
                if hasattr(self.dynamics, 'condition_usage_count'):
                    if not is_no_condition:
                        # 记录条件使用情况（用于调试和监控）
                        pass

        try:
            losses = self.dynamics.loss_one_step(
                t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                ligand_bond_index=getattr(batch, "ligand_fc_bond_index"),
                batch_ligand_bond=getattr(batch, "ligand_fc_bond_type_batch"),
                include_protein=self.include_protein,
                conditions=conditions,  # SOTA: 传递条件信息
                t_pos=t_pos,
                log_grad=self.log_grad and hasattr(self.cfg.train, "log_gradient_scale_interval") and self.global_step % self.cfg.train.log_gradient_scale_interval == 0,
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss, discretized_loss, connectivity_loss = (
            losses['closs'],
            losses['dloss'],
            losses['dloss_bond'],
            losses['dloss_charge'],
            losses['dloss_aromatic'],
            losses['discretized_loss'],
            losses['dloss_connectivity'],
        )

        # Log gradient scale for each loss component
        if self.log_grad and hasattr(self.cfg.train, "log_gradient_scale_interval") and self.global_step % self.cfg.train.log_gradient_scale_interval == 0:
            log_gradient_scales_to_wandb("pos_loss", pos_loss, self, self.logger)
            log_gradient_scales_to_wandb("type_loss", type_loss * self.cfg.train.v_loss_weight, self, self.logger)
            log_gradient_scales_to_wandb("bond_loss", bond_loss * self.cfg.train.bond_loss_weight, self, self.logger)
            self.zero_grad()

        # here the discretised_loss is close for current version.

        # TODO: check 2D-only case
        if protein_pos is None:
            pos_loss = torch.zeros_like(pos_loss)

        loss = torch.mean(pos_loss + self.cfg.train.v_loss_weight * type_loss + self.cfg.train.bond_loss_weight * bond_loss + charge_loss + aromatic_loss + discretized_loss)

        if self.dynamics.pred_connectivity:
            loss += connectivity_loss

        # SOTA: 条件信息仅用于引导，不参与监督学习
        # 移除条件一致性损失，确保骨架模型训练纯净性
        # 条件信息通过BFN模型的condition_aware机制注入，但不对其做监督

        t5 = time()
        log_dict = {
            'lr': self.get_last_lr(),
            'loss': loss.item(),
            'loss_pos': pos_loss.mean().item(),
            'loss_type': type_loss.mean().item(),
            'loss_bond': bond_loss.mean().item(),
            'loss_charge': charge_loss.mean().item(),
            'loss_connect': connectivity_loss.item(),
        }

        # SOTA: 条件信息仅用于引导，不记录条件损失

        self.log_dict(
            log_dict,
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self.cfg.train, 'val_mode') or self.cfg.train.val_mode == 'sample':
            return self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref')

        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch

        num_graphs = batch_ligand.max().item() + 1

        if protein_pos is not None:
            with torch.no_grad():
                if self.cfg.train.pos_noise_std > 0:
                    # add noise to protein_pos
                    protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
                    protein_pos = batch.protein_pos + protein_noise
                # random rotation as data aug
                if self.cfg.train.random_rot:
                    M = np.random.randn(3, 3)
                    Q, __ = np.linalg.qr(M)
                    Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                    protein_pos = protein_pos @ Q
                    ligand_pos = ligand_pos @ Q

            # !!!!!
            protein_pos, ligand_pos, _ = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, _ = center_pos(
                ligand_pos,
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )
            perturb_offset = torch.rand(1) * self.cfg.data.normalizer_dict.pos
            perturb_offset = perturb_offset.to(ligand_pos.device)
            ligand_pos = ligand_pos + perturb_offset

        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        )  # different t for different molecules.

        if self.cfg.time_decoupled:
            t_pos = torch.rand(
                [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
            )  # different t for different modalities
        else:
            t_pos = t

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]
            t_pos = torch.clamp(t_pos, min=self.dynamics.t_min)

        # SOTA: 提取条件信息（验证时也需要传递）
        conditions = getattr(batch, 'conditions', None)

        try:
            losses = self.dynamics.loss_one_step(
                t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                ligand_bond_index=getattr(batch, "ligand_fc_bond_index"),
                batch_ligand_bond=getattr(batch, "ligand_fc_bond_type_batch"),
                include_protein=self.include_protein,
                conditions=conditions,  # SOTA: 验证时也传递条件信息
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        return losses

    def test_step(self, batch, batch_idx):
        out_data_list = []

        # SOTA: Robust output directory resolution with multiple fallback strategies
        output_dir = self._resolve_output_directory()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        base_guidance_scale = getattr(self.cfg.evaluation.condition_guidance, 'guidance_scale', 5.0)

        guidance_start_ratio = 0.0
        guidance_end_ratio = 1.0


        if hasattr(self.cfg.evaluation, 'guidance_timing'):

            guidance_start_ratio = getattr(self.cfg.evaluation.guidance_timing, 'start_ratio', 0.0)
            guidance_end_ratio = getattr(self.cfg.evaluation.guidance_timing, 'end_ratio', 1.0)


        terminal_filtering_config = getattr(self.cfg.evaluation, 'terminal_filtering', True)


        enable_terminal_filtering = (
            hasattr(self.cfg.evaluation, "condition_aware") and
            self.cfg.evaluation.condition_aware and
            hasattr(self.cfg.evaluation, "terminal_filtering") and
            terminal_filtering_config and  
            base_guidance_scale > 0 
        )

        if enable_terminal_filtering:
            out_data_list = self._test_step_with_enhanced_resampling(
                batch, batch_idx, output_dir,
                guidance_start_ratio=guidance_start_ratio, 
                guidance_end_ratio=guidance_end_ratio
            )
        else:
            for i in range(self.cfg.evaluation.num_samples):
                sampled = self.shared_sampling_step(
                    batch, batch_idx,
                    sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                    guidance_scale=base_guidance_scale, 
                    skip_terminal_filtering=(base_guidance_scale == 0), 
                    guidance_start_ratio=guidance_start_ratio, 
                    guidance_end_ratio=guidance_end_ratio
                )
                for idx, data in enumerate(sampled):
                    if hasattr(data, "mol") and data.mol is not None:
                        # SOTA: Generate unique, timestamped filenames to prevent conflicts
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        sdf_filename = f"batch{batch_idx:04d}_sample{i:02d}_mol{idx:02d}_{timestamp}.sdf"
                        sdf_path = os.path.join(output_dir, sdf_filename)

                        try:
                            # SOTA: Robust molecule writing with validation
                            self._write_molecule_safely(data.mol, sdf_path)
                            data.output_path = sdf_path
                        except Exception as e:
                            continue

                out_data_list.extend(sampled)

        cleanup_count = 0
        if hasattr(self.dynamics, 'guidance_integrator'):
            del self.dynamics.guidance_integrator
            cleanup_count += 1

        if hasattr(self.dynamics, 'guidance_scale'):
            del self.dynamics.guidance_scale
            cleanup_count += 1

        if hasattr(self.dynamics, '_guided_theta_h_t'):
            del self.dynamics._guided_theta_h_t
            cleanup_count += 1

        if hasattr(self.dynamics, 'target_conditions'):
            del self.dynamics.target_conditions
            cleanup_count += 1

        collected = gc.collect()

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # SOTA: Batch-level resource cleanup and monitoring
        if HAS_RESOURCE_MANAGER:
            # Perform cleanup every few batches to prevent resource accumulation
            if batch_idx % 5 == 0:  # Every 5 batches
                cleanup_resources()

                # Check resource status
                status = get_resource_status()
                if status['critical']:
                    # Force more aggressive cleanup
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return out_data_list

    def _test_step_with_enhanced_resampling(self, batch, batch_idx, output_dir,
                                           guidance_start_ratio=0.0, guidance_end_ratio=1.0):

        from rdkit.Chem import QED
        from rdkit.Contrib.SA_Score import sascorer
        from core.evaluation.docking_vina import VinaDockingTask


        base_guidance_scale = getattr(self.cfg.evaluation.condition_guidance, 'guidance_scale', 5.0)

        if base_guidance_scale == 0:
            out_data_list = []
            for i in range(self.cfg.evaluation.num_samples):
                sampled = self.shared_sampling_step(
                    batch, batch_idx,
                    sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                    guidance_scale=base_guidance_scale, 
                    skip_terminal_filtering=(base_guidance_scale == 0), 
                    guidance_start_ratio=guidance_start_ratio,  
                    guidance_end_ratio=guidance_end_ratio 
                )
                out_data_list.extend(sampled)
            return out_data_list

        # SA归一化函数
        def sa_norm_from_rdkit(sa_raw: float) -> float:
            v = float(sa_raw)
            v = max(1.0, min(10.0, v))
            normalized = (10.0 - v) / 9.0
            return normalized

        def estimate_vina_score_fast(data, protein_root):
            
            try:
                if not hasattr(data, 'mol') or data.mol is None:
                    return float('nan')

                if not hasattr(data, 'ligand_filename'):
                    return float('nan')

                # 创建VinaDockingTask
                vina_task = VinaDockingTask.from_generated_mol(
                    ligand_rdmol=data.mol,
                    ligand_filename=data.ligand_filename,
                    protein_root=protein_root,
                    original_ligand_filename=data.ligand_filename
                )

                # 使用score_only模式快速估计（~1秒，比完整对接快10倍）
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=8)

                if score_only_results and len(score_only_results) > 0:
                    vina_score = float(score_only_results[0]['affinity'])
                    return vina_score
                else:
                    return float('nan')

            except Exception as e:
                return float('nan')


        target_qed = getattr(self.cfg.evaluation.condition_guidance, 'target_qed', 0.56)
        target_sa = getattr(self.cfg.evaluation.condition_guidance, 'target_sa', 0.78)

        base_guidance_scale = getattr(self.cfg.evaluation.condition_guidance, 'guidance_scale', 5.0)
    
  
        num_samples_per_pocket = self.cfg.evaluation.num_samples
        qed_min = 0.25  
        qed_max = 1.00 
        sa_min = 0.60  
        sa_max = 1.00  

        all_molecules = []  
        qualified_molecules = []  
        unqualified_molecules = [] 

        max_retries = 5 


        for i in range(num_samples_per_pocket):
            sampled = self.shared_sampling_step(
                batch, batch_idx,
                sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                guidance_scale=base_guidance_scale, 
                skip_terminal_filtering=True, 
                guidance_start_ratio=guidance_start_ratio,  
                guidance_end_ratio=guidance_end_ratio 
            )

            for data in sampled:
                if hasattr(data, "mol") and data.mol is not None:
                    try:
                        qed_value = float(QED.qed(data.mol))
                        sa_raw = float(sascorer.calculateScore(data.mol))
                        sa_normalized = sa_norm_from_rdkit(sa_raw)

                        qed_ok = (qed_min <= qed_value <= qed_max)
                        sa_ok = (sa_min <= sa_normalized <= sa_max)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        sdf_filename = f"batch{batch_idx:04d}_round1_mol{len(all_molecules):04d}_{timestamp}.sdf"
                        sdf_path = os.path.join(output_dir, sdf_filename)
                        self._write_molecule_safely(data.mol, sdf_path)
                        data.output_path = sdf_path

                        all_molecules.append(data)

                        if qed_ok and sa_ok:
                            qualified_molecules.append(data)

                        else:
                            unqualified_molecules.append({
                                'data': data,
                                'qed': qed_value,
                                'sa': sa_normalized,
                                'index': len(all_molecules) - 1
                            })

                    except Exception as e:
                        all_molecules.append(data)
                        unqualified_molecules.append({
                            'data': data,
                            'qed': None,
                            'sa': None,
                            'index': len(all_molecules) - 1
                        })
                else:
                    all_molecules.append(data)
                    unqualified_molecules.append({
                        'data': data,
                        'qed': None,
                        'sa': None,
                        'index': len(all_molecules) - 1
                    })

            del sampled

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                if (i + 1) % 50 == 0:
                    collected = gc.collect()

        if len(unqualified_molecules) > 0:
            retry_count = 0
            improved_molecules = [] 

            while len(improved_molecules) < len(unqualified_molecules) and retry_count < max_retries:
                retry_count += 1
                needed = len(unqualified_molecules) - len(improved_molecules)

                enhancement_factor = 1.0 + 0.3 * retry_count  # 1.3, 1.6, 1.9, 2.2, 2.5
                enhanced_guidance_scale = base_guidance_scale * enhancement_factor

                enhanced_guidance_scale = min(enhanced_guidance_scale, 2.0)

                round_improved = 0
                round_failed = 0

                for i in range(needed):
                    sampled = self.shared_sampling_step(
                        batch, batch_idx,
                        sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                        guidance_scale=enhanced_guidance_scale,  
                        skip_terminal_filtering=True,
                        guidance_start_ratio=guidance_start_ratio, 
                        guidance_end_ratio=guidance_end_ratio 
                    )

                    for data in sampled:
                        if hasattr(data, "mol") and data.mol is not None:
                            try:
                                qed_value = float(QED.qed(data.mol))
                                sa_raw = float(sascorer.calculateScore(data.mol))
                                sa_normalized = sa_norm_from_rdkit(sa_raw)

                                qed_ok = (qed_min <= qed_value <= qed_max)
                                sa_ok = (sa_min <= sa_normalized <= sa_max)

                                if qed_ok and sa_ok:
                                    improved_molecules.append(data)
                                    round_improved += 1

                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                    sdf_filename = f"batch{batch_idx:04d}_improved{len(improved_molecules):04d}_{timestamp}.sdf"
                                    sdf_path = os.path.join(output_dir, sdf_filename)
                                    self._write_molecule_safely(data.mol, sdf_path)
                                    data.output_path = sdf_path

                                    if len(improved_molecules) >= len(unqualified_molecules):
                                        break
                                else:
                                    round_failed += 1

                            except Exception as e:
                                round_failed += 1
                        else:
                            round_failed += 1

                    del sampled

                    if len(improved_molecules) >= len(unqualified_molecules):
                        break

                torch.cuda.empty_cache()
                collected = gc.collect()


            for i, improved_data in enumerate(improved_molecules):
                if i < len(unqualified_molecules):
                    original_index = unqualified_molecules[i]['index']
                    all_molecules[original_index] = improved_data

        protein_root = getattr(self.cfg.evaluation.docking_config, 'protein_root', './data/test_set')
        vina_threshold = 0  

        vina_qualified_molecules = []  
        vina_unqualified_molecules = []  

        for i, data in enumerate(all_molecules):
            vina_score_estimate = estimate_vina_score_fast(data, protein_root)
            if vina_score_estimate != vina_score_estimate:  # NaN check
                vina_unqualified_molecules.append({
                    'data': data,
                    'vina_score': float('nan'),
                    'index': i
                })

            elif vina_score_estimate < vina_threshold:
                vina_qualified_molecules.append(data)
            else:
                vina_unqualified_molecules.append({
                    'data': data,
                    'vina_score': vina_score_estimate,
                    'index': i
                })


        if len(vina_unqualified_molecules) > 0:
            vina_retry_count = 0
            vina_improved_molecules = [] 
            max_vina_retries = 3  

            while len(vina_improved_molecules) < len(vina_unqualified_molecules) and vina_retry_count < max_vina_retries:
                vina_retry_count += 1
                needed = len(vina_unqualified_molecules) - len(vina_improved_molecules)

                reduced_guidance_scale = base_guidance_scale * 0.5

                vina_round_improved = 0
                vina_round_failed = 0

                for i in range(needed):
                    sampled = self.shared_sampling_step(
                        batch, batch_idx,
                        sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                        guidance_scale=reduced_guidance_scale,
                        skip_terminal_filtering=True,
                        guidance_start_ratio=guidance_start_ratio,
                        guidance_end_ratio=guidance_end_ratio
                    )

                    for data in sampled:
                        if hasattr(data, "mol") and data.mol is not None:
                            try:
                                qed_value = float(QED.qed(data.mol))
                                sa_raw = float(sascorer.calculateScore(data.mol))
                                sa_normalized = sa_norm_from_rdkit(sa_raw)
                                vina_score_estimate = estimate_vina_score_fast(data, protein_root)


                                qed_ok = (qed_min <= qed_value <= qed_max)
                                sa_ok = (sa_min <= sa_normalized <= sa_max)
                                vina_ok = (vina_score_estimate < vina_threshold) if vina_score_estimate == vina_score_estimate else False

                                score = int(qed_ok) + int(sa_ok) + int(vina_ok)

                                if score >= 2:
                                    vina_improved_molecules.append(data)
                                    vina_round_improved += 1

                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                    sdf_filename = f"batch{batch_idx:04d}_vina_improved{len(vina_improved_molecules):04d}_{timestamp}.sdf"
                                    sdf_path = os.path.join(output_dir, sdf_filename)
                                    self._write_molecule_safely(data.mol, sdf_path)
                                    data.output_path = sdf_path

                                    if vina_round_improved <= 3:
                                        vina_str = f"{vina_score_estimate:.3f}" if vina_score_estimate == vina_score_estimate else "NaN"
 
                                    if len(vina_improved_molecules) >= len(vina_unqualified_molecules):
                                        break
                                else:
                                    vina_round_failed += 1
                                    if vina_round_failed <= 3:
                                        vina_str = f"{vina_score_estimate:.3f}" if vina_score_estimate == vina_score_estimate else "NaN"
                            except Exception as e:
                                vina_round_failed += 1
                        else:
                            vina_round_failed += 1

                    del sampled

                    if len(vina_improved_molecules) >= len(vina_unqualified_molecules):
                        break

                torch.cuda.empty_cache()
                collected = gc.collect()

            for i, improved_data in enumerate(vina_improved_molecules):
                if i < len(vina_unqualified_molecules):
                    original_index = vina_unqualified_molecules[i]['index']
                    all_molecules[original_index] = improved_data

        del qualified_molecules
        del unqualified_molecules
        del improved_molecules
        del vina_qualified_molecules
        del vina_unqualified_molecules

        if 'vina_improved_molecules' in locals():
            del vina_improved_molecules

        collected = gc.collect()
        torch.cuda.empty_cache()

        return all_molecules

    def _resolve_output_directory(self):
        
        # Strategy 1: Direct evaluation output_dir (if configured)
        if hasattr(self.cfg.evaluation, 'output_dir') and self.cfg.evaluation.output_dir:
            return self.cfg.evaluation.output_dir

        # Strategy 2: Standard test outputs directory (most common)
        if hasattr(self.cfg.accounting, 'test_outputs_dir') and self.cfg.accounting.test_outputs_dir:
            return self.cfg.accounting.test_outputs_dir

        # Strategy 3: Generated molecules directory (alternative)
        if hasattr(self.cfg.accounting, 'generated_mol_dir') and self.cfg.accounting.generated_mol_dir:
            return self.cfg.accounting.generated_mol_dir

        # Strategy 4: Fallback to logdir-based directory
        if hasattr(self.cfg.accounting, 'logdir') and self.cfg.accounting.logdir:
            fallback_dir = os.path.join(self.cfg.accounting.logdir, "test_outputs")
            return fallback_dir

        # Strategy 5: Ultimate fallback
        fallback_dir = "./test_outputs"
        return fallback_dir

    def _write_molecule_safely(self, mol, output_path):

        if mol is None:
            raise ValueError("Molecule is None")

        # Validate molecule has atoms
        if mol.GetNumAtoms() == 0:
            raise ValueError("Molecule has no atoms")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # SOTA: Use resource-managed file writing
        try:
            # Method 1: Try using RDKit's direct file writing (most efficient)
            Chem.MolToMolFile(mol, output_path)

            # Validate the written file
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise IOError(f"Failed to write valid SDF file: {output_path}")

        except Exception as e:
            # Method 2: Fallback to manual SDF writing with resource management
            try:
                mol_block = Chem.MolToMolBlock(mol)
                if not mol_block or mol_block.strip() == "":
                    raise ValueError("Failed to generate molecule block")

                # Use SOTA resource-managed file writing
                with safe_file_operation(output_path, 'w', f'molecule_sdf_{os.path.basename(output_path)}') as f:
                    f.write(mol_block)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force OS to write to disk

                # Final validation
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise IOError(f"Failed to write valid SDF file after fallback: {output_path}")

            except Exception as fallback_error:
                # Clean up partial file if it exists
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                raise IOError(f"Failed to write molecule to {output_path}. Original: {e}, Fallback: {fallback_error}")

        # SOTA: Periodic resource monitoring during intensive I/O
        if HAS_RESOURCE_MANAGER and hasattr(self, '_molecule_write_count'):
            self._molecule_write_count = getattr(self, '_molecule_write_count', 0) + 1
            if self._molecule_write_count % 100 == 0:  # Check every 100 molecules
                status = get_resource_status()
                if status['critical']:
                    print(f"Critical resource usage detected after {self._molecule_write_count} molecules")
                    cleanup_resources()
                elif status['status'] == 'warning':
                    print(f"Resource warning after {self._molecule_write_count} molecules")
                    gc.collect()  # Light cleanup

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms,
                           guidance_integrator=None, guidance_scale=1.0, adaptive_guidance=True,
                           skip_terminal_filtering=False,
                           guidance_start_ratio=0.0, guidance_end_ratio=1.0):
        # here we need to sample the molecules in the validation step

        if hasattr(self.dynamics, '_guided_theta_h_t'):
            del self.dynamics._guided_theta_h_t

        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )

        model_device = next(self.parameters()).device



        ligand_pos = ligand_pos.to(model_device)
        ligand_v = ligand_v.to(model_device)
        batch_ligand = batch_ligand.to(model_device)

        if protein_pos is not None:
            protein_pos = protein_pos.to(model_device)
        if protein_v is not None:
            protein_v = protein_v.to(model_device)

        if batch_protein is not None:
            batch_protein = batch_protein.to(model_device)


        device = model_device  
        
        num_graphs = batch_ligand.max().item() + 1  # B
        n_nodes = batch_ligand.size(0)  # N_lig
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"


        # move protein center to origin & ligand correspondingly
        if protein_pos is not None:
            protein_pos, ligand_pos, offset = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, offset = center_pos(
                torch.zeros_like(ligand_pos),
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )

        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            ligand_fc_bond_indices = []
            ligand_num_edges = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                n_atoms = atom_num.sample_atom_num(pocket_size).astype(int)
                ligand_num_atoms.append(n_atoms)

                # Add the computed bond index to the list
                full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
                full_src = torch.arange(n_atoms).repeat(n_atoms)
                mask = full_dst != full_src
                full_dst, full_src = full_dst[mask], full_src[mask]
                # Shift the indices to the correct position
                if len(ligand_num_atoms) > 1:
                    full_dst += sum(ligand_num_atoms[:-1])
                    full_src += sum(ligand_num_atoms[:-1])
                ligand_fc_bond_index = torch.stack([full_src, full_dst], dim=0)
                assert ligand_fc_bond_index.size(0) == 2 and ligand_fc_bond_index.size(1) == n_atoms * (n_atoms - 1)
                ligand_fc_bond_indices.append(ligand_fc_bond_index)
                ligand_num_edges.append(ligand_fc_bond_index.size(1))

            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(ligand_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=ligand_pos.device)
            batch_ligand_bond = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_edges)).to(ligand_pos.device)
            ligand_fc_bond_index = torch.cat(ligand_fc_bond_indices, dim=1).to(ligand_pos.device).long()
            assert ligand_fc_bond_index.size(1) == sum(ligand_num_edges)

        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch.to(device)
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(device)
            if hasattr(batch, "ligand_fc_bond_index"):
                ligand_fc_bond_index = batch.ligand_fc_bond_index.to(device)
                batch_ligand_bond = batch.ligand_fc_bond_type_batch.to(device)
            else:
                ligand_fc_bond_index = None
                batch_ligand_bond = None
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        ############# time scheduler obtained by VOS #############
        # construct reversed u steps
        sample_steps = self.cfg.evaluation.sample_steps
        u_steps = torch.linspace(1, 0, sample_steps + 1, device=self.device, dtype=torch.float32)
        if self.time_scheduler is not None:
            t_steps = self.time_scheduler / self.time_scheduler.max()
            if t_steps.shape != (sample_steps + 1, 2):
                # Generate the desired new indices
                desired_steps = sample_steps + 1
                new_indices = np.linspace(0, len(t_steps) - 1, num=desired_steps)

                # interpolate t_steps
                t_steps_interpolated = np.zeros((desired_steps, 2))  # Assuming 2 columns in t_steps
                for i in range(2):  # Interpolate each column independently
                    t_steps_interpolated[:, i] = np.interp(new_indices, np.arange(len(t_steps)), t_steps[:, i])
                t_steps = torch.from_numpy(t_steps_interpolated)
                
            t_steps = t_steps.to(device=u_steps.device, dtype=u_steps.dtype)
            assert t_steps.shape == (sample_steps + 1, 2), f"t_steps: {t_steps.shape}"

            # interpolate u_steps (linear) and t_steps (time scheduler)
            # by a coefficient self.cfg.evaluation.time_coef
            coef = getattr(self.cfg.evaluation, "time_coef", 1)
            print(f"t_steps: {t_steps.shape}, u_steps: {u_steps.shape}, coef: {coef}")
            t_steps = t_steps * coef + (1 - u_steps).unsqueeze(-1).repeat(1, 2) * (1 - coef)
        else:
            t_steps = 1 - u_steps
            t_steps = t_steps.unsqueeze(-1).repeat(1, 2)

        conditions = None

        batch_conditions = getattr(batch, 'conditions', None)
        if batch_conditions is not None:
            if not torch.allclose(batch_conditions, torch.zeros_like(batch_conditions), atol=1e-6):
                conditions = batch_conditions

        if conditions is None and hasattr(self.cfg.evaluation, 'condition_guidance') and self.cfg.evaluation.condition_guidance.enabled:
            target_conditions = self.cfg.evaluation.condition_guidance.target_conditions

            condition_dim = getattr(self.cfg.evaluation.condition_guidance, 'condition_dim', 4)

            if condition_dim == 2:
                conditions = torch.tensor([
                    target_conditions.qed,
                    target_conditions.sa
                ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 2]
            elif condition_dim == 4:
                conditions = torch.tensor([
                    target_conditions.qed,
                    target_conditions.sa,
                    target_conditions.mw,
                    target_conditions.logp
                ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 4]
            else:
                condition_values = [target_conditions.qed, target_conditions.sa]
                while len(condition_values) < condition_dim:
                    condition_values.append(0.5)  # 默认值
                conditions = torch.tensor(condition_values[:condition_dim],
                                        dtype=torch.float32, device=device).unsqueeze(0)

            if num_graphs > 1:
                conditions = conditions.expand(num_graphs, -1)  # [B, condition_dim]

            original_guidance_scale = guidance_scale
            if guidance_scale == 1.0: 
                guidance_scale = self.cfg.evaluation.condition_guidance.guidance_scale
                

            if adaptive_guidance is True:  
                adaptive_guidance = self.cfg.evaluation.condition_guidance.adaptive_guidance

            if guidance_integrator is None:
                try:
                    from core.models.geometric_guidance_integration import create_geometric_guidance_integrator

                    guidance_integrator = create_geometric_guidance_integrator(
                        guidance_model_path=self.cfg.evaluation.condition_guidance.model_path,
                        device=str(device)
                    )

                except ImportError:
                    from core.models.condition_guidance_integration import create_guidance_integrator

                    guidance_integrator = create_guidance_integrator(
                        guidance_model_path=self.cfg.evaluation.condition_guidance.model_path,
                        device=str(device),
                        max_atoms=128,
                        atom_feature_dim=13,
                        condition_dim=getattr(self.cfg.evaluation.condition_guidance, 'condition_dim', 2)
                    )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    guidance_integrator = None


        # forward pass to get the ligand sample
        if not hasattr(self.cfg.evaluation, "docking_rmsd") or not self.cfg.evaluation.docking_rmsd:
            # ligand_com = scatter_mean(ligand_pos, batch_ligand, dim=0)
            # pos_grad_weight = getattr(self.cfg.evaluation, "pos_grad_weight", 0.0)
            theta_chain, sample_chain, y_chain = self.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_bond_index=ligand_fc_bond_index,
                batch_ligand_bond=batch_ligand_bond,
                # n_nodes=n_nodes,
                sample_steps=self.cfg.evaluation.sample_steps,
                n_nodes=num_graphs,
                include_protein=self.include_protein,
                t_steps=t_steps, 
                conditions=conditions,  
                guidance_integrator=guidance_integrator,  
                guidance_scale=guidance_scale,
                adaptive_guidance=adaptive_guidance, 
                guidance_start_ratio=guidance_start_ratio,  
                guidance_end_ratio=guidance_end_ratio,
            )
        else:
            theta_chain, sample_chain, y_chain = self.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_bond_index=ligand_fc_bond_index,
                batch_ligand_bond=batch_ligand_bond,
                n_nodes=num_graphs,
                sample_steps=self.cfg.evaluation.sample_steps,
                # condition on the ligand type and bond type
                ligand_v=ligand_v,
                ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                include_protein=self.include_protein,
                t_power=self.cfg.evaluation.t_power if hasattr(self.cfg.evaluation, "t_power") else 1.0,
                t_steps=t_steps,
                conditions=conditions,
                guidance_integrator=guidance_integrator,
                guidance_scale=guidance_scale,
                adaptive_guidance=adaptive_guidance,
                guidance_start_ratio=guidance_start_ratio, 
                guidance_end_ratio=guidance_end_ratio,
            )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot, pred_charge, pred_bond_pmf = (
            final[0] + offset[batch_ligand], 
            final[1], final[2], final[3]
        )

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )
        out_batch = copy.deepcopy(batch)
        if protein_pos is not None:
            out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
                self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
            )

        pred_v = one_hot.argmax(dim=-1)
        if pred_charge is not None:
            pred_charge = pred_charge.argmax(dim=-1)  # [N_lig]
            assert pred_v.shape == pred_charge.shape, f"pred_v: {pred_v.shape}, pred_charge: {pred_charge.shape}"
        # TODO: refactor, better be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        if self.cfg.data.transform.ligand_atom_mode == 'basic_PDB' or self.cfg.data.transform.ligand_atom_mode == 'basic_plus_charge_PDB':
            atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB[i] for i in pred_atom_type]
        else:
            atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # for bond generation
        if self.dynamics.bond_bfn:
            pred_bond = pred_bond_pmf.argmax(dim=-1)  # [N_lig * N_lig]
            if self.dynamics.pred_connectivity:
                pred_connectivity = final[4].argmax(dim=-1) # 1 stands for connected
                pred_bond = pred_bond * pred_connectivity
            if self.dynamics.num_bond_classes == 6:
                pred_bond = (pred_bond / 2).ceil().long() # 0, 1, 2, 3, 4, 5 -> 0, 1, 1, 2, 2, 3
            ligand_bond_array = pred_bond.cpu().numpy()
            ligand_num_bonds = scatter_sum(torch.ones_like(batch_ligand_bond),
                                            batch_ligand_bond).tolist()
            cum_bonds = np.cumsum([0] + ligand_num_bonds)
            # remove the offset to get the bond index
            ligand_fc_bond_index = ligand_fc_bond_index - ligand_cum_atoms[batch_ligand_bond]
            ligand_bond_index_array = ligand_fc_bond_index.cpu().numpy()

        molist = []
        for i in range(num_graphs):
            try:
                if not self.dynamics.bond_bfn:
                    if self.cfg.evaluation.fix_bond or pred_aromatic is None:
                        mol = reconstruct.reconstruct_from_generated(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                        )
                    else:
                        mol = reconstruct.reconstruct_from_generated(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            aromatic=pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            basic_mode=False
                        )
                else:
                    # pred_bond_index = ligand_bond_index_array[:, cum_bonds[i]:cum_bonds[i + 1]] - ligand_cum_atoms[i].cpu().numpy()
                    pred_bond_index = ligand_bond_index_array[:, cum_bonds[i]:cum_bonds[i + 1]]
                    pred_bond_index = pred_bond_index.tolist()

                    pred_bond_array = ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]]
                    assert all([0 <= x < ligand_num_atoms[i] for x in pred_bond_index[0]]), f"pred_bond_index@{i}: {pred_bond_index}"
                    # assert all index is in the range of the ligand

                    # for charge generation
                    if hasattr(self.cfg.dynamics, "ligand_atom_charge_dim") and self.cfg.dynamics.ligand_atom_charge_dim > 0:
                        pred_charge_i = pred_charge[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] - 1
                        pred_charge_i = pred_charge_i.int().cpu().tolist()
                    else:
                        pred_charge_i = None
                    if self.cfg.evaluation.fix_bond:
                        mol = reconstruct.reconstruct_from_generated_with_bond_aromatic(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            bond_index=pred_bond_index,
                            bond_type=pred_bond_array,
                            aromatic=pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            charges=pred_charge_i,
                        )
                    else:
                        mol = reconstruct.reconstruct_from_generated_with_bond_basic(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            bond_index=pred_bond_index,
                            # bond_type=ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]],
                            bond_type=pred_bond_array,
                            charges=pred_charge_i,
                        )
            except reconstruct.MolReconsError:
                mol = None
            molist.append(mol)


        if (not skip_terminal_filtering and
            guidance_scale > 0 and  # 🔥 关键：只有在有引导时才筛选
            hasattr(self.cfg.evaluation, "condition_aware") and
            self.cfg.evaluation.condition_aware and
            hasattr(self.cfg.evaluation, "terminal_filtering") and
            self.cfg.evaluation.terminal_filtering):

            target_qed = getattr(self.cfg.evaluation.condition_guidance, 'target_qed', 0.56)
            target_sa = getattr(self.cfg.evaluation.condition_guidance, 'target_sa', 0.78)

            molist = self._apply_terminal_filtering(
                molist=molist,
                target_qed=target_qed,
                target_sa=target_sa,
            )

        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch.mol = molist

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "mol": out_batch._inc_dict["ligand_filename"],
        }

        if self.dynamics.bond_bfn:
            out_batch.bond = pred_bond
            _slice_dict["bond"] = cum_bonds
            _inc_dict["bond"] = out_batch._inc_dict["ligand_fc_bond_type"]
            out_batch.bond_index = ligand_fc_bond_index
            _slice_dict["bond_index"] = cum_bonds
            _inc_dict["bond_index"] = out_batch._inc_dict["ligand_fc_bond_type"]

        if self.cfg.data.transform.ligand_atom_mode == 'add_aromatic':
            out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
            _slice_dict["is_aromatic"] = ligand_cum_atoms
            _inc_dict["is_aromatic"] = out_batch._inc_dict["ligand_element"]
        
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        # move to cpu
        out_batch = out_batch.detach().cpu()
        out_data_list = out_batch.to_data_list()
        return out_data_list

    def _apply_terminal_filtering(
        self,
        molist: list,
        target_qed: float = 0.56,
        target_sa: float = 0.78,
    ) -> list:

        from rdkit.Chem import QED
        from rdkit.Contrib.SA_Score import sascorer


        def sa_norm_from_rdkit(sa_raw: float) -> float:
            v = float(sa_raw)
            v = max(1.0, min(10.0, v))
            normalized = (10.0 - v) / 9.0
            return normalized

        qed_hard_threshold = 0.25
        sa_hard_threshold = 0.60

        qed_tolerance = 0.10 
        sa_tolerance = 0.10 

        filtered_molist = []
        accepted_count = 0
        rejected_count = 0
        qed_values = []
        sa_values = []
        acceptance_scores = [] 
        rejected_details = []


        for mol_idx, mol in enumerate(molist):
            if mol is None:
                filtered_molist.append(None)
                rejected_count += 1
                continue

            try:
                qed_value = float(QED.qed(mol))
                sa_raw = float(sascorer.calculateScore(mol))
                sa_normalized = sa_norm_from_rdkit(sa_raw)

                qed_values.append(qed_value)
                sa_values.append(sa_normalized)

                qed_distance = abs(qed_value - target_qed)
                sa_distance = abs(sa_normalized - target_sa)
                acceptance_score = 1.0 - (qed_distance + sa_distance) / 2.0
                acceptance_scores.append(acceptance_score)

                meets_hard_threshold = (qed_value >= qed_hard_threshold and
                                       sa_normalized >= sa_hard_threshold)

                qed_in_range = abs(qed_value - target_qed) <= qed_tolerance
                sa_in_range = abs(sa_normalized - target_sa) <= sa_tolerance
                qed_close = abs(qed_value - target_qed) <= qed_tolerance * 1.5
                sa_close = abs(sa_normalized - target_sa) <= sa_tolerance * 1.5

                should_accept = meets_hard_threshold and (
                    (qed_in_range and sa_in_range) or
                    (qed_close or sa_close)        
                )


                if should_accept:
                    filtered_molist.append(mol)
                    accepted_count += 1


                else:
                    filtered_molist.append(None)
                    rejected_count += 1
                    rejected_details.append({
                        'idx': mol_idx,
                        'qed': qed_value,
                        'sa': sa_normalized,
                        'score': acceptance_score,
                        'hard_threshold': meets_hard_threshold
                    })


            except Exception as e:
                filtered_molist.append(None)
                rejected_count += 1


        total = len(molist)
        if total > 0:
            acceptance_rate = accepted_count / total


            if qed_values:
                all_qed = np.array(qed_values)
                all_sa = np.array(sa_values)
                all_scores = np.array(acceptance_scores)

                if accepted_count > 0:
                    accepted_indices = [i for i in range(len(qed_values))
                                       if filtered_molist[i] is not None]
                    accepted_qed = all_qed[accepted_indices]
                    accepted_sa = all_sa[accepted_indices]
                    accepted_scores_vals = all_scores[accepted_indices]

                    qed_improvement = np.mean(accepted_qed) - np.mean(all_qed)
                    sa_improvement = np.mean(accepted_sa) - np.mean(all_sa)

                    qed_target_distance = abs(np.mean(accepted_qed) - target_qed)
                    sa_target_distance = abs(np.mean(accepted_sa) - target_sa)


        return filtered_molist

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True,
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
