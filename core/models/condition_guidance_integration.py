import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import os
from pathlib import Path

try:
    from train_guidance_mol import MolPilotGuidanceNetwork
except ImportError:
    MolPilotGuidanceNetwork = None


class MolPilotGuidanceIntegrator:

    def __init__(self, guidance_model_path: str, max_atoms: int = 128,
                 atom_feature_dim: int = 13, condition_dim: int = 2, device: str = 'cuda'):
        self.max_atoms = max_atoms
        self.atom_feature_dim = atom_feature_dim
        self.condition_dim = condition_dim
        self.device = device

        self.guidance_model = self._load_guidance_model(guidance_model_path)

        if self.guidance_model is not None:
            self.guidance_model.eval()
        else:
            raise ValueError(f"{guidance_model_path}")
    
    def _load_guidance_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)


            from train_condition_guidance_network import ConditionGuidanceNetwork

            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                max_atoms = config.get('max_atoms', 128)
                atom_feature_dim = config.get('atom_feature_dim', 13)
                condition_dim = config.get('condition_dim', 2)
                hidden_dim = config.get('hidden_dim', 256)
            else:
                max_atoms = 128
                atom_feature_dim = 13
                condition_dim = 2
                hidden_dim = 256

            model = ConditionGuidanceNetwork(
                max_atoms=max_atoms,
                atom_feature_dim=atom_feature_dim,
                condition_dim=condition_dim,
                hidden_dim=hidden_dim
            ).to(self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # PyTorch Lightning格式：{'state_dict': {...}, ...}
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()


            return model

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def graph_to_guidance_input(self, batch, current_time: float):

        try:
            if hasattr(batch, 'batch_ligand'):
                batch_indices = batch.batch_ligand
            elif hasattr(batch, 'ligand_element_batch'):
                batch_indices = batch.ligand_element_batch
            else:
                n = batch.ligand_atom_feature_full.size(0) if hasattr(batch, 'ligand_atom_feature_full') else 1
                batch_indices = torch.zeros(n, device=self.device, dtype=torch.long)

            batch_size = int(torch.unique(batch_indices).size(0))

            if hasattr(batch, 'theta_h_t_for_guidance') and batch.theta_h_t_for_guidance is not None:
                theta_raw = batch.theta_h_t_for_guidance  # [N_ligand, atom_feature_dim]

                theta_atoms = torch.zeros(batch_size, self.max_atoms, self.atom_feature_dim,
                                        device=theta_raw.device, dtype=theta_raw.dtype)
                atom_mask = torch.zeros(batch_size, self.max_atoms, device=theta_raw.device)

                for mol_idx in range(batch_size):
                    mol_atoms = theta_raw[batch_indices == mol_idx]  # [N_mol_atoms, atom_feature_dim]
                    n_atoms = min(mol_atoms.size(0), self.max_atoms)
                    if n_atoms > 0:
                        theta_atoms[mol_idx, :n_atoms] = mol_atoms[:n_atoms]
                        atom_mask[mol_idx, :n_atoms] = 1.0

                t = torch.full((batch_size, 1), current_time, device=theta_raw.device)

                return theta_atoms, t, atom_mask
            else:
                if hasattr(batch, 'ligand_atom_feature_full'):
                    atom_features = batch.ligand_atom_feature_full  # [N_atoms, atom_feature_dim]

                    theta_atoms = torch.zeros(batch_size, self.max_atoms, self.atom_feature_dim,
                                            device=atom_features.device, dtype=atom_features.dtype)
                    atom_mask = torch.zeros(batch_size, self.max_atoms, device=atom_features.device)

                    for mol_idx in range(batch_size):
                        mol_atoms = atom_features[batch_indices == mol_idx]
                        n_atoms = min(mol_atoms.size(0), self.max_atoms)
                        if n_atoms > 0:
                            theta_atoms[mol_idx, :n_atoms] = mol_atoms[:n_atoms]
                            atom_mask[mol_idx, :n_atoms] = 1.0

                    t = torch.full((batch_size, 1), current_time, device=atom_features.device)

                    return theta_atoms, t, atom_mask
                else:
                    return None, None, None

        except Exception as e:
            batch_size = 1
            theta = torch.ones(batch_size, self.seq_len, self.vocab_size, device=self.device) / self.vocab_size
            t = torch.full((batch_size, 1), current_time, device=self.device)
            return theta, t
    
    def _convert_graph_to_sequence_enhanced(self, batch, batch_size: int, batch_indices: torch.Tensor) -> torch.Tensor:

        try:
            if hasattr(batch, 'theta_h_t_for_guidance') and batch.theta_h_t_for_guidance is not None:
                theta_atom = batch.theta_h_t_for_guidance.to(self.device)
                N_atoms, K = theta_atom.shape
            else:
                if hasattr(batch, 'ligand_atom_feature_full'):
                    atom_idx = batch.ligand_atom_feature_full[:, 0].long().to(self.device)
                    K = int(atom_idx.max().item() + 1)
                    theta_atom = torch.nn.functional.one_hot(atom_idx.clamp(min=0), num_classes=K).float()
                else:
                    K = 1
                    theta_atom = torch.ones(1, K, device=self.device)

            theta_list = []
            for b in range(batch_size):
                mask = (batch_indices == b)
                if mask.sum() > 0:
                    batch_dist = theta_atom[mask]  # [N_b, K]

                    if batch_dist.size(0) >= self.seq_len:
                        seq_dist = batch_dist[:self.seq_len]  # [S, K]
                    else:
                        padding_size = self.seq_len - batch_dist.size(0)
                        padding = torch.zeros(padding_size, K, device=self.device)
                        seq_dist = torch.cat([batch_dist, padding], dim=0)  # [S, K]

                    seq = torch.zeros(self.seq_len, self.vocab_size, device=self.device)
                    seq[:, :K] = seq_dist
                    theta_list.append(seq)
                else:
                    uniform_theta = torch.ones(self.seq_len, self.vocab_size,
                                             device=self.device) / self.vocab_size
                    theta_list.append(uniform_theta)

            theta = torch.stack(theta_list, dim=0)  # [B, S, V]
            return theta

        except Exception as e:
            theta = torch.ones(batch_size, self.seq_len, self.vocab_size,
                             device=self.device) / self.vocab_size
            return theta
            self._feature_to_vocab_matrix = torch.randn(
                self.vocab_size, feature_dim, device=self.device
            ) * 0.1
        
        atom_logits = F.linear(ligand_features, self._feature_to_vocab_matrix)  # [N_atoms, vocab_size]
        
        theta_list = []
        for b in range(batch_size):
            mask = (batch.batch_ligand == b)
            batch_atoms = atom_logits[mask]  # [N_b, vocab_size]
            
            if batch_atoms.size(0) >= self.seq_len:
                batch_seq = batch_atoms[:self.seq_len]
            else:
                padding = torch.zeros(
                    self.seq_len - batch_atoms.size(0), 
                    self.vocab_size, 
                    device=batch_atoms.device
                )
                batch_seq = torch.cat([batch_atoms, padding], dim=0)
            
            batch_seq = F.softmax(batch_seq, dim=-1)
            theta_list.append(batch_seq)
        
        theta = torch.stack(theta_list, dim=0)  # [B, seq_len, vocab_size]
        
        return theta
    
    def get_guidance_parameters(self, batch, current_time: float,
                              target_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():

            theta_atoms, t, atom_mask = self.graph_to_guidance_input(batch, current_time)

            if theta_atoms is not None and self.guidance_model is not None:
                try:

                    pred_mu, pred_sigma = self.guidance_model(theta_atoms, t, atom_mask)


                    if pred_mu.dim() == 2 and pred_mu.size(1) == 2:
                        pass
                    else:
                        batch_size = theta_atoms.size(0)
                        pred_mu = torch.zeros(batch_size, 2, device=self.device)
                        pred_sigma = torch.ones(batch_size, 2, device=self.device) * 0.1


                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        self._validate_prediction_accuracy(pred_mu, target_conditions)

                    return pred_mu, pred_sigma

                except Exception as model_error:
                    batch_size = theta_atoms.size(0)
                    pred_mu = torch.zeros(batch_size, 2, device=self.device)
                    pred_sigma = torch.ones(batch_size, 2, device=self.device) * 0.1
                    return pred_mu, pred_sigma
            else:
                batch_size = target_conditions.size(0)
                device = target_conditions.device
                pred_mu = torch.zeros(batch_size, 2, device=device)
                pred_sigma = torch.ones(batch_size, 2, device=device) * 0.1
                return pred_mu, pred_sigma

    def _call_guidance_model_adaptive(self, theta: torch.Tensor, t: torch.Tensor,
                                    target_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        import inspect

        forward_signature = inspect.signature(self.guidance_model.forward)
        param_names = list(forward_signature.parameters.keys())
        num_params = len(param_names) - 1 

        try:
            if num_params >= 4:
                if 'pos_t' in param_names or 'pos' in param_names:
                    batch_size = theta.size(0)
                    pos_t = torch.zeros(batch_size, 3, device=theta.device)
                    batch_idx = torch.zeros(batch_size, dtype=torch.long, device=theta.device)
                    pred_mu, pred_sigma = self.guidance_model(theta, pos_t, t, batch_idx)
                else:
                    pred_mu, pred_sigma = self.guidance_model(theta, t, target_conditions)
            elif num_params >= 3:
                if 'target_conditions' in param_names:
                    pred_mu, pred_sigma = self.guidance_model(theta, t, target_conditions)
                else:
                    pred_mu, pred_sigma = self.guidance_model(theta, t, target_conditions)

            elif num_params == 2:
                pred_mu, pred_sigma = self.guidance_model(theta, t)

            else:
                raise ValueError(f"{num_params}")

            return pred_mu, pred_sigma

        except Exception as e:

            try:
                pred_mu, pred_sigma = self.guidance_model(theta, t, target_conditions)
                return pred_mu, pred_sigma
            except:
                pred_mu, pred_sigma = self.guidance_model(theta, t)
                return pred_mu, pred_sigma
    
    def apply_guidance(self, batch, current_time: float, target_conditions: torch.Tensor,
                      guidance_scale: float = 1.0) -> Dict[str, torch.Tensor]:

        try:
            if current_time < 0.3:
                effective_guidance_scale = guidance_scale * 0.6 
                guidance_mode = "structure_assembly"
            elif current_time < 0.7:
                progress = (current_time - 0.3) / 0.4
                effective_guidance_scale = guidance_scale * (0.6 + 0.4 * progress) 
                guidance_mode = "gradual_guidance"
            else:
                effective_guidance_scale = guidance_scale * 1.0 
                guidance_mode = "property_refinement"

            pred_mu, pred_sigma = self.get_guidance_parameters(batch, current_time, target_conditions)

            qed_mu, sa_mu = pred_mu[:, 0:1], pred_mu[:, 1:2]  # [B, 1]
            qed_sigma, sa_sigma = pred_sigma[:, 0:1], pred_sigma[:, 1:2]  # [B, 1]

            qed_target, sa_target = target_conditions[:, 0:1], target_conditions[:, 1:2]  # [B, 1]

            qed_sigma_effective = torch.clamp(qed_sigma, min=0.05, max=0.5)
            sa_sigma_effective = torch.clamp(sa_sigma, min=0.05, max=0.5)

            qed_direction = torch.sign(qed_target - qed_mu)  # [B, 1]
            sa_direction = torch.sign(sa_target - sa_mu)     # [B, 1]

            qed_error = torch.abs(qed_target - qed_mu)
            sa_error = torch.abs(sa_target - sa_mu)

            qed_normalized_error = qed_error / (qed_sigma_effective + 1e-6)
            sa_normalized_error = sa_error / (sa_sigma_effective + 1e-6)

            qed_guidance_strength = torch.tanh(qed_normalized_error)
            sa_guidance_strength = torch.tanh(sa_normalized_error)

            qed_correction = 1.0 + effective_guidance_scale * qed_guidance_strength * qed_direction
            sa_correction = 1.0 + effective_guidance_scale * sa_guidance_strength * sa_direction

            max_correction = 0.3
            qed_correction_clamped = torch.clamp(qed_correction, 1.0 - max_correction, 1.0 + max_correction)
            sa_correction_clamped = torch.clamp(sa_correction, 1.0 - max_correction, 1.0 + max_correction)

            device = pred_mu.device
            batch_size = pred_mu.size(0)

            qed_weight = 0.8
            sa_weight = 0.2
            guidance_probability = qed_weight * qed_correction_clamped + sa_weight * sa_correction_clamped

            guidance_probability = torch.clamp(guidance_probability, min=0.8, max=1.5)

            joint_guidance_strength = torch.max(qed_guidance_strength, sa_guidance_strength)

 
            if guidance_probability.numel() > 1:
                guidance_probability_scalar = guidance_probability[0, 0]
            else:
                guidance_probability_scalar = guidance_probability

            guidance_info = {
                'guidance_mu': pred_mu,
                'guidance_sigma': pred_sigma,
                'target_conditions': target_conditions,
                'guidance_probability': guidance_probability_scalar,
                'qed_error': qed_error,
                'sa_error': sa_error,
                'qed_guidance_strength': qed_guidance_strength,
                'sa_guidance_strength': sa_guidance_strength,
                'qed_direction': qed_direction,
                'sa_direction': sa_direction,
                'current_time': current_time,
                'guidance_scale': effective_guidance_scale,
                'guidance_mode': guidance_mode,
                'joint_guidance_strength': joint_guidance_strength
            }


            return guidance_info

        except Exception as e:
            try:
                batch_size = torch.unique(batch.batch_ligand).size(0)
            except:
                batch_size = 1
            return {
                'guidance_mu': torch.zeros(batch_size, device=self.device),
                'guidance_sigma': torch.ones(batch_size, device=self.device) * 0.01,
                'guidance_strength': torch.zeros(batch_size, device=self.device),
                'guidance_noise': torch.ones(batch_size, device=self.device) * 0.001,
                'target_conditions': target_conditions,
                'current_time': current_time,
                'guidance_scale': 0.0,
                'guidance_mode': 'fallback',
                'original_scale': guidance_scale
            }


def create_guidance_integrator(guidance_model_path: str, device: str = 'cuda',
                             max_atoms: int = 128, atom_feature_dim: int = 13,
                             condition_dim: int = 2) -> Optional[MolPilotGuidanceIntegrator]:

    try:
        if not os.path.exists(guidance_model_path):
            return None


        integrator = MolPilotGuidanceIntegrator(
            guidance_model_path=guidance_model_path,
            max_atoms=max_atoms,              # 128 
            atom_feature_dim=atom_feature_dim, # 13 
            condition_dim=condition_dim,       # 2 
            device=device
        )
        return integrator

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def integrate_guidance_to_bfn_sampling(bfn_model, guidance_integrator: MolPilotGuidanceIntegrator,
                                     batch, current_time: float, target_conditions: torch.Tensor,
                                     guidance_scale: float = 1.0):
    if guidance_integrator is None:
        return None
    
    guidance_info = guidance_integrator.apply_guidance(
        batch, current_time, target_conditions, guidance_scale
    )
    
    
    return guidance_info


def validate_guidance_model(model_path: str) -> bool:

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        model_type = checkpoint.get('model_type', '')
        if 'guidance' in model_type.lower() or 'conditional' in model_type.lower():
            return True

        if 'dynamics' in checkpoint or 'sbdd' in model_type.lower() or 'backbone' in model_type.lower():
            return False

        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'guidance_training') or hasattr(args, 'condition_supervision'):
                return True

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            guidance_layers = [k for k in state_dict.keys() if
                             'condition' in k.lower() or 'guidance' in k.lower()]
            if guidance_layers:
                return True
        return True

    except Exception as e:
        return False
