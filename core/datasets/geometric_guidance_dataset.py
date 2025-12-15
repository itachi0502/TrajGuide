import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GeometricGuidanceDataset(Dataset):

    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        backbone_model=None,
        max_samples: Optional[int] = None,
        cache_theta: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.backbone_model = backbone_model
        self.cache_theta = cache_theta
        self.theta_cache = {}
        
        self._load_data(max_samples)
        
    
    def _load_data(self, max_samples: Optional[int] = None):
        data_file = self.data_dir / f"{self.split}_condition_data.pkl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"data not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
    
    def __len__(self):
        return len(self.data)
    
    def _compute_theta_from_backbone(self, mol_data: Data, t: float) -> torch.Tensor:

        if hasattr(mol_data, 'ligand_element'):
            ligand_element_raw = mol_data.ligand_element
        elif hasattr(mol_data, 'ligand_atom_feature_full'):
            ligand_element_raw = mol_data.ligand_atom_feature_full[:, 0].long()
        else:
            raise AttributeError(f"conditions: {list(mol_data.keys())}")

        from core.utils.transforms import MAP_ATOM_TYPE_ONLY_TO_INDEX

        ligand_element = torch.zeros_like(ligand_element_raw)
        for i, atomic_num in enumerate(ligand_element_raw):
            atomic_num_item = atomic_num.item()
            if atomic_num_item in MAP_ATOM_TYPE_ONLY_TO_INDEX:
                ligand_element[i] = MAP_ATOM_TYPE_ONLY_TO_INDEX[atomic_num_item]
            else:

                ligand_element[i] = 1
                if not hasattr(self, '_unknown_atoms_warned'):
                    self._unknown_atoms_warned = True

        num_atoms = ligand_element.size(0)

        if self.backbone_model is None:
            K = max(ligand_element.max().item() + 1, 100)
            theta_clean = F.one_hot(ligand_element.long(), num_classes=K).float()
            noise_level = t
            uniform_dist = torch.ones_like(theta_clean) / K
            theta_t = (1 - noise_level) * theta_clean + noise_level * uniform_dist
            theta_t = theta_t / theta_t.sum(dim=-1, keepdim=True)
            return theta_t
        
        cache_key = (id(mol_data), t) if self.cache_theta else None
        if cache_key and cache_key in self.theta_cache:
            return self.theta_cache[cache_key]
        
        with torch.no_grad():
            device = next(self.backbone_model.parameters()).device

            ligand_element_device = ligand_element.to(device)
            batch_ligand = torch.zeros(ligand_element_device.size(0), dtype=torch.long, device=device)

            batch_size = 1
            t_batch = torch.full((batch_size,), t, device=device)

            K = getattr(self.backbone_model.dynamics, 'num_classes', 100)

            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
            max_element = ligand_element_device.max().item()
            min_element = ligand_element_device.min().item()


            ligand_type_onehot = F.one_hot(ligand_element_device.long(), num_classes=K).float()

            t_atom_level = t_batch[batch_ligand]  # [35]
            if not hasattr(self, '_onehot_debug_printed'):
                beta_test = self.backbone_model.dynamics.beta1 * (t_atom_level**2)
                self._onehot_debug_printed = True


            theta_atom = self.backbone_model.dynamics.discrete_var_bayesian_update(
                t_atom_level,  # [N]
                beta1=self.backbone_model.dynamics.beta1,
                x=ligand_type_onehot,
                K=K
            )  # [N, K]

            theta_t = theta_atom
            KH = getattr(self.backbone_model.dynamics, 'num_charge', 0)
            if KH > 0:
                charge_probs = torch.tensor([0.2, 0.7, 0.1], device=device)
                if KH == 3:
                    charge_indices = torch.multinomial(charge_probs, num_atoms, replacement=True)
                else:
                    charge_indices = torch.randint(0, KH, (num_atoms,), device=device)

                ligand_charge_onehot = torch.zeros(num_atoms, KH, device=device)
                ligand_charge_onehot.scatter_(1, charge_indices.unsqueeze(1), 1.0)

                theta_charge = self.backbone_model.dynamics.discrete_var_bayesian_update(
                    t_atom_level,
                    beta1=getattr(self.backbone_model.dynamics, 'beta1_charge', self.backbone_model.dynamics.beta1),
                    x=ligand_charge_onehot,
                    K=KH
                )  # [N, KH]

                theta_t = torch.cat([theta_t, theta_charge], dim=-1)

            KA = getattr(self.backbone_model.dynamics, 'num_aromatic', 0)
            if KA > 0:
                aromatic_probs = torch.tensor([0.8, 0.2], device=device) 
                if KA == 2:
                    aromatic_indices = torch.multinomial(aromatic_probs, num_atoms, replacement=True)
                else:
                    aromatic_indices = torch.randint(0, KA, (num_atoms,), device=device)

                ligand_aromatic_onehot = torch.zeros(num_atoms, KA, device=device)
                ligand_aromatic_onehot.scatter_(1, aromatic_indices.unsqueeze(1), 1.0)

                theta_aromatic = self.backbone_model.dynamics.discrete_var_bayesian_update(
                    t_atom_level,
                    beta1=getattr(self.backbone_model.dynamics, 'beta1_aromatic', self.backbone_model.dynamics.beta1),
                    x=ligand_aromatic_onehot,
                    K=KA
                )  # [N, KA]

                theta_t = torch.cat([theta_t, theta_aromatic], dim=-1)

        if cache_key:
            self.theta_cache[cache_key] = theta_t.cpu()
        
        return theta_t.cpu()
    
    def _compute_pos_from_backbone(self, mol_data: Data, t: float) -> torch.Tensor:
        if self.backbone_model is None:
            pos = mol_data.ligand_pos
            noise_scale = t * 0.1
            noise = torch.randn_like(pos) * noise_scale
            return pos + noise
        
        with torch.no_grad():
            device = next(self.backbone_model.parameters()).device
            ligand_pos = mol_data.ligand_pos.to(device)
            batch_ligand = torch.zeros(ligand_pos.size(0), dtype=torch.long, device=device)

            batch_size = 1 
            t_batch = torch.full((batch_size,), t, device=device)

            t_atom_level = t_batch[batch_ligand]  # [N]

            if not hasattr(self, '_pos_debug_printed'):
                gamma_test = 1 - torch.pow(self.backbone_model.dynamics.sigma1_coord, 2 * t_atom_level)
                self._pos_debug_printed = True

            mu_pos_t, gamma_coord = self.backbone_model.dynamics.continuous_var_bayesian_update(
                t_atom_level,  # [N] 
                sigma1=self.backbone_model.dynamics.sigma1_coord,
                x=ligand_pos
            )
        
        return mu_pos_t.cpu()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        mol_data_dict = self.data[idx]
        mol_data = mol_data_dict['molecule'] 
        target_properties = torch.tensor(mol_data_dict['conditions'], dtype=torch.float32)

        
        t_i = torch.rand(1).item()  

        theta_i = self._compute_theta_from_backbone(mol_data, t_i)
        pos_t = self._compute_pos_from_backbone(mol_data, t_i)

        alpha_i = torch.tensor([1.0 - t_i], dtype=torch.float32)

        sender_weight = self._compute_sender_weight(theta_i, mol_data, alpha_i)

        num_atoms = theta_i.size(0)
        batch_ligand = torch.zeros(num_atoms, dtype=torch.long)

        return {
            'theta_i': theta_i,
            'alpha_i': alpha_i,
            'pos_t': pos_t,
            'target_properties': target_properties,
            'batch_ligand': batch_ligand,
            'sender_weight': sender_weight
        }

    def _compute_sender_weight(self, theta_i: torch.Tensor, mol_data: Data, alpha_i: torch.Tensor) -> torch.Tensor:

        try:

            max_probs = torch.max(theta_i, dim=-1)[0]  # [N]
            avg_certainty = torch.mean(max_probs)  

            entropy = -torch.sum(theta_i * torch.log(torch.clamp(theta_i, min=1e-8)), dim=-1)  # [N]
            avg_entropy = torch.mean(entropy) 


            certainty_factor = avg_certainty  # [0, 1]
            noise_factor = alpha_i.item()     # [0, 1]
            entropy_factor = torch.exp(-avg_entropy) 

            sender_weight = certainty_factor * noise_factor * entropy_factor

            sender_weight = torch.clamp(sender_weight, min=0.1, max=2.0)

            return sender_weight.unsqueeze(0)  # [1]

        except Exception as e:
            return torch.tensor([1.0], dtype=torch.float32)


def geometric_guidance_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)

    theta_i_list = []
    alpha_i_list = []
    pos_t_list = []
    target_properties_list = []
    sender_weight_list = []
    batch_indices = []

    atom_offset = 0
    for i, sample in enumerate(batch):
        theta_i_list.append(sample['theta_i'])
        alpha_i_list.append(sample['alpha_i'])
        pos_t_list.append(sample['pos_t'])
        target_properties_list.append(sample['target_properties'])
        sender_weight_list.append(sample['sender_weight'])

        num_atoms = sample['theta_i'].size(0)
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))
        atom_offset += num_atoms

    batched_data = {
        'theta_i': torch.cat(theta_i_list, dim=0),      # [N_total, K]
        'alpha_i': torch.stack(alpha_i_list, dim=0),    # [B, 1]
        'pos_t': torch.cat(pos_t_list, dim=0),          # [N_total, 3]
        'target_properties': torch.stack(target_properties_list, dim=0), # [B, 2]
        'sender_weight': torch.stack(sender_weight_list, dim=0),  # [B, 1]
        'batch_ligand': torch.cat(batch_indices, dim=0),   # [N_total]
    }

    return batched_data


def create_geometric_guidance_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    backbone_model=None,
    num_workers: int = 0,
    shuffle: bool = None,
    max_samples: Optional[int] = None
) -> DataLoader:
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = GeometricGuidanceDataset(
        data_dir=data_dir,
        split=split,
        backbone_model=backbone_model,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=geometric_guidance_collate_fn,
        pin_memory=True
    )
    
    return dataloader


def test_geometric_guidance_dataset():
    import tempfile
    import os
    
    test_data = []
    for i in range(10):
        mol_data = type('MockData', (), {
            'ligand_atom_type': torch.randint(0, 10, (5,)),
            'ligand_pos': torch.randn(5, 3)
        })()
        
        test_data.append({
            'molecule': mol_data,
            'conditions': [0.5 + 0.1 * i, 0.6 + 0.05 * i],
            'original_index': i
        })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "train_condition_data.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        dataset = GeometricGuidanceDataset(temp_dir, split='train')

        sample = dataset[0]
        
        dataloader = create_geometric_guidance_dataloader(
            temp_dir, split='train', batch_size=3
        )
        
        batch = next(iter(dataloader))


if __name__ == "__main__":
    test_geometric_guidance_dataset()
