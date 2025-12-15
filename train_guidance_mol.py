#!/usr/bin/env python3


import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import wandb
from datetime import datetime

print(f" {torch.__version__}")

compatibility_issues = []
if not hasattr(F, 'softplus'):
    compatibility_issues.append("F.softplus")
if not hasattr(torch, 'square'):
    compatibility_issues.append("torch.square")


def safe_softplus(x):
    try:
        return F.softplus(x)
    except AttributeError:
        return torch.log(1.0 + torch.exp(torch.clamp(x, max=20)))  

def safe_square(x):
    try:
        return torch.square(x)
    except AttributeError:
        return x * x

# MolPilot核心模块
from core.config.config import Config, Struct
from core.models.sbdd4train import SBDD4Train
from core.datasets import get_dataset
import core.utils.transforms as trans
from torch_geometric.transforms import Compose
from core.utils.condition_transforms import create_condition_aware_transform


class MolPilotGuidanceNetwork(nn.Module):

    def __init__(self, vocab_size: int = 263, seq_len: int = 64,
                 hidden_dim: int = 256, num_layers: int = 6, num_heads: int = 8,
                 condition_dim: int = 4, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        self.theta_proj = nn.Linear(vocab_size, hidden_dim)
        self.theta_conv1d = nn.Conv1d(vocab_size, hidden_dim, kernel_size=3, padding=1)
        self.theta_pool = nn.AdaptiveAvgPool1d(1)

        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )


        self.condition_emb = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


        self.pos_emb = nn.Parameter(torch.randn(seq_len, hidden_dim))
        

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)


        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # theta + time + condition
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  
            nn.Sigmoid() 
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2), 
            nn.Softplus()
        )


        self.uncertainty_calibration = nn.Parameter(torch.ones(2) * 0.1)

        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        

        try:
            test_tensor = torch.tensor([1.0])
            _ = safe_softplus(test_tensor)
            _ = safe_square(test_tensor)
        except Exception as e:
            raise e
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, theta: torch.Tensor, t: torch.Tensor,
                target_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, vocab_size = theta.shape

        if seq_len != self.seq_len:
            
            if seq_len > self.seq_len:
                theta = theta[:, :self.seq_len, :]  
            else:
                padding = torch.zeros(batch_size, self.seq_len - seq_len, vocab_size,
                                    device=theta.device, dtype=theta.dtype)
                theta = torch.cat([theta, padding], dim=1)
            seq_len = self.seq_len

        if vocab_size != self.vocab_size:
            raise ValueError(f"{vocab_size} vs {self.vocab_size}")

        if t.dim() == 0:
            t = t.unsqueeze(0)
            batch_size_t = 1
        elif t.dim() == 1:
            batch_size_t = t.size(0)
        else:
            raise ValueError(f"{t.dim()}D, {t.shape}")

        if batch_size_t == 1 and batch_size > 1:
            t = t.expand(batch_size)
        elif batch_size_t != batch_size:
            raise ValueError(f"{batch_size_t} vs {batch_size}")

        if target_conditions.dim() == 0:
            target_conditions = target_conditions.unsqueeze(0).repeat(1, self.condition_dim)
            if batch_size > 1:
                target_conditions = target_conditions.expand(batch_size, -1)
        elif target_conditions.dim() == 1:
            if target_conditions.size(0) == self.condition_dim:
                # [4] -> [1, 4]
                target_conditions = target_conditions.unsqueeze(0)
                if batch_size > 1:
                    target_conditions = target_conditions.expand(batch_size, -1)
            elif target_conditions.size(0) == batch_size * self.condition_dim:
                # [B*4] -> [B, 4]
                target_conditions = target_conditions.view(batch_size, self.condition_dim)
            else:
                raise ValueError(f"{target_conditions.size(0)}, "
                               f"{self.condition_dim} or {batch_size * self.condition_dim}")
        elif target_conditions.dim() == 2:
            batch_size_cond, condition_dim = target_conditions.shape
            if condition_dim != self.condition_dim:
                raise ValueError(f"{condition_dim} vs {self.condition_dim}")
            if batch_size_cond == 1 and batch_size > 1:
                target_conditions = target_conditions.expand(batch_size, -1)
            elif batch_size_cond != batch_size:
                raise ValueError(f"{batch_size_cond} vs {batch_size}")
        else:
            raise ValueError(f"{target_conditions.dim()}D, {target_conditions.shape}")

        assert target_conditions.shape == (batch_size, self.condition_dim), \
            f"target_conditions{target_conditions.shape}, ({batch_size}, {self.condition_dim})"
        
 
        theta_emb = self.theta_embedding(theta)  # [B, S, H]
        
        time_emb = self.time_embedding(t.unsqueeze(-1))  # [B, H]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, H]
        
        pos_emb = self.position_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, H]
        
        combined_emb = theta_emb + time_emb + pos_emb  # [B, S, H]
        
        transformer_out = self.transformer(combined_emb)  # [B, S, H]
        
        pooled_out = transformer_out.mean(dim=1)  # [B, H]
        
        condition_emb = self.condition_encoder(target_conditions)  # [B, H]
        
        fused_input = torch.cat([pooled_out, condition_emb], dim=-1)  # [B, 2H]
        guidance_params = self.fusion_layer(fused_input)  # [B, 2]
        

        mu = guidance_params[:, 0:1]  # [B, 1] 
        sigma = safe_softplus(guidance_params[:, 1:2]) + 1e-6  # [B, 1]
        
        mu_expanded = mu.expand(-1, 2)      # [B, 2]
        sigma_expanded = sigma.expand(-1, 2)  # [B, 2]
        
        return mu_expanded, sigma_expanded


class MolPilotGuidanceDataset(Dataset):

    
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            theta = sample['theta'].float()

            t_value = sample['t_value'].float()
            if t_value.dim() != 0:
                if t_value.dim() == 1 and t_value.size(0) == 1:
                    t_value = t_value.squeeze(0)  # [1] -> scalar
                else:
                    raise ValueError(f"t_value.shape")

            mu_label = sample['mu_label'].float()
            if mu_label.dim() == 0:
                mu_label = mu_label.unsqueeze(0).repeat(2)  # scalar -> [2]
            elif mu_label.dim() == 1:
                if mu_label.size(0) == 1:
                    mu_label = mu_label.repeat(2)  # [1] -> [2]
                elif mu_label.size(0) != 2:
                    mu_label = torch.zeros(2)  
            else:
                mu_label = torch.zeros(2)  

            sigma_label = sample['sigma_label'].float()
            if sigma_label.dim() == 0:
                sigma_label = sigma_label.unsqueeze(0).repeat(2)  # scalar -> [2]
            elif sigma_label.dim() == 1:
                if sigma_label.size(0) == 1:
                    sigma_label = sigma_label.repeat(2)  # [1] -> [2]
                elif sigma_label.size(0) != 2:
                    sigma_label = torch.ones(2) * 0.1 
            else:
                sigma_label = torch.ones(2) * 0.1 

            target_conditions = sample.get('original_conditions', None)
            if target_conditions is None:
                target_conditions = torch.zeros(4)
            else:
                target_conditions = target_conditions.float()

                if target_conditions.dim() == 0:
                    target_conditions = target_conditions.unsqueeze(0).repeat(4)
                elif target_conditions.dim() == 1:
                    if target_conditions.size(0) == 1:

                        target_conditions = target_conditions.repeat(4)
                    elif target_conditions.size(0) != 4:
                        target_conditions = torch.zeros(4)
                elif target_conditions.dim() == 2:
                    if target_conditions.size(0) == 1:
                        target_conditions = target_conditions.squeeze(0)  # [1, 4] -> [4]
                    else:
                        target_conditions = torch.zeros(4)
                else:
                    target_conditions = torch.zeros(2)

                if target_conditions.size(0) != 2:
                    target_conditions = torch.zeros(2)

            return {
                'theta': theta,                    # [S, V]
                't_value': t_value,                # scalar
                'mu_label': mu_label,              # [2]
                'sigma_label': sigma_label,        # [2]
                'target_conditions': target_conditions  # [2]
            }

        except Exception as e:
            return {
                'theta': torch.randn(64, 263),    
                't_value': torch.tensor(0.5),      
                'mu_label': torch.zeros(2),        
                'sigma_label': torch.ones(2) * 0.1,  
                'target_conditions': torch.zeros(2)  
            }


def kl_divergence_loss(mu1: torch.Tensor, sigma1: torch.Tensor, 
                      mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:

    sigma1 = torch.clamp(sigma1, min=1e-6)
    sigma2 = torch.clamp(sigma2, min=1e-6)
    
    log_term = torch.log(sigma2) - torch.log(sigma1)
    mean_diff_term = safe_square(mu1 - mu2)
    sigma_ratio_term = safe_square(sigma1) / safe_square(sigma2)

    kl_div = log_term + 0.5 * (sigma_ratio_term + mean_diff_term / safe_square(sigma2) - 1.0)
    
    return kl_div


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def custom_collate_fn(batch):

    try:
        theta_list = []
        t_value_list = []
        mu_label_list = []
        sigma_label_list = []
        target_conditions_list = []

        for sample in batch:
            theta = sample['theta']
            if theta.dim() != 2 or theta.shape != (64, 263):
                continue

            t_value = sample['t_value']
            if t_value.dim() != 0:
                if t_value.dim() == 1 and t_value.size(0) == 1:
                    t_value = t_value.squeeze(0)
                else:
                    continue

            mu_label = sample['mu_label']
            if mu_label.shape != (2,):
                continue

            sigma_label = sample['sigma_label']
            if sigma_label.shape != (2,):
                continue

            target_conditions = sample['target_conditions']
            if target_conditions.shape != (4,):
                continue

            theta_list.append(theta)
            t_value_list.append(t_value)
            mu_label_list.append(mu_label)
            sigma_label_list.append(sigma_label)
            target_conditions_list.append(target_conditions)

        if len(theta_list) == 0:
            return None

        batch_data = {
            'theta': torch.stack(theta_list, dim=0),                    # [B, S, V]
            't_value': torch.stack(t_value_list, dim=0),                # [B]
            'mu_label': torch.stack(mu_label_list, dim=0),              # [B, 2]
            'sigma_label': torch.stack(sigma_label_list, dim=0),        # [B, 2]
            'target_conditions': torch.stack(target_conditions_list, dim=0)  # [B, 4]
        }

        return batch_data

    except Exception as e:
        return None


def create_dataloaders(train_data_path: str, val_data_path: str,
                      batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:

    train_dataset = MolPilotGuidanceDataset(train_data_path)
    val_dataset = MolPilotGuidanceDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn  
    )

    return {'train': train_loader, 'val': val_loader}


def validate_model(model: MolPilotGuidanceNetwork, val_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_mu_loss = 0.0
    total_sigma_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue

            try:

                theta = batch['theta'].to(device)          # [B, S, V]
                t_value = batch['t_value'].to(device)      # [B]
                mu_label = batch['mu_label'].to(device)    # [B, 2]
                sigma_label = batch['sigma_label'].to(device)  # [B, 2]
                target_conditions = batch['target_conditions'].to(device)  # [B, 4]

                batch_size = theta.size(0)
                if t_value.size(0) != batch_size:
                    if t_value.size(0) == 1:
                        t_value = t_value.expand(batch_size)
                    else:
                        continue  

                if target_conditions.size(0) != batch_size:
                    if target_conditions.size(0) == 1:
                        target_conditions = target_conditions.expand(batch_size, -1)
                    else:
                        continue 

                pred_mu, pred_sigma = model(theta, t_value, target_conditions)

                if pred_mu.shape != mu_label.shape or pred_sigma.shape != sigma_label.shape:
                    continue 

                kl_loss = kl_divergence_loss(mu_label, sigma_label, pred_mu, pred_sigma)
                loss = mean_flat(kl_loss).mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    continue 

                mu_loss = F.mse_loss(pred_mu, mu_label)
                sigma_loss = F.mse_loss(pred_sigma, sigma_label)

                total_loss += loss.item()
                total_mu_loss += mu_loss.item()
                total_sigma_loss += sigma_loss.item()
                num_batches += 1

            except Exception as e:
                continue 
    
    avg_loss = total_loss / num_batches
    avg_mu_loss = total_mu_loss / num_batches
    avg_sigma_loss = total_sigma_loss / num_batches
    
    return avg_loss, avg_mu_loss, avg_sigma_loss


def train_guidance_model(model: MolPilotGuidanceNetwork, dataloaders: Dict[str, DataLoader],
                        optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
                        num_epochs: int, device: torch.device,
                        save_dir: str, use_wandb: bool = False) -> MolPilotGuidanceNetwork:


    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        model.train()
        epoch_train_loss = 0.0
        epoch_mu_loss = 0.0
        epoch_sigma_loss = 0.0
        num_train_batches = 0

        train_pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(train_pbar):
            if batch is None:
                continue

            try:
                theta = batch['theta'].to(device)          # [B, S, V]
                t_value = batch['t_value'].to(device)      # [B]
                mu_label = batch['mu_label'].to(device)    # [B, 2]
                sigma_label = batch['sigma_label'].to(device)  # [B, 2]
                target_conditions = batch['target_conditions'].to(device)  # [B, 4]

                batch_size = theta.size(0)
                if t_value.size(0) != batch_size:
                    if t_value.size(0) == 1:
                        t_value = t_value.expand(batch_size)
                    else:
                        continue

                if target_conditions.size(0) != batch_size:

                    if target_conditions.size(0) == 1:
                        target_conditions = target_conditions.expand(batch_size, -1)
                    else:
                        continue 

                try:
                    pred_mu, pred_sigma = model(theta, t_value, target_conditions)
                except Exception as model_error:
                    raise model_error

                kl_loss = kl_divergence_loss(mu_label, sigma_label, pred_mu, pred_sigma)
                loss = mean_flat(kl_loss).mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                mu_loss = F.mse_loss(pred_mu, mu_label)
                sigma_loss = F.mse_loss(pred_sigma, sigma_label)

                epoch_train_loss += loss.item()
                epoch_mu_loss += mu_loss.item()
                epoch_sigma_loss += sigma_loss.item()
                num_train_batches += 1

                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'μ_Loss': f'{mu_loss.item():.4f}',
                    'σ_Loss': f'{sigma_loss.item():.4f}'
                })

                if use_wandb and batch_idx % 100 == 0:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/batch_mu_loss': mu_loss.item(),
                        'train/batch_sigma_loss': sigma_loss.item(),
                        'train/step': epoch * len(dataloaders['train']) + batch_idx
                    })

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue 

        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_mu_loss = epoch_mu_loss / num_train_batches
        avg_train_sigma_loss = epoch_sigma_loss / num_train_batches

        val_loss, val_mu_loss, val_sigma_loss = validate_model(model, dataloaders['val'], device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if use_wandb:
            wandb.log({
                'train/epoch_loss': avg_train_loss,
                'train/epoch_mu_loss': avg_train_mu_loss,
                'train/epoch_sigma_loss': avg_train_sigma_loss,
                'val/epoch_loss': val_loss,
                'val/epoch_mu_loss': val_mu_loss,
                'val/epoch_sigma_loss': val_sigma_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
                'config': {
                    'vocab_size': model.vocab_size,
                    'seq_len': model.seq_len,
                    'hidden_dim': model.hidden_dim,
                    'condition_dim': model.condition_dim
                }
            }, best_model_path)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
            }, checkpoint_path)


    best_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    return model


def main():
    parser = argparse.ArgumentParser(description='condition guidance network')

    # 数据参数
    parser.add_argument('--train_data', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True,
                       help='验证数据路径')

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=263,
                       help='词汇表大小V（离散候选集合大小）')
    parser.add_argument('--seq_len', type=int, default=64,
                       help='序列长度S（固定序列长度）')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏维度')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--condition_dim', type=int, default=4,
                       help='条件维度（QED, SA, MW, LogP）')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'step', 'none'],
                       help='学习率调度器')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='训练设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--save_dir', type=str, default='./logs/molpilot_guidance',
                       help='模型保存目录')
    parser.add_argument('--exp_name', type=str, default='molpilot_guidance',
                       help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    # WandB参数
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用WandB记录')
    parser.add_argument('--wandb_project', type=str, default='molpilot-guidance',
                       help='WandB项目名称')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 初始化WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args)
        )

    dataloaders = create_dataloaders(
        args.train_data, args.val_data,
        args.batch_size, args.num_workers
    )

    # 创建模型
    model = MolPilotGuidanceNetwork(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        condition_dim=args.condition_dim,
        dropout=args.dropout
    ).to(device)

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 创建学习率调度器
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None

    trained_model = train_guidance_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
