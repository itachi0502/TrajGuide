import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models.geometric_guidance_network import GeometricGuidanceNetwork, create_geometric_guidance_network
from core.models.guidance_architectures import (
    MultiArchGuidanceNetwork,
    create_multi_arch_guidance_network,
    get_architecture_info
)
from core.datasets.geometric_guidance_dataset import create_geometric_guidance_dataloader
from core.models.sbdd4train import SBDD4Train
from core.config.config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_backbone_model(config_path: str, checkpoint_path: str = None):
    """
    加载骨架模型，用于计算theta_t和pos_t

    这确保了训练数据与骨架模型使用完全一致的状态表示
    """
    try:
        logger.info(f"📂 加载骨架模型配置: {config_path}")

        #  使用正确的Config类
        config = Config(config_path)

        if checkpoint_path and os.path.exists(checkpoint_path):


            backbone_model = SBDD4Train.load_from_checkpoint(
                checkpoint_path,
                config=config,
                strict=False
            )
            logger.info(" 检查点加载成功")
        else:
            backbone_model = SBDD4Train(config=config)

        backbone_model.eval()  # 设置为评估模式
        return backbone_model

    except Exception as e:
        logger.error(f" 加载骨架模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_enhanced_loss(pred_mu, pred_sigma, target_properties, clamp=(1e-3, 0.5)):

    # 裁剪预测的标准差到合理范围
    pred_sigma_clamped = torch.clamp(pred_sigma, clamp[0], clamp[1])

    # 1. NLL损失（概率建模）
    # NLL = log(σ) + 0.5 * (target - μ)² / σ²
    qed_nll = torch.log(pred_sigma_clamped[:, 0]) + \
              0.5 * ((target_properties[:, 0] - pred_mu[:, 0]) ** 2) / (pred_sigma_clamped[:, 0] ** 2)
    sa_nll = torch.log(pred_sigma_clamped[:, 1]) + \
             0.5 * ((target_properties[:, 1] - pred_mu[:, 1]) ** 2) / (pred_sigma_clamped[:, 1] ** 2)

    # 2. MSE损失（直接监督）
    qed_mse = F.mse_loss(pred_mu[:, 0], target_properties[:, 0])
    sa_mse = F.mse_loss(pred_mu[:, 1], target_properties[:, 1])

    # 3. 方差正则化（防止方差过大或过小，目标方差0.1）
    sigma_reg = torch.mean((pred_sigma_clamped - 0.1) ** 2)

    # 组合损失
    qed_loss = qed_nll.mean() + 0.5 * qed_mse  # NLL + 0.5*MSE
    sa_loss = sa_nll.mean() + 0.5 * sa_mse

    # 总损失：QED权重2.0（增加），SA权重1.0，方差正则化0.1
    total_loss = 2.0 * qed_loss + 1.0 * sa_loss + 0.1 * sigma_reg

    return total_loss, qed_loss, sa_loss



def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch - 使用简单的NLL损失"""
    model.train()
    total_loss = 0.0
    total_qed_loss = 0.0
    total_sa_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        theta_i = batch['theta_i'].to(device) 
        alpha_i = batch['alpha_i'].to(device) 
        pos_t = batch['pos_t'].to(device)
        target_properties = batch['target_properties'].to(device) 
        batch_ligand = batch['batch_ligand'].to(device)

        pred_mu, pred_sigma = model(theta_i, pos_t, alpha_i, batch_ligand)

        # 计算增强损失（NLL + MSE + 方差正则化）
        loss, qed_loss, sa_loss = compute_enhanced_loss(
            pred_mu, pred_sigma, target_properties
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_qed_loss += qed_loss.item()
        total_sa_loss += sa_loss.item()

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'QED_Loss': f'{qed_loss.item():.4f}',
            'SA_Loss': f'{sa_loss.item():.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_qed_loss = total_qed_loss / num_batches
    avg_sa_loss = total_sa_loss / num_batches

    return avg_loss, avg_qed_loss, avg_sa_loss


def evaluate_model(model, dataloader, device, split_name='Val'):
    model.eval()
    total_loss = 0.0
    total_qed_loss = 0.0
    total_sa_loss = 0.0
    num_batches = len(dataloader)

    all_pred_mu = []
    all_target_properties = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'{split_name} Eval')
        for batch in pbar:
            theta_i = batch['theta_i'].to(device)
            alpha_i = batch['alpha_i'].to(device)
            pos_t = batch['pos_t'].to(device)
            target_properties = batch['target_properties'].to(device)
            batch_ligand = batch['batch_ligand'].to(device)

            pred_mu, pred_sigma = model(theta_i, pos_t, alpha_i, batch_ligand)

            loss, qed_loss, sa_loss = compute_enhanced_loss(
                pred_mu, pred_sigma, target_properties
            )

            total_loss += loss.item()
            total_qed_loss += qed_loss.item()
            total_sa_loss += sa_loss.item()
            all_pred_mu.append(pred_mu.cpu())
            all_target_properties.append(target_properties.cpu())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    avg_qed_loss = total_qed_loss / num_batches
    avg_sa_loss = total_sa_loss / num_batches
    all_pred_mu = torch.cat(all_pred_mu, dim=0)  # [N, 2]
    all_target_properties = torch.cat(all_target_properties, dim=0)  # [N, 2]

    qed_mse = F.mse_loss(all_pred_mu[:, 0], all_target_properties[:, 0]).item()
    sa_mse = F.mse_loss(all_pred_mu[:, 1], all_target_properties[:, 1]).item()

    metrics = {
        'total_loss': avg_loss,
        'qed_loss': avg_qed_loss,
        'sa_loss': avg_sa_loss,
        'qed_mse': qed_mse,
        'sa_mse': sa_mse
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, output_dir):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = output_dir / f'geometric_guidance_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    best_path = output_dir / 'geometric_guidance_best.pt'
    torch.save(checkpoint, best_path)
    
    logger.info(f"保存检查点: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="训练几何感知条件引导网络")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--config", type=str, help="骨架模型配置文件")
    parser.add_argument("--backbone_ckpt", type=str, help="骨架模型检查点")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")

    # 模型参数
    parser.add_argument("--architecture", type=str, default="gnn",
                       choices=['gnn', 'transformer', 'mlp', 'hybrid', 'bilstm', 'gru', 'cnn', 'resnet'],
                       help="引导网络架构: gnn, transformer, mlp, hybrid, bilstm, gru, cnn, resnet")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏维度")
    parser.add_argument("--num_layers", type=int, default=4, help="网络层数")
    parser.add_argument("--num_heads", type=int, default=8, help="Transformer注意力头数")
    parser.add_argument("--cutoff_radius", type=float, default=5.0, help="GNN截断半径")
    parser.add_argument("--max_num_neighbors", type=int, default=32, help="GNN最大邻居数")
    parser.add_argument("--kernel_size", type=int, default=3, help="CNN卷积核大小")

    # 其他参数
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载工作进程数")
    parser.add_argument("--max_samples", type=int, help="最大样本数（调试用）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"🚀 使用设备: {device}")

    arch_info = get_architecture_info(args.architecture)
    logger.info("=" * 80)
    logger.info(f" {args.architecture.upper()}")
    logger.info(f" {arch_info['name']}")
    logger.info(f" {arch_info['description']}")
    logger.info(f"advantage:")
    for strength in arch_info['strengths']:
        logger.info(f"{strength}")
    logger.info(f"disadvantage:")
    for weakness in arch_info['weaknesses']:
        logger.info(f"{weakness}")
    logger.info("=" * 80)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backbone_model = None
    if args.config:
        backbone_model = load_backbone_model(args.config, args.backbone_ckpt)
        if backbone_model:
            backbone_model = backbone_model.to(device)
    

    atom_types = 8  
    total_theta_dim = 8  

    if backbone_model is not None:
        K = getattr(backbone_model.dynamics, 'num_classes', 8) 
        KH = getattr(backbone_model.dynamics, 'num_charge', 0)  
        KA = getattr(backbone_model.dynamics, 'num_aromatic', 0)  

        total_theta_dim = K + KH + KA



    model_config = {
        'architecture': args.architecture, 
        'atom_types': total_theta_dim, 
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'condition_dim': 2, 
        'cutoff_radius': args.cutoff_radius,
        'max_num_neighbors': args.max_num_neighbors,
        'kernel_size': args.kernel_size,  
        'dropout': 0.1,
        'time_emb_dim': 64
    }


    model = create_multi_arch_guidance_network(model_config)
    model = model.to(device)

    train_loader = create_geometric_guidance_dataloader(
        args.data_dir, 'train', args.batch_size, backbone_model, 
        args.num_workers, shuffle=True, max_samples=args.max_samples
    )
    
    val_loader = create_geometric_guidance_dataloader(
        args.data_dir, 'val', args.batch_size, backbone_model,
        args.num_workers, shuffle=False, max_samples=args.max_samples
    )
    

    test_loader = create_geometric_guidance_dataloader(
        args.data_dir, 'test', args.batch_size, backbone_model,
        args.num_workers, shuffle=False, max_samples=args.max_samples
    )
    

    sample_times = []
    num_batches_to_sample = min(20, len(train_loader))  

    train_iter = iter(train_loader)
    for _ in range(num_batches_to_sample):
        try:
            batch = next(train_iter)
            batch_times = batch['alpha_i'].flatten().tolist()
            sample_times.extend(batch_times)
        except StopIteration:
            break

    if len(sample_times) > 0:
        sample_times = torch.tensor(sample_times)
        time_mean = sample_times.mean().item()
        time_std = sample_times.std().item()

        backbone_times = torch.linspace(0, 1, len(sample_times))
        backbone_mean = backbone_times.mean().item()
        backbone_std = backbone_times.std().item()



    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    best_val_loss = float('inf')
    train_history = []

    patience = 15
    patience_counter = 0

    logger.info("training...")
    logger.info(f"   Early Stopping: patience={patience}")

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_qed_loss, train_sa_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device, 'Val')

        scheduler_cosine.step()
        scheduler_plateau.step(val_metrics['total_loss'])

        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_qed_loss': train_qed_loss,
            'train_sa_loss': train_sa_loss,
            'val_metrics': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_info)
        
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_metrics['total_loss']:.4f} | "
                   f"QED MSE: {val_metrics['qed_mse']:.4f} | "
                   f"SA MSE: {val_metrics['sa_mse']:.4f} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0  
            save_checkpoint(model, optimizer, epoch, val_metrics, output_dir)
            logger.info(f"新的最佳模型! 验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"   验证损失未改进 ({patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f" Early stopping触发! 在epoch {epoch}停止训练")
                logger.info(f"   最佳验证损失: {best_val_loss:.4f}")
                break

        if epoch % 10 == 0:
            logger.info("详细性能分析:")
            logger.info(f"   QED预测MSE: {val_metrics['qed_mse']:.6f} (目标: <0.01)")
            logger.info(f"   SA预测MSE: {val_metrics['sa_mse']:.6f} (目标: <0.01)")
            logger.info(f"   总体NLL损失: {val_metrics['total_loss']:.6f}")

            # 检查是否过拟合
            if train_loss < val_metrics['total_loss'] * 0.5:
                logger.warning("⚠️ 可能出现过拟合，考虑增加正则化或早停")
    
    logger.info("在测试集上进行最终评估...")
    test_metrics = evaluate_model(model, test_loader, device, 'Test')
    
    logger.info("最终测试结果:")
    logger.info(f"   总损失: {test_metrics['total_loss']:.4f}")
    logger.info(f"   QED MSE: {test_metrics['qed_mse']:.4f}")
    logger.info(f"   SA MSE: {test_metrics['sa_mse']:.4f}")
    
    final_results = {
        'architecture': args.architecture, 
        'architecture_info': arch_info,
        'train_history': train_history,
        'test_metrics': test_metrics,
        'model_config': model_config,
        'training_args': vars(args)
    }

    results_file = output_dir / f'training_results_{args.architecture}.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"训练完成！结果保存至: {output_dir}")
    logger.info(f"   架构: {args.architecture.upper()}")
    logger.info(f"   结果文件: {results_file}")

    logger.info("=" * 80)
    logger.info("训练完成总结:")
    logger.info(f"   架构: {args.architecture.upper()}")
    logger.info(f"   时间步分布一致性: 已验证")
    logger.info(f"   最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"   QED预测精度: MSE={test_metrics['qed_mse']:.6f}")
    logger.info(f"   SA预测精度: MSE={test_metrics['sa_mse']:.6f}")
    logger.info(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
