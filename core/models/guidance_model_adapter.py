"""
SOTA级条件引导模型适配器
将训练好的ConditionalGuidanceClassifier适配为推理脚本期望的MolPilotGuidanceNetwork接口

作者：SOTA级优化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import os


class MolPilotGuidanceNetworkAdapter(nn.Module):
    """
    SOTA级引导模型适配器
    
    将ConditionalGuidanceClassifier包装成MolPilotGuidanceNetwork的接口
    确保与推理脚本完全兼容
    """
    
    def __init__(self, vocab_size: int = 263, seq_len: int = 64, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8, condition_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # 存储实际的引导模型
        self.actual_model = None
        
        # 创建兼容的属性名（推理脚本期望的）
        self.position_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.theta_embedding = nn.Linear(vocab_size, hidden_dim)
        self.time_embedding = nn.Linear(1, hidden_dim)
        
        # Transformer层（兼容性）
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 输出μ和σ
        )
        
    def load_trained_model(self, checkpoint_path: str, device: str = 'cuda'):
        try:
            
            from train_guidance_mol_fixed import ConditionalGuidanceClassifier
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'args' in checkpoint:
                args = checkpoint['args']
                vocab_size = args.vocab_size
                seq_len = args.seq_len
                hidden_dim = args.hidden_dim
                num_layers = args.num_layers
                num_heads = args.num_heads
                condition_dim = args.condition_dim
                dropout = args.dropout
            else:
                vocab_size = self.vocab_size
                seq_len = self.seq_len
                hidden_dim = self.hidden_dim
                num_layers = 4
                num_heads = 8
                condition_dim = self.condition_dim
                dropout = 0.0
            
            self.actual_model = ConditionalGuidanceClassifier(
                vocab_size=vocab_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                condition_dim=condition_dim,
                dropout=dropout
            ).to(device)
            
            self.actual_model.load_state_dict(checkpoint['model_state_dict'])
            self.actual_model.eval()
            

            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, theta: torch.Tensor, t: torch.Tensor, 
                target_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.actual_model is not None:
            # 使用训练好的模型
            with torch.no_grad():
                mu, sigma = self.actual_model(theta, t, target_conditions)
                return mu, sigma
        else:
            # 备选方案：使用适配器的内置实现
            return self._fallback_forward(theta, t, target_conditions)
    
    def _fallback_forward(self, theta: torch.Tensor, t: torch.Tensor,
                         target_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, vocab_size = theta.shape
        
        # 1. 输入嵌入
        theta_emb = self.theta_embedding(theta)  # [B, S, H]
        
        # 2. 时间嵌入
        time_emb = self.time_embedding(t.unsqueeze(-1))  # [B, H]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, H]
        
        # 3. 位置嵌入
        pos_emb = self.position_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, H]
        
        # 4. 组合嵌入
        combined_emb = theta_emb + time_emb + pos_emb  # [B, S, H]
        
        # 5. Transformer编码
        x = combined_emb
        for layer in self.transformer:
            x = layer(x)  # [B, S, H]
        
        # 6. 全局池化
        pooled_out = x.mean(dim=1)  # [B, H]
        
        # 7. 条件编码
        condition_emb = self.condition_encoder(target_conditions)  # [B, H]
        
        # 8. 融合
        fused_input = torch.cat([pooled_out, condition_emb], dim=-1)  # [B, 2H]
        output = self.fusion_layer(fused_input)  # [B, 2]
        
        # 9. 分离μ和σ
        mu = torch.sigmoid(output)  # [B, 2] 确保在[0,1]范围
        sigma = torch.softplus(output) * 0.1 + 1e-6  # [B, 2] 确保为正且合理
        
        return mu, sigma


def create_adapted_guidance_model(checkpoint_path: str, device: str = 'cuda',
                                vocab_size: int = 263, seq_len: int = 64,
                                hidden_dim: int = 256, num_layers: int = 6,
                                num_heads: int = 8, condition_dim: int = 4) -> Optional[MolPilotGuidanceNetworkAdapter]:

    try:
        # 创建适配器
        adapter = MolPilotGuidanceNetworkAdapter(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            condition_dim=condition_dim,
            dropout=0.0  # 推理时不使用dropout
        ).to(device)
        
        # 加载训练好的模型
        success = adapter.load_trained_model(checkpoint_path, device)
        
        if success:
            return adapter
        else:
            return None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


MolPilotGuidanceNetwork = MolPilotGuidanceNetworkAdapter
