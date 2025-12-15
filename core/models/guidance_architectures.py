"""
支持的架构：
1. GNN (Geometric Graph Neural Network) - 当前使用的架构
2. Transformer (Self-Attention based) - 基于自注意力机制
3. MLP (Multi-Layer Perceptron) - 简单的全连接网络baseline
4. Hybrid (GNN + Transformer) - 混合架构
5. BiLSTM (Bidirectional LSTM) - 双向长短期记忆网络
6. GRU (Gated Recurrent Unit) - 门控循环单元
7. CNN (Convolutional Neural Network) - 卷积神经网络
8. ResNet (Residual Network) - 残差网络

用于消融实验：分析不同架构对引导效果的影响

Author: MolCRAFT Team
Date: 2025-10-14
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import radius_graph, global_mean_pool, global_max_pool
from torch_scatter import scatter_add


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        half = time_emb_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 1:
            t = t[:, None]
        sin_inp = t * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(sin_inp), torch.cos(sin_inp)], dim=-1)
        if emb.shape[-1] < self.time_emb_dim:
            emb = F.pad(emb, (0, self.time_emb_dim - emb.shape[-1]))
        return self.proj(emb)


class MLP(nn.Module):
    """多层感知机（所有架构共享）"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 架构1: GNN (Geometric Graph Neural Network)
# ============================================================================

class GeoMPBlock(nn.Module):
    """几何消息传递块"""
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = MLP(in_dim=hidden_dim*2 + edge_dim,
                           hidden_dim=hidden_dim,
                           out_dim=hidden_dim,
                           dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim*2, out_dim=hidden_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_feat):
        row, col = edge_index
        m_ij = self.msg_mlp(torch.cat([x[row], x[col], edge_feat], dim=-1))
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x = self.ln1(x + m_i)
        x = self.ln2(x + self.ffn(x))
        return x


class GNNBackbone(nn.Module):
    """GNN骨干网络"""
    def __init__(self, atom_types, hidden_dim, num_layers, time_emb_dim, dropout,
                 cutoff_radius=5.0, max_num_neighbors=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff_radius = cutoff_radius
        self.max_num_neighbors = max_num_neighbors

        # 节点编码
        self.node_in = nn.Linear(atom_types, hidden_dim)

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 边编码
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # GNN层
        self.blocks = nn.ModuleList([
            GeoMPBlock(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 图池化
        self.pool_proj = nn.Linear(hidden_dim*2, hidden_dim)

    def build_molecular_graph(self, pos, batch):
        return radius_graph(pos, r=self.cutoff_radius, batch=batch,
                          max_num_neighbors=self.max_num_neighbors, loop=False)

    def encode_edges(self, pos, edge_index):
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)

        dist_feat = self.dist_mlp(edge_dist)
        dir_feat = self.dir_mlp(edge_dir)
        edge_feat = torch.cat([dist_feat, dir_feat], dim=-1)
        return self.edge_norm(edge_feat)

    def forward(self, theta_t, pos_t, t, batch):
        device = theta_t.device
        batch_safe = batch.detach().clone()

        # 构建图
        edge_index = self.build_molecular_graph(pos_t, batch_safe)

        # 节点初始化
        theta_t = theta_t.softmax(dim=-1) if (theta_t.min() < 0) or (theta_t.max() > 1.0) else theta_t
        x = self.node_in(theta_t)

        # 添加时间嵌入
        t_emb = self.time_proj(self.time_embed(t.to(device)))
        x = x + t_emb[batch_safe]

        # 边特征
        edge_feat = self.encode_edges(pos_t, edge_index)

        # 消息传递
        for blk in self.blocks:
            x = blk(x, edge_index, edge_feat)

        # 图池化
        mean_pool = global_mean_pool(x, batch_safe)
        max_pool = global_max_pool(x, batch_safe)
        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)
        graph_feat = F.silu(self.pool_proj(graph_feat))

        return graph_feat


# ============================================================================
# 架构2: Transformer (Self-Attention based)
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        # FFN
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerBackbone(nn.Module):
    """Transformer骨干网络"""
    def __init__(self, atom_types, hidden_dim, num_layers, time_emb_dim, dropout, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 节点编码
        self.node_in = nn.Linear(atom_types, hidden_dim)

        # 位置编码（使用3D坐标）
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Transformer层
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 图池化
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, theta_t, pos_t, t, batch):
        device = theta_t.device
        batch_safe = batch.detach().clone()

        # 节点初始化
        theta_t = theta_t.softmax(dim=-1) if (theta_t.min() < 0) or (theta_t.max() > 1.0) else theta_t
        x = self.node_in(theta_t)

        # 添加位置编码
        pos_emb = self.pos_encoder(pos_t)
        x = x + pos_emb

        # 添加时间嵌入
        t_emb = self.time_proj(self.time_embed(t.to(device)))
        x = x + t_emb[batch_safe]

        # 将节点特征按batch组织成序列
        # 为了使用Transformer，需要将每个分子的原子组织成一个序列
        batch_size = batch_safe.max().item() + 1
        max_atoms = scatter_add(torch.ones_like(batch_safe), batch_safe).max().item()

        # 创建填充的序列 [B, max_atoms, H]
        x_padded = torch.zeros(batch_size, max_atoms, self.hidden_dim, device=device)
        mask = torch.ones(batch_size, max_atoms, dtype=torch.bool, device=device)

        for b in range(batch_size):
            mask_b = (batch_safe == b)
            n_atoms = mask_b.sum().item()
            x_padded[b, :n_atoms] = x[mask_b]
            mask[b, :n_atoms] = False

        # Transformer处理
        for blk in self.blocks:
            x_padded = blk(x_padded, mask)

        # 图池化（平均池化，忽略padding）
        graph_feat = []
        for b in range(batch_size):
            n_atoms = (~mask[b]).sum().item()
            graph_feat.append(x_padded[b, :n_atoms].mean(dim=0))
        graph_feat = torch.stack(graph_feat, dim=0)
        graph_feat = F.silu(self.pool_proj(graph_feat))

        return graph_feat


# ============================================================================
# 架构3: MLP (Multi-Layer Perceptron) - Baseline
# ============================================================================

class MLPBackbone(nn.Module):
    """MLP骨干网络（简单baseline）"""
    def __init__(self, atom_types, hidden_dim, num_layers, time_emb_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 节点编码 + 位置编码
        self.node_in = nn.Linear(atom_types + 3, hidden_dim)  # theta + pos

        # MLP层
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*layers)

        # 图池化
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, theta_t, pos_t, t, batch):
        device = theta_t.device
        batch_safe = batch.detach().clone()

        # 节点初始化（拼接theta和pos）
        theta_t = theta_t.softmax(dim=-1) if (theta_t.min() < 0) or (theta_t.max() > 1.0) else theta_t
        x = torch.cat([theta_t, pos_t], dim=-1)
        x = self.node_in(x)

        # 添加时间嵌入
        t_emb = self.time_proj(self.time_embed(t.to(device)))
        x = x + t_emb[batch_safe]

        # MLP处理
        x = self.mlp(x)

        # 图池化（简单平均）
        graph_feat = global_mean_pool(x, batch_safe)
        graph_feat = F.silu(self.pool_proj(graph_feat))

        return graph_feat


# ============================================================================
# 架构4: Hybrid (GNN + Transformer)
# ============================================================================

class HybridBackbone(nn.Module):
    """混合骨干网络（GNN + Transformer）"""
    def __init__(self, atom_types, hidden_dim, num_layers, time_emb_dim, dropout,
                 cutoff_radius=5.0, max_num_neighbors=32, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff_radius = cutoff_radius
        self.max_num_neighbors = max_num_neighbors

        # 节点编码
        self.node_in = nn.Linear(atom_types, hidden_dim)

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 边编码（用于GNN）
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # 位置编码（用于Transformer）
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 混合层：前半部分GNN，后半部分Transformer
        num_gnn_layers = num_layers // 2
        num_transformer_layers = num_layers - num_gnn_layers

        self.gnn_blocks = nn.ModuleList([
            GeoMPBlock(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # 图池化
        self.pool_proj = nn.Linear(hidden_dim*2, hidden_dim)

    def build_molecular_graph(self, pos, batch):
        return radius_graph(pos, r=self.cutoff_radius, batch=batch,
                          max_num_neighbors=self.max_num_neighbors, loop=False)

    def encode_edges(self, pos, edge_index):
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_dir = edge_vec / (edge_dist + 1e-8)

        dist_feat = self.dist_mlp(edge_dist)
        dir_feat = self.dir_mlp(edge_dir)
        edge_feat = torch.cat([dist_feat, dir_feat], dim=-1)
        return self.edge_norm(edge_feat)

    def forward(self, theta_t, pos_t, t, batch):
        device = theta_t.device
        batch_safe = batch.detach().clone()

        # 节点初始化
        theta_t = theta_t.softmax(dim=-1) if (theta_t.min() < 0) or (theta_t.max() > 1.0) else theta_t
        x = self.node_in(theta_t)

        # 添加位置编码
        pos_emb = self.pos_encoder(pos_t)
        x = x + pos_emb

        # 添加时间嵌入
        t_emb = self.time_proj(self.time_embed(t.to(device)))
        x = x + t_emb[batch_safe]

        # 第一阶段：GNN处理
        edge_index = self.build_molecular_graph(pos_t, batch_safe)
        edge_feat = self.encode_edges(pos_t, edge_index)

        for blk in self.gnn_blocks:
            x = blk(x, edge_index, edge_feat)

        # 第二阶段：Transformer处理
        batch_size = batch_safe.max().item() + 1
        max_atoms = scatter_add(torch.ones_like(batch_safe), batch_safe).max().item()

        # 创建填充的序列
        x_padded = torch.zeros(batch_size, max_atoms, self.hidden_dim, device=device)
        mask = torch.ones(batch_size, max_atoms, dtype=torch.bool, device=device)

        for b in range(batch_size):
            mask_b = (batch_safe == b)
            n_atoms = mask_b.sum().item()
            x_padded[b, :n_atoms] = x[mask_b]
            mask[b, :n_atoms] = False

        for blk in self.transformer_blocks:
            x_padded = blk(x_padded, mask)

        # 图池化（结合平均和最大池化）
        mean_pool = []
        max_pool = []
        for b in range(batch_size):
            n_atoms = (~mask[b]).sum().item()
            mean_pool.append(x_padded[b, :n_atoms].mean(dim=0))
            max_pool.append(x_padded[b, :n_atoms].max(dim=0)[0])

        mean_pool = torch.stack(mean_pool, dim=0)
        max_pool = torch.stack(max_pool, dim=0)
        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)
        graph_feat = F.silu(self.pool_proj(graph_feat))

        return graph_feat



# ============================================================================
# 架构5: BiLSTM (Bidirectional LSTM)
# ============================================================================

class BiLSTMBackbone(nn.Module):

    def __init__(
        self,
        atom_types: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.atom_types = atom_types

        # 🔧 修复1：节点编码（使用 Linear 代替 Embedding，与 GNN 一致）
        self.node_in = nn.Linear(atom_types, hidden_dim)

        # 🔧 修复2：位置编码（增强为2层MLP，与 Transformer 一致）
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 🔧 修复4：改为单向LSTM，避免"作弊"
        # 双向LSTM可以同时看到前后信息，导致性能异常好
        # 单向LSTM更符合序列建模的实际场景
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,  # 单向，所以hidden_size不减半
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # 🔥 改为单向
        )

        # 输出投影（保持不变）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            theta_t: [N, K] 原子类型（one-hot或logits）
            pos_t: [N, 3] 3D坐标
            t: [B] 时间步
            batch: [N] batch索引

        Returns:
            graph_feat: [B, hidden_dim] 图级特征
        """
        device = theta_t.device
        batch_safe = batch.detach().clone()

        # 🔧 修复3：节点编码（使用概率分布，与 GNN 一致）
        # 归一化 theta_t 到概率分布
        theta_t = theta_t.softmax(dim=-1) if (theta_t.min() < 0) or (theta_t.max() > 1.0) else theta_t
        x = self.node_in(theta_t)  # [N, hidden_dim] - 使用 Linear 处理概率分布

        # 添加位置编码
        pos_emb = self.pos_encoder(pos_t)  # [N, hidden_dim]
        x = x + pos_emb

        # 时间嵌入（广播到所有节点）
        t_emb = self.time_embedding(t.to(device))    # [B, time_emb_dim]
        t_emb = self.time_proj(t_emb)                # [B, hidden_dim]
        x = x + t_emb[batch_safe]                    # [N, hidden_dim]

        # 将节点按batch分组，转换为序列
        batch_size = batch.max().item() + 1
        sequences = []
        lengths = []

        for b in range(batch_size):
            mask = (batch == b)
            seq = x[mask]  # [n_atoms, hidden_dim]
            sequences.append(seq)
            lengths.append(seq.size(0))

        # Pad序列到相同长度
        max_len = max(lengths)
        padded_seqs = torch.zeros(batch_size, max_len, self.hidden_dim, device=x.device)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = seq

        # 🔧 修复5：单向LSTM处理
        # Pack padded sequence for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            padded_seqs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.bilstm(packed_input)

        # 使用最后一层的隐藏状态（单向）
        # h_n: [num_layers, B, hidden_dim]
        graph_feat = h_n[-1, :, :]  # [B, hidden_dim] - 取最后一层

        # 输出投影
        graph_feat = self.output_proj(graph_feat)

        return graph_feat


# ============================================================================
# 架构6: GRU (Gated Recurrent Unit)
# ============================================================================

class GRUBackbone(nn.Module):
    """
    GRU骨干网络

    特点：
    - 使用双向GRU捕捉序列信息
    - 比LSTM参数更少，训练更快
    - 适合处理原子序列
    """
    def __init__(
        self,
        atom_types: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 节点编码
        self.node_encoder = nn.Embedding(atom_types, hidden_dim)

        # 位置编码（3D坐标）
        self.pos_encoder = nn.Linear(3, hidden_dim)

        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # BiGRU层
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 双向，所以hidden_size减半
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            theta_t: [N, K] 原子类型（one-hot或logits）
            pos_t: [N, 3] 3D坐标
            t: [B] 时间步
            batch: [N] batch索引

        Returns:
            graph_feat: [B, hidden_dim] 图级特征
        """
        # 节点编码
        if theta_t.dim() == 2 and theta_t.size(1) > 1:
            theta_idx = theta_t.argmax(dim=-1)
        else:
            theta_idx = theta_t.long().squeeze(-1)

        x = self.node_encoder(theta_idx)  # [N, hidden_dim]
        x = x + self.pos_encoder(pos_t)   # 添加位置信息

        # 时间嵌入（广播到所有节点）
        t_emb = self.time_embedding(t)    # [B, time_emb_dim]
        t_emb = self.time_proj(t_emb)     # [B, hidden_dim]
        x = x + t_emb[batch]              # [N, hidden_dim]

        # 将节点按batch分组，转换为序列
        batch_size = batch.max().item() + 1
        sequences = []
        lengths = []

        for b in range(batch_size):
            mask = (batch == b)
            seq = x[mask]  # [n_atoms, hidden_dim]
            sequences.append(seq)
            lengths.append(seq.size(0))

        # Pad序列到相同长度
        max_len = max(lengths)
        padded_seqs = torch.zeros(batch_size, max_len, self.hidden_dim, device=x.device)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = seq

        # BiGRU处理
        # Pack padded sequence for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            padded_seqs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.bigru(packed_input)

        # 使用最后一层的隐藏状态（前向和后向拼接）
        # h_n: [num_layers * 2, B, hidden_dim // 2]
        h_forward = h_n[-2, :, :]   # [B, hidden_dim // 2]
        h_backward = h_n[-1, :, :]  # [B, hidden_dim // 2]
        graph_feat = torch.cat([h_forward, h_backward], dim=-1)  # [B, hidden_dim]

        # 输出投影
        graph_feat = self.output_proj(graph_feat)

        return graph_feat




# ============================================================================
# 架构7: CNN (Convolutional Neural Network)
# ============================================================================

class CNNBackbone(nn.Module):
    """
    CNN骨干网络

    特点：
    - 使用1D卷积处理原子序列
    - 捕捉局部模式
    - 参数共享，计算高效
    """
    def __init__(
        self,
        atom_types: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        kernel_size: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # 节点编码
        self.node_encoder = nn.Embedding(atom_types, hidden_dim)

        # 位置编码（3D坐标）
        self.pos_encoder = nn.Linear(3, hidden_dim)

        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 1D卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
            )

        # 全局池化
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            theta_t: [N, K] 原子类型（one-hot或logits）
            pos_t: [N, 3] 3D坐标
            t: [B] 时间步
            batch: [N] batch索引

        Returns:
            graph_feat: [B, hidden_dim] 图级特征
        """
        # 节点编码
        if theta_t.dim() == 2 and theta_t.size(1) > 1:
            theta_idx = theta_t.argmax(dim=-1)
        else:
            theta_idx = theta_t.long().squeeze(-1)

        x = self.node_encoder(theta_idx)  # [N, hidden_dim]
        x = x + self.pos_encoder(pos_t)   # 添加位置信息

        # 时间嵌入（广播到所有节点）
        t_emb = self.time_embedding(t)    # [B, time_emb_dim]
        t_emb = self.time_proj(t_emb)     # [B, hidden_dim]
        x = x + t_emb[batch]              # [N, hidden_dim]

        # 将节点按batch分组，转换为序列
        batch_size = batch.max().item() + 1
        sequences = []
        lengths = []

        for b in range(batch_size):
            mask = (batch == b)
            seq = x[mask]  # [n_atoms, hidden_dim]
            sequences.append(seq)
            lengths.append(seq.size(0))

        # Pad序列到相同长度
        max_len = max(lengths)
        padded_seqs = torch.zeros(batch_size, max_len, self.hidden_dim, device=x.device)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = seq

        # 转换为CNN输入格式: [B, C, L]
        x = padded_seqs.transpose(1, 2)  # [B, hidden_dim, max_len]

        # 卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x) + x  # 残差连接

        # 全局池化
        x = self.global_pool(x).squeeze(-1)  # [B, hidden_dim]

        # 输出投影
        graph_feat = self.output_proj(x)

        return graph_feat


# ============================================================================
# 架构8: ResNet (Residual Network)
# ============================================================================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNetBackbone(nn.Module):
    """
    ResNet骨干网络

    特点：
    - 使用残差连接缓解梯度消失
    - 可以训练更深的网络
    - 适合复杂的特征提取
    """
    def __init__(
        self,
        atom_types: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 节点编码
        self.node_encoder = nn.Embedding(atom_types, hidden_dim)

        # 位置编码（3D坐标）
        self.pos_encoder = nn.Linear(3, hidden_dim)

        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            theta_t: [N, K] 原子类型（one-hot或logits）
            pos_t: [N, 3] 3D坐标
            t: [B] 时间步
            batch: [N] batch索引

        Returns:
            graph_feat: [B, hidden_dim] 图级特征
        """
        # 节点编码
        if theta_t.dim() == 2 and theta_t.size(1) > 1:
            theta_idx = theta_t.argmax(dim=-1)
        else:
            theta_idx = theta_t.long().squeeze(-1)

        x = self.node_encoder(theta_idx)  # [N, hidden_dim]
        x = x + self.pos_encoder(pos_t)   # 添加位置信息

        # 时间嵌入（广播到所有节点）
        t_emb = self.time_embedding(t)    # [B, time_emb_dim]
        t_emb = self.time_proj(t_emb)     # [B, hidden_dim]
        x = x + t_emb[batch]              # [N, hidden_dim]

        # 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)

        # 图池化（mean + max）
        batch_size = batch.max().item() + 1
        mean_pool = global_mean_pool(x, batch)  # [B, hidden_dim]
        max_pool = global_max_pool(x, batch)    # [B, hidden_dim]
        graph_feat = mean_pool + max_pool       # [B, hidden_dim]

        # 输出投影
        graph_feat = self.output_proj(graph_feat)

        return graph_feat


# ============================================================================
# 统一接口：多架构条件引导网络
# ============================================================================

class MultiArchGuidanceNetwork(nn.Module):
    """
    支持多种架构的条件引导网络

    Args:
        architecture: 'gnn', 'transformer', 'mlp', 'hybrid', 'bilstm', 'gru', 'cnn', 'resnet'
        atom_types: 原子类型数量
        hidden_dim: 隐藏层维度
        num_layers: 网络层数
        time_emb_dim: 时间嵌入维度
        condition_dim: 条件维度（QED, SA）
        dropout: Dropout率
        cutoff_radius: GNN的截断半径
        max_num_neighbors: GNN的最大邻居数
        num_heads: Transformer的注意力头数
        kernel_size: CNN的卷积核大小
    """
    def __init__(
        self,
        architecture: str = 'gnn',
        atom_types: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 64,
        condition_dim: int = 2,
        dropout: float = 0.1,
        cutoff_radius: float = 5.0,
        max_num_neighbors: int = 32,
        num_heads: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.condition_dim = condition_dim

        # 选择骨干网络
        if self.architecture == 'gnn':
            self.backbone = GNNBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout,
                cutoff_radius, max_num_neighbors
            )
        elif self.architecture == 'transformer':
            self.backbone = TransformerBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout, num_heads
            )
        elif self.architecture == 'mlp':
            self.backbone = MLPBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout
            )
        elif self.architecture == 'hybrid':
            self.backbone = HybridBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout,
                cutoff_radius, max_num_neighbors, num_heads
            )
        elif self.architecture == 'bilstm':
            self.backbone = BiLSTMBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout
            )
        elif self.architecture == 'gru':
            self.backbone = GRUBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout
            )
        elif self.architecture == 'cnn':
            self.backbone = CNNBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout, kernel_size
            )
        elif self.architecture == 'resnet':
            self.backbone = ResNetBackbone(
                atom_types, hidden_dim, num_layers, time_emb_dim, dropout
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Choose from ['gnn', 'transformer', 'mlp', 'hybrid', "
                           f"'bilstm', 'gru', 'cnn', 'resnet']")

        # 预测头（所有架构共享）
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim),
            nn.Sigmoid()  # 归一化到[0,1]
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim)
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            mu: [B, condition_dim] 预测的均值
            sigma: [B, condition_dim] 预测的标准差（正数）
        """
        with torch.enable_grad():
            # 骨干网络提取特征
            graph_feat = self.backbone(theta_t, pos_t, t, batch)

            # 预测头
            mu = self.mu_head(graph_feat)
            raw_sigma = self.sigma_head(graph_feat)
            sigma = self.softplus(raw_sigma) + 1e-3
            sigma = sigma.clamp(min=1e-3, max=0.5)

            return mu, sigma


# ============================================================================
# 工厂函数
# ============================================================================

def create_multi_arch_guidance_network(config: Optional[dict] = None) -> MultiArchGuidanceNetwork:
    """
    创建多架构条件引导网络

    Args:
        config: 配置字典，包含以下键：
            - architecture: 'gnn', 'transformer', 'mlp', 'hybrid'
            - atom_types: 原子类型数量
            - hidden_dim: 隐藏层维度
            - num_layers: 网络层数
            - time_emb_dim: 时间嵌入维度
            - condition_dim: 条件维度
            - dropout: Dropout率
            - cutoff_radius: GNN的截断半径
            - max_num_neighbors: GNN的最大邻居数
            - num_heads: Transformer的注意力头数

    Returns:
        MultiArchGuidanceNetwork实例
    """
    default_config = {
        'architecture': 'gnn',
        'atom_types': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'time_emb_dim': 64,
        'condition_dim': 2,
        'dropout': 0.1,
        'cutoff_radius': 5.0,
        'max_num_neighbors': 32,
        'num_heads': 8,
    }

    if config is not None:
        default_config.update(config)

    return MultiArchGuidanceNetwork(**default_config)


def get_architecture_info(architecture: str) -> dict:
    """
    获取架构信息

    Args:
        architecture: 'gnn', 'transformer', 'mlp', 'hybrid', 'bilstm', 'gru', 'cnn', 'resnet'

    Returns:
        包含架构描述的字典
    """
    info = {
        'gnn': {
            'name': 'Geometric Graph Neural Network',
            'description': '基于几何图神经网络，使用消息传递捕捉3D空间结构',
            'strengths': ['捕捉局部几何信息', '参数效率高', '对分子图结构敏感'],
            'weaknesses': ['感受野受限于截断半径', '难以捕捉长程相互作用']
        },
        'transformer': {
            'name': 'Self-Attention Transformer',
            'description': '基于自注意力机制，全局建模原子间相互作用',
            'strengths': ['全局感受野', '捕捉长程相互作用', '灵活的注意力模式'],
            'weaknesses': ['计算复杂度高O(N²)', '参数量大', '可能忽略局部几何']
        },
        'mlp': {
            'name': 'Multi-Layer Perceptron',
            'description': '简单的全连接网络，作为baseline',
            'strengths': ['简单高效', '参数少', '训练快'],
            'weaknesses': ['无法捕捉原子间相互作用', '忽略图结构', '性能较差']
        },
        'hybrid': {
            'name': 'Hybrid GNN + Transformer',
            'description': '混合架构：先用GNN捕捉局部几何，再用Transformer建模全局',
            'strengths': ['结合GNN和Transformer优势', '局部+全局信息', '性能最优'],
            'weaknesses': ['参数量最大', '计算复杂度高', '训练时间长']
        },
        'bilstm': {
            'name': 'Bidirectional LSTM',
            'description': '双向长短期记忆网络，捕捉序列的前向和后向依赖',
            'strengths': ['捕捉长期依赖', '双向信息流', '适合序列建模'],
            'weaknesses': ['训练速度慢', '难以并行化', '可能忽略3D几何']
        },
        'gru': {
            'name': 'Gated Recurrent Unit',
            'description': '门控循环单元，比LSTM参数更少的序列模型',
            'strengths': ['比LSTM参数少', '训练更快', '捕捉序列依赖'],
            'weaknesses': ['难以并行化', '可能忽略3D几何', '长序列性能下降']
        },
        'cnn': {
            'name': 'Convolutional Neural Network',
            'description': '1D卷积神经网络，捕捉局部模式',
            'strengths': ['参数共享', '计算高效', '捕捉局部模式', '易于并行化'],
            'weaknesses': ['感受野受限', '难以捕捉长程依赖', '可能忽略全局结构']
        },
        'resnet': {
            'name': 'Residual Network',
            'description': '残差网络，使用残差连接训练更深的网络',
            'strengths': ['缓解梯度消失', '可训练更深网络', '特征提取能力强'],
            'weaknesses': ['参数量较大', '可能忽略图结构', '需要更多训练数据']
        }
    }

    return info.get(architecture.lower(), {'name': 'Unknown', 'description': 'Unknown architecture'})


