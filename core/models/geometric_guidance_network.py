"""
------------------------------------
- Uses torch_geometric.nn.radius_graph to build edges per molecule in the batch.
- Adds *directional* geometry: for each edge i->j we encode
    - edge distance  ||x_j - x_i||
    - normalized direction (x_j - x_i) / ||x_j - x_i||
- Simple residual geometric GNN blocks aggregate messages using the encoded edge features.

Inputs
------
theta_t : [N, K]    (per-atom posterior over atom types, or logits projected to a simplex)
pos     : [N, 3]    (per-atom 3D coordinates at time t)
t       : [B] or [] (scalar or per-graph diffusion time in [0,1])
batch   : [N]       (graph id of each node)

Outputs
-------
mu      : [B, C]    (predicted mean of properties; C=condition_dim)
sigma   : [B, C]    (predicted std, strictly positive)

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import radius_graph, global_mean_pool, global_max_pool
from torch_scatter import scatter_add


# ---------- Utility modules ----------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic Transformer-style sinusoidal embedding for a scalar t in [0,1].
    Returns a vector of size time_emb_dim.
    """
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # Use a fixed set of frequencies
        half = time_emb_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [] or [B] or [B,1]; values in [0,1]
        returns: [B, time_emb_dim]
        """
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 1:
            t = t[:, None]
        # [B, 1] -> [B, half]
        sin_inp = t * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(sin_inp), torch.cos(sin_inp)], dim=-1)
        if emb.shape[-1] < self.time_emb_dim:  # odd dim guard
            emb = F.pad(emb, (0, self.time_emb_dim - emb.shape[-1]))
        return self.proj(emb)


class MLP(nn.Module):
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


# ---------- Geometric message passing block ----------

class GeoMPBlock(nn.Module):
    """
    A lightweight geometric message passing block with:
    - edge encoding: distance + direction -> edge_feat
    - message: MLP([x_i, x_j, edge_feat])
    - aggregation: sum
    - update: residual + FFN with LayerNorm
    """
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
        row, col = edge_index  # i <- j
        m_ij = self.msg_mlp(torch.cat([x[row], x[col], edge_feat], dim=-1))
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x = self.ln1(x + m_i)
        x = self.ln2(x + self.ffn(x))
        return x


# ---------- Main Network ----------

class GeometricGuidanceNetwork(nn.Module):
    def __init__(
        self,
        atom_types: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 64,
        condition_dim: int = 2,
        dropout: float = 0.1,
        cutoff_radius: float = 5.0,
        max_num_neighbors: int = 32,
        num_heads: int = 4,  # 只是为了兼容训练脚本传参，不用也行
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.cutoff_radius = cutoff_radius
        self.max_num_neighbors = max_num_neighbors

        # Project per-atom theta_t (K classes) to node hidden
        self.node_in = nn.Linear(atom_types, hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Edge encoders: distance and direction
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
        edge_dim = hidden_dim  # distance(=hidden/2) + direction(=hidden/2)

        # Geometric message passing stack
        self.blocks = nn.ModuleList([
            GeoMPBlock(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Readout heads (graph-level)
        self.pool_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim),
            nn.Sigmoid()  # properties normalized to [0,1]
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_dim)
        )
        self.softplus = nn.Softplus()  # positive std

    # ---- geometry helpers ----
    def build_molecular_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Use radius_graph per batch item.
        pos:   [N, 3]
        batch: [N]
        return edge_index [2, E]
        """
        edge_index = radius_graph(
            pos, r=self.cutoff_radius, batch=batch,
            max_num_neighbors=self.max_num_neighbors, loop=False
        )
        return edge_index

    def encode_edges(self, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute distance and direction features and map to hidden_dim.
        Returns edge_feat [E, hidden_dim].
        """
        row, col = edge_index
        edge_vec = pos[col] - pos[row]             # [E, 3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)  # [E,1]
        # safe normalize
        edge_dir = edge_vec / (edge_dist + 1e-8)

        dist_feat = self.dist_mlp(edge_dist)       # [E, hidden/2]
        dir_feat  = self.dir_mlp(edge_dir)         # [E, hidden/2]
        edge_feat = torch.cat([dist_feat, dir_feat], dim=-1)  # [E, hidden]
        edge_feat = self.edge_norm(edge_feat)
        return edge_feat

    # ---- forward ----
    def _normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """
        🔥 移除@torch.no_grad()装饰器，以支持梯度引导

        Ensure theta is a valid distribution if logits were passed.
        """
        # 🔥 关键修复：使用条件判断而不是@torch.no_grad()
        # 这样可以保持梯度传播
        if (theta.min() < 0) or (theta.max() > 1.0) or (theta.sum(-1) > 1.001).any():
            theta = theta.softmax(dim=-1)
        return theta

    def forward(
        self,
        theta_t: torch.Tensor,       # [N, K]
        pos_t: torch.Tensor,         # [N, 3]
        t: torch.Tensor,             # [] or [B] or [B,1]
        batch: torch.Tensor,         # [N]
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:    [B, condition_dim]
            sigma: [B, condition_dim], positive and modestly bounded
        """
        # 🔥 关键修复：强制启用梯度计算
        # 使用enable_grad()上下文管理器包装整个forward
        with torch.enable_grad():
            return self._forward_impl(theta_t, pos_t, t, batch, edge_index)

    def _forward_impl(
        self,
        theta_t: torch.Tensor,
        pos_t: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = theta_t.device

 
        if batch.numel() > 0:
            batch_min_input = batch.min()
            batch_max_input = batch.max()


            if batch_max_input > 10000 or batch_min_input < 0:

                batch_safe = torch.zeros_like(batch)
            else:
                batch_safe = batch.detach().clone()
        else:
            batch_safe = batch.detach().clone()

        if batch_safe.numel() > 0:
            batch_min = batch_safe.min()
            batch_max = batch_safe.max()

            if batch_max > 10000 or batch_min < 0:
                n_atoms = theta_t.size(0)
                if t.dim() == 0:
                    n_batch = 1
                elif t.dim() == 1:
                    n_batch = t.size(0)
                else:
                    n_batch = t.size(0)

                if n_batch == 1:

                    batch_safe = torch.zeros(n_atoms, dtype=torch.long, device=device)
                else:

                    atoms_per_mol = n_atoms // n_batch
                    batch_safe = torch.arange(n_atoms, device=device) // atoms_per_mol
                    batch_safe = torch.clamp(batch_safe, max=n_batch - 1)


        if edge_index is None:
            edge_index = self.build_molecular_graph(pos_t, batch_safe)  # [2, E]

        theta_t = self._normalize_theta(theta_t)
        x = self.node_in(theta_t)  # [N, H]

        if t.dim() == 0:
            t = t.unsqueeze(0)  # [] -> [1]

        num_molecules = batch_safe.max().item() + 1

        if t.size(0) == 1 and num_molecules > 1:
            t = t.expand(num_molecules)  # [1] -> [B]

        t_emb = self.time_proj(self.time_embed(t.to(device)))  # [B, H]

        if batch_safe.max() >= t_emb.size(0):
            x = x + t_emb[0:1].expand(x.size(0), -1)
        else:
            x = x + t_emb[batch_safe]  # broadcast by graph id

        # edge features
        edge_feat = self.encode_edges(pos_t, edge_index)       # [E, H]

        # message passing
        for blk in self.blocks:
            x = blk(x, edge_index, edge_feat)

        if batch_safe.max() >= 1000 or batch_safe.min() < 0:
            batch_for_pooling = torch.zeros_like(batch_safe)
        else:
            unique_batch = torch.unique(batch_safe, sorted=True)

            batch_for_pooling = batch_safe
            if unique_batch.numel() > 0 and unique_batch.max() >= unique_batch.numel():
                batch_mapping = torch.zeros(unique_batch.max() + 1, dtype=torch.long, device=batch_safe.device)
                for new_idx, old_idx in enumerate(unique_batch):
                    batch_mapping[old_idx] = new_idx
                batch_for_pooling = batch_mapping[batch_safe]

        mean_pool = global_mean_pool(x, batch_for_pooling)                 # [B, H]
        max_pool  = global_max_pool(x, batch_for_pooling)                  # [B, H]

        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]
        graph_feat = F.silu(self.pool_proj(graph_feat))        # [B, H]

        # heads
        mu = self.mu_head(graph_feat)                          # [B, C] in [0,1]
        raw_sigma = self.sigma_head(graph_feat)                # [B, C]
        sigma = self.softplus(raw_sigma) + 1e-3               # strictly positive
        sigma = sigma.clamp(min=1e-3, max=0.08)  # 从0.5降到0.08

        return mu, sigma



def build_geometric_guidance_network(config: Optional[dict] = None) -> GeometricGuidanceNetwork:
    default_config = {
        'atom_types': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'time_emb_dim': 64,
        'condition_dim': 2,
        'dropout': 0.1,
        'cutoff_radius': 5.0,
        'max_num_neighbors': 32,
    }
    if config is not None:
        config = dict(config)
        config.pop('num_heads', None)
        default_config.update(config)
    return GeometricGuidanceNetwork(**default_config)

def create_geometric_guidance_network(config=None):
    return build_geometric_guidance_network(config)
