"""
ST-GNN — Spatial-Temporal Graph Neural Network
Orders book depth and cross-asset graph modelling.

Uses graph convolutions over bid/ask ladder levels connected
as graph nodes, with temporal convolutions for signal propagation.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class STGNNConfig:
    node_features: int = 24
    hidden_dim: int = 128
    num_nodes: int = 10
    num_layers: int = 3
    temporal_kernel: int = 3
    dropout: float = 0.1


class TemporalConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = x.shape
        out = torch.zeros_like(x)
        for b in range(batch):
            for i in range(num_nodes):
                neighbours = edge_idx[1][edge_idx[0] == i]
                if len(neighbours) > 0:
                    agg = x[b, neighbours].mean(dim=0)
                    out[b, i] = F.relu(self.linear(x[b, i]) + self.linear(agg))
                else:
                    out[b, i] = x[b, i]
        return out


class STGNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_nodes: int, kernel: int = 3):
        super().__init__()
        self.temporal = TemporalConv(in_ch, out_ch, kernel)
        self.spatial = SpatialGraphConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        b, nnodes, seq_len, feat = x.shape
        x_t = x.permute(0, 1, 3, 2).reshape(b * nnodes, feat, seq_len)
        x_t = self.temporal(x_t)
        _, feat, seq_len = x_t.shape
        x = x_t.reshape(b, nnodes, seq_len, feat)
        x = x.permute(0, 2, 1, 3)
        for t in range(seq_len):
            x[:, t] = self.spatial(x[:, t], edge_idx)
        return x.permute(0, 2, 1, 3)


class STGNNEncoder(nn.Module):
    def __init__(self, config: STGNNConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.node_features, config.hidden_dim)
        self.blocks = nn.ModuleList([
            STGNNBlock(config.hidden_dim, config.hidden_dim, config.num_nodes, config.temporal_kernel)
            for _ in range(config.num_layers)
        ])
        self.out_proj = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_idx)
        x = x[:, :, -1, :].mean(dim=1)
        return self.out_proj(x)


class OrderBookSTGNN(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.emb = nn.Linear(1, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, emb_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(emb_dim, emb_dim, 3, padding=1)
        self.cls = nn.Sequential(nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.unsqueeze(-1).permute(0, 1, 3, 2).reshape(b * 10, 1, 60)
        x = self.emb(x).reshape(b, 10, -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)).permute(0, 2, 1)
        bid, ask = x[:, :5].mean(dim=1), x[:, 5:].mean(dim=1)
        return self.cls(torch.cat([bid, ask], dim=-1))


def create_edge_index(num_levels: int = 5, include_correlated: bool = True) -> torch.Tensor:
    edges = []
    for i in range(num_levels - 1):
        edges.extend([[i, i + 1], [i + 1, i]])
        edges.extend([[i + num_levels, i + num_levels + 1], [i + num_levels + 1, i + num_levels]])
        edges.extend([[i, i + num_levels], [i + num_levels, i]])
    if include_correlated:
        base = num_levels * 2
        for i in range(num_levels):
            edges.extend([[i, base], [base, i], [i + num_levels, base + 1], [base + 1, i + num_levels]])
    return torch.tensor(edges, dtype=torch.long).t()


def create_stgnn(node_features: int = 24, hidden_dim: int = 128, num_nodes: int = 10) -> STGNNEncoder:
    return STGNNEncoder(STGNNConfig(node_features, hidden_dim, num_nodes))


def create_orderbook_stgnn() -> OrderBookSTGNN:
    return OrderBookSTGNN()
