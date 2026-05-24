"""
Neural SDE — Neural Stochastic Differential Equations
Model BTC price as a continuous-time stochastic process;
output probability clouds for the future price trajectory.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SDEConfig:
    state_dim: int = 1
    hidden_dim: int = 64
    num_samples: int = 100
    dt: float = 0.001
    method: str = "euler"
    noise_type: str = "additive"


class DriftNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class DiffusionNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = self.net(torch.cat([x, t], dim=-1))
        sd = x.shape[-1]
        sigma = flat.view(-1, sd, sd)
        return sigma @ sigma.transpose(-2, -1) + 1e-4


class NeuralSDE(nn.Module):
    def __init__(self, config: SDEConfig):
        super().__init__()
        self.config = config
        self.drift = DriftNet(config.state_dim, config.hidden_dim)
        self.diffusion = DiffusionNet(config.state_dim, config.hidden_dim)

    def forward(self, x0: torch.Tensor, t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n_steps = x0.shape[0], len(t_span)
        trajectories = []
        for _ in range(self.config.num_samples):
            x = x0.clone()
            path = [x]
            for i in range(n_steps - 1):
                t = t_span[i].expand(b, 1)
                mu = self.drift(x, t)
                sigma = self.diffusion(x, t)
                dt_val = self.config.dt
                dw = torch.randn_like(x) * torch.sqrt(torch.tensor(dt_val))
                x = x + mu * dt_val + sigma @ dw.unsqueeze(-1)
                path.append(x)
            trajectories.append(torch.stack(path, dim=1))
        return torch.stack(trajectories, dim=1), torch.zeros(b, self.config.num_samples)

    def sample(self, x0: torch.Tensor, horizon: float = 0.5, num_steps: int = 50) -> torch.Tensor:
        t_span = torch.linspace(0, horizon, num_steps, device=x0.device)
        traces, _ = self.forward(x0, t_span)
        return traces[:, :, :, 0]

    def expected_price(self, x0: torch.Tensor, horizon: float = 0.5, num_steps: int = 50) -> torch.Tensor:
        return self.sample(x0, horizon, num_steps).mean(dim=1)


class NeuralODEModule(nn.Module):
    def __init__(self, input_dim: int = 24, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, t], dim=-1))


class NeuralODEPredictor(nn.Module):
    def __init__(self, feature_dim: int = 24, hidden_dim: int = 64, pred_len: int = 1):
        super().__init__()
        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        b, seq_len, _ = x.shape
        h = self.encoder(x)
        h0 = h[:, -1]
        path = [h0]
        dt_val = 0.1
        for t in range(seq_len):
            t_ten = torch.full((b, 1), t * dt_val, device=x.device)
            dh = self.ode_func(torch.cat([h0, t_ten], dim=-1))
            h0 = h0 + dh * dt_val
            path.append(h0)
        return torch.stack(path, dim=1) if return_trajectory else self.decoder(h0)


class MicroPriceV2(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.bilstm = nn.LSTM(2, emb_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(emb_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.bilstm(depth)
        return self.head(torch.cat([h_n[0], h_n[-1]], dim=-1))


def create_neural_sde(state_dim: int = 1, hidden_dim: int = 64) -> NeuralSDE:
    return NeuralSDE(SDEConfig(state_dim, hidden_dim))


def create_neural_ode(feature_dim: int = 24, hidden_dim: int = 64, pred_len: int = 1) -> NeuralODEPredictor:
    return NeuralODEPredictor(feature_dim, hidden_dim, pred_len)


def create_micro_price_v2() -> MicroPriceV2:
    return MicroPriceV2()
