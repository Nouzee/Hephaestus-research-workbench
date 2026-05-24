"""
Mamba Block - 状态空间模型 (SSM)
用于处理超长 Tick 序列，捕捉订单簿的极短期记忆和流动性冲击衰减

基于选择性状态空间 (Selective State Space) 架构
Reference: "Mamba: Linear-time Sequence Modeling with Selective State Spaces" (2024)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    """Mamba 配置"""
    d_model: int = 256        # 模型维度
    d_state: int = 16        # 状态扩展维度
    d_conv: int = 4         # 卷积核大小
    expand: int = 2          # 扩展因子
    dt_rank: int = 64        # 时间投影维度
    bias: bool = False


class SSM(nn.Module):
    """
    状态空间层 — 简化 Mamba SSM

    h_t = A * h_{t-1} + B_t * x_t
    y_t = C_t^T @ h_t + D * x_t
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape
        A = -torch.exp(self.A_log)  # (d_model, d_state), negative exponential
        B = self.B_proj(x)          # (batch, seq, d_state)
        C = self.C_proj(x)          # (batch, seq, d_state)

        if state is None:
            state = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            state = state * A.unsqueeze(0) + x_t.unsqueeze(-1) * B[:, t].unsqueeze(1)
            y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), state

    def step(self, x: torch.Tensor, state: torch.Tensor):
        A = -torch.exp(self.A_log)
        B = self.B_proj(x)
        C = self.C_proj(x)
        x_t = x.squeeze(1)
        state = state * A.unsqueeze(0) + x_t.unsqueeze(-1) * B.unsqueeze(1)
        y = (state * C.unsqueeze(1)).sum(dim=-1) + self.D * x_t
        return y, state


class MambaBlock(nn.Module):
    """
    Mamba 块

    组成:
    1. 投影层 (Input Projection)
    2. 深度可分离卷积 (Depthwise Conv)
    3. SSM 状态空间层
    4. 激活 + Dropout + 输出投影
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        d_model = config.d_model
        d_inner = int(config.expand * d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=config.bias)

        # 深度可分离卷积
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner,
        )

        # SSM
        self.ssm = SSM(d_inner, config.d_state)

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)

        # 激活
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # 投影
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 卷积
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_inner.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        # SSM
        y, state = self.ssm(x_conv)

        # 门控
        y = y * self.act(z)

        # 输出
        y = self.out_proj(y)

        return y

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向 (用于推理)"""
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 卷积单步 (简化)
        x_conv = self.act(x_inner)

        # SSM 单步
        y, state = self.ssm.step(x_conv, state)

        # 门控
        y = y * self.act(z)

        # 输出
        y = self.out_proj(y)

        return y, state


class MambaEncoder(nn.Module):
    """Mamba 编码器 (堆叠多个 MambaBlock)"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
        )

        self.layers = nn.ModuleList([
            MambaBlock(config)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) - token IDs
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()

        generated = prompt.clone()
        current_state = None

        for _ in range(max_len):
            # 嵌入最后一个 token
            x = self.embedding(generated[:, -1:])

            for layer in self.layers:
                x, current_state = layer.step(x, current_state or torch.zeros_like(x))

            logits = x[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)

            # 采样
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def create_mamba(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
) -> MambaEncoder:
    """创建 Mamba 编码器"""
    return MambaEncoder(vocab_size, d_model, n_layers)


# ===== 用于高频数据的简化版 =====

class HFTMamba(nn.Module):
    """
    简化版 Mamba，用于高频 Tick 序列预测

    输入: (batch, seq_len, feature_dim) - 价格/深度/订单流
    输出: (batch, pred_len, 1) - 预测价格变动
    """

    def __init__(
        self,
        feature_dim: int = 24,
        d_model: int = 128,
        d_state: int = 8,
        pred_len: int = 1,
    ):
        super().__init__()

        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
        )

        # 输入投影
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Mamba 层
        self.mamba = MambaBlock(config)

        # 输出投影
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            (batch, pred_len)
        """
        x = self.input_proj(x)
        x = self.mamba(x)

        # 取最后一个隐藏状态
        pred = self.output_proj(x[:, -1])

        return pred


def create_hft_mamba(
    feature_dim: int = 24,
    d_model: int = 128,
    pred_len: int = 1,
) -> HFTMamba:
    """创建高频交易专用 Mamba"""
    return HFTMamba(feature_dim, d_model, pred_len)


# 使用示例
if __name__ == "__main__":
    print("Mamba SSM v1.0")

    # 测试
    model = create_hft_mamba(feature_dim=24, d_model=64)
    x = torch.randn(8, 60, 24)  # batch=8, seq=60, features=24
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")