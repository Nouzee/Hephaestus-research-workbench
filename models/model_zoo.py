"""
Hephaestus Model Zoo — adapted from 高频项目's proven HybridTransformerLSTM.

Converts from 3-class classification to scalar regression.
Architecture: Input Projection → PositionalEncoding → Transformer → LSTM → Regressor
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Building blocks (from 高频项目)
# ═══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(torch.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n


# ═══════════════════════════════════════════════════════════════
# HybridTransformerLSTM — Regression variant (from 高频项目)
# ═══════════════════════════════════════════════════════════════
class HybridTransformerLSTM(nn.Module):
    """
    Transformer + LSTM hybrid for BTC L2 regression.

    Adapted from 高频项目's proven classification model.
    Changed: 3-class output → scalar regression output.
    Changed: 14 input_dim → configurable.
    """

    def __init__(self,
                 input_dim=12,   # 11 features + 1 time-delta
                 d_model=64,
                 nhead=4,
                 num_transformer_layers=3,
                 lstm_hidden=64,
                 lstm_layers=2,
                 dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.lstm = LSTMLayer(d_model, lstm_hidden, lstm_layers, dropout)
        self.dropout = nn.Dropout(dropout)

        # Regression head (replaces 3-class classifier)
        self.regressor = nn.Linear(lstm_hidden * lstm_layers, 1)

        self._init_weights()

    def forward(self, x_feat, x_dt):
        """
        x_feat: (B, T, n_features) — from WindowDataset, squeezed from (B,1,T,F)
        x_dt:   (B, T, 1) — time delta (appended as extra feature)

        Returns: (B,) scalar predictions
        """
        # Squeeze the channel dim that WindowDataset adds
        if x_feat.dim() == 4:
            x_feat = x_feat.squeeze(1)  # (B, 1, T, F) → (B, T, F)

        # Append time delta as an extra feature channel
        x = torch.cat([x_feat, x_dt], dim=-1)  # (B, T, F+1)

        # Project to model dimension
        x = self.input_projection(x)             # (B, T, d_model)
        x = self.pos_encoder(x)

        # Transformer blocks
        for layer in self.transformer_layers:
            x = layer(x)

        # LSTM
        lstm_out, hidden = self.lstm(x)          # hidden: (num_layers, B, hidden)

        # Aggregate LSTM hidden states
        hidden_flat = hidden.permute(1, 0, 2).contiguous()  # (B, num_layers, hidden)
        hidden_flat = hidden_flat.reshape(hidden_flat.size(0), -1)  # (B, num_layers*hidden)
        hidden_flat = self.dropout(hidden_flat)

        # Regression output
        pred = self.regressor(hidden_flat).squeeze(-1)  # (B,)

        return pred

    def _init_weights(self):
        """Xavier init all Linear layers, zero-init final regressor bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════
# Lighter variant for quick experiments
# ═══════════════════════════════════════════════════════════════
class HybridTransformerLSTM_Small(nn.Module):
    """Smaller version: fewer params, faster training."""

    def __init__(self, input_dim=12, d_model=32, nhead=4,
                 num_transformer_layers=2, lstm_hidden=32, lstm_layers=1,
                 dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.lstm = LSTMLayer(d_model, lstm_hidden, lstm_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(lstm_hidden * lstm_layers, 1)

        self._init_weights()

    def forward(self, x_feat, x_dt):
        if x_feat.dim() == 4:
            x_feat = x_feat.squeeze(1)
        x = torch.cat([x_feat, x_dt], dim=-1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        for layer in self.transformer_layers:
            x = layer(x)
        _, hidden = self.lstm(x)
        hidden_flat = hidden.permute(1, 0, 2).contiguous()
        hidden_flat = hidden_flat.reshape(hidden_flat.size(0), -1)
        hidden_flat = self.dropout(hidden_flat)
        return self.regressor(hidden_flat).squeeze(-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════
# Also keep the BTCNet for comparison experiments
# ═══════════════════════════════════════════════════════════════
class BTCLobNet_RegWithDT(nn.Module):
    """
    1D Conv + LSTM hybrid for BTC L2 regression (from ExperimentC architecture).
    Simpler but proven pattern.
    """

    def __init__(self, n_features=14, T=100, channels=16):
        super().__init__()
        n_pv = max(1, n_features // 2)
        n_na = n_features - n_pv

        self.n_pv = n_pv
        self.n_na = n_na
        ch = channels

        self.pv_fuse = nn.Conv2d(1, ch, kernel_size=(1, 2), stride=(1, 2))
        self.na_fuse = nn.Conv2d(1, ch, kernel_size=(1, 2), stride=(1, 2))
        self.pv_temp = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0))
        self.na_temp = nn.Conv2d(ch, ch, kernel_size=(5, 1), padding=(2, 0))
        self.fusion_conv = nn.Conv2d(ch * 2, 32, kernel_size=1)
        self.joint_temp = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))

        self.dt_conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.dt_conv2 = nn.Conv1d(8, 8, kernel_size=5, padding=2)

        self.lstm = nn.LSTM(32 + 8, 64, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x_feat, x_dt):
        B, _, T, F = x_feat.shape
        x_pv = x_feat[:, :, :, :self.n_pv]
        x_na = x_feat[:, :, :, self.n_pv:self.n_pv + self.n_na]

        pv = F.relu(self.pv_fuse(x_pv))
        pv = F.relu(self.pv_temp(pv))
        na = F.relu(self.na_fuse(x_na))
        na = F.relu(self.na_temp(na))

        pv = pv.mean(dim=-1)
        na = na.mean(dim=-1)
        fused = torch.cat([pv, na], dim=1).unsqueeze(-1)
        fused = F.relu(self.fusion_conv(fused))
        fused = F.relu(self.joint_temp(fused)).squeeze(-1)

        dt = x_dt.permute(0, 2, 1)
        dt = F.relu(self.dt_conv1(dt))
        dt = F.relu(self.dt_conv2(dt))

        seq = torch.cat([fused.permute(0, 2, 1), dt.permute(0, 2, 1)], dim=2)
        _, (hidden, _) = self.lstm(seq)
        return self.regressor(hidden.squeeze(0)).squeeze(-1)
