"""
Causal Wavelet / Multi-Scale Decomposition — Past-Only Filter Bank

Decomposes a time series into three frequency bands using causal
exponential moving averages (EMA). No future-looking filters.

  HF (high-freq):  x(t) - EMA_fast(t)     → sudden shocks, impulses
  MF (mid-freq):   EMA_fast - EMA_slow    → sustained pressure, trends
  LF (low-freq):   EMA_slow(t)            → regime, structural drift

Properties:
  - Strictly causal: output[t] depends only on x[0..t]
  - O(1) per sample, online-friendly
  - Zero-phase distortion risk (no symmetric kernel)

Spans are in batch units (~2048 ticks/batch for Hephaestus).
Typical: hf_span=2 (sudden), mf_span=10 (sustained), lf_span=50 (regime).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class CausalWaveletConfig:
    """Configuration for causal multi-scale decomposition."""

    # EMA spans (in samples/batches)
    hf_span: int = 2       # high-frequency: ~2 batches (~4K ticks)
    mf_span: int = 10      # mid-frequency: ~10 batches (~20K ticks)
    lf_span: int = 50      # low-frequency: ~50 batches (~100K ticks)

    # Causal: ensure no leakage
    causal: bool = True    # always true — enforced by EMA recursion


# ===========================================================================
# Core: Causal Decomposer
# ===========================================================================

class CausalDecomposer:
    """
    Past-only multi-scale decomposition via EMA filter bank.

    Usage
    -----
    >>> cd = CausalDecomposer()
    >>> hf, mf, lf = cd.decompose(signal)
    >>> # hf=shock, mf=trend, lf=regime
    """

    def __init__(self, config: Optional[CausalWaveletConfig] = None):
        self.config = config or CausalWaveletConfig()
        cfg = self.config

        self.alpha_hf = 2.0 / (cfg.hf_span + 1)
        self.alpha_mf = 2.0 / (cfg.mf_span + 1)
        self.alpha_lf = 2.0 / (cfg.lf_span + 1)

        # State for online updates
        self.ema_fast = 0.0
        self.ema_mid = 0.0
        self.ema_slow = 0.0
        self._initialized = False

    # ── Batch decomposition (offline) ─────────────────────────────────

    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose entire signal into HF, MF, LF components.

        Parameters
        ----------
        signal : (N,) array

        Returns
        -------
        hf, mf, lf : each (N,) array
        """
        n = len(signal)
        ema_fast = np.zeros(n, dtype=np.float64)
        ema_mid = np.zeros(n, dtype=np.float64)
        ema_slow = np.zeros(n, dtype=np.float64)

        ema_fast[0] = float(signal[0])
        ema_mid[0] = float(signal[0])
        ema_slow[0] = float(signal[0])

        for t in range(1, n):
            x = float(signal[t])
            ema_fast[t] = self.alpha_hf * x + (1 - self.alpha_hf) * ema_fast[t-1]
            ema_mid[t] = self.alpha_mf * x + (1 - self.alpha_mf) * ema_mid[t-1]
            ema_slow[t] = self.alpha_lf * x + (1 - self.alpha_lf) * ema_slow[t-1]

        hf = signal.astype(np.float64) - ema_fast
        mf = ema_fast - ema_slow
        lf = ema_slow

        return hf.astype(np.float32), mf.astype(np.float32), lf.astype(np.float32)

    # ── Online single-sample update ───────────────────────────────────

    def update(self, x: float) -> Tuple[float, float, float]:
        """
        Push one sample, return current HF/MF/LF values.

        Online-friendly: O(1), no buffer, no future.
        """
        if not self._initialized:
            self.ema_fast = float(x)
            self.ema_mid = float(x)
            self.ema_slow = float(x)
            self._initialized = True
            return 0.0, 0.0, float(x)

        self.ema_fast = self.alpha_hf * float(x) + (1 - self.alpha_hf) * self.ema_fast
        self.ema_mid = self.alpha_mf * float(x) + (1 - self.alpha_mf) * self.ema_mid
        self.ema_slow = self.alpha_lf * float(x) + (1 - self.alpha_lf) * self.ema_slow

        hf = float(x) - self.ema_fast
        mf = self.ema_fast - self.ema_slow
        lf = self.ema_slow

        return hf, mf, lf

    def reset(self):
        """Reset online state."""
        self.ema_fast = 0.0
        self.ema_mid = 0.0
        self.ema_slow = 0.0
        self._initialized = False


# ===========================================================================
# Factory
# ===========================================================================

def create_decomposer(hf_span: int = 2, mf_span: int = 10, lf_span: int = 50) -> CausalDecomposer:
    """Factory with custom spans."""
    return CausalDecomposer(CausalWaveletConfig(hf_span=hf_span, mf_span=mf_span, lf_span=lf_span))
