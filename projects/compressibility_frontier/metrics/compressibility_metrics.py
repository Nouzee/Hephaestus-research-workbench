"""
Compressibility Metrics — four measures of market structural density.

  A. Reconstruction Residual:  ||X - Dα||_F / ||X||_F
      How much of the market is NOT captured by the dictionary?
      Lower = more compressible.

  B. Effective Rank:  exp(-Σ p_i log p_i)  where p_i = σ_i / Σσ_j
      How many independent dimensions does the market actually use?
      Lower = more compressible (fewer active modes).

  C. Atom Usage Entropy:  -Σ f_k log f_k  where f_k = usage frequency of atom k
      Are atoms used evenly (high entropy) or concentrated (low entropy)?
      Lower = market has clearer "grammar."

  D. Temporal Redundancy:  1 - mean(|X_t - X_{t-1}|) / mean(|X_t|)
      How much does the market repeat itself across consecutive batches?
      Higher = more redundant = more compressible.

All four are normalized to [0, 1] where 0 = incompressible noise, 1 = perfectly structured.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def reconstruction_residual(X: np.ndarray, D: np.ndarray, alpha: np.ndarray) -> float:
    """
    A. Relative reconstruction error of dictionary representation.

    residual = ||X - αD||_F / ||X||_F

    Returns float in [0, ∞). Lower = better compression.
    Normalized: compressibility = 1 / (1 + residual) → [0, 1]
    """
    X_recon = alpha @ D
    residual = float(np.linalg.norm(X - X_recon, ord='fro')
                     / max(np.linalg.norm(X, ord='fro'), 1e-12))
    return residual


def effective_rank(X: np.ndarray) -> float:
    """
    B. Effective rank via spectral entropy.

    σ = svd(X)
    p_i = σ_i / Σσ_j
    H = -Σ p_i log(p_i)
    effective_rank = exp(H)

    Returns float. Lower = fewer active dimensions = more compressible.
    """
    try:
        sv = linalg.svdvals(X.astype(np.float64))
    except Exception:
        sv = np.ones(min(X.shape))

    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 1.0

    p = sv / sv.sum()
    p = p[p > 0]

    if len(p) <= 1:
        return 1.0

    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def atom_usage_entropy(alpha: np.ndarray, threshold: float = 1e-8) -> float:
    """
    C. Entropy of atom usage distribution.

    f_k = fraction of samples where atom k is active (|α| > threshold)
    H = -Σ f_k log(f_k)
    Normalized: H / log(K) → [0, 1]

    Returns float in [0, 1]. Lower = atoms used more selectively.
    """
    K = alpha.shape[1]
    if K <= 1:
        return 0.0

    usage = np.mean(np.abs(alpha) > threshold, axis=0)
    usage = usage[usage > 0]

    if len(usage) <= 1:
        return 0.0

    H = -np.sum(usage * np.log(usage))
    H_max = np.log(K)
    return float(H / H_max)


def temporal_redundancy(X: np.ndarray) -> float:
    """
    D. Temporal redundancy: how much does the market repeat itself?

    redundancy = 1 - mean(|ΔX|) / (mean(|X|) + ε)

    Higher = adjacent batches are more similar = more compressible.
    Normalized to [0, 1] via sigmoid.
    """
    if X.shape[0] < 2:
        return 0.0

    delta_mean = float(np.mean(np.abs(np.diff(X, axis=0))))
    level_mean = float(np.mean(np.abs(X)))

    if level_mean < 1e-12:
        return 0.0

    raw = 1.0 - delta_mean / level_mean
    # Sigmoid normalize: center at 0, slope high
    return float(1.0 / (1.0 + np.exp(-raw * 5)))


def compressibility_summary(
    X: np.ndarray,
    D: np.ndarray,
    alpha: np.ndarray,
) -> dict:
    """
    Compute all four compressibility metrics for a data segment.

    Returns dict with raw values + normalized scores [0,1].
    """
    resid = reconstruction_residual(X, D, alpha)
    erank = effective_rank(X)
    entropy = atom_usage_entropy(alpha)
    redundancy = temporal_redundancy(X)

    # Normalize residual to [0, 1]: compressibility_score = 1 / (1 + residual)
    resid_norm = 1.0 / (1.0 + resid)

    # Normalize effective rank: score = 1 / (1 + log(erank))
    erank_norm = 1.0 / (1.0 + np.log(max(erank, 1.01)))

    # Composite: average of all four normalized scores
    composite = float(np.mean([resid_norm, erank_norm, 1.0 - entropy, redundancy]))

    return {
        "reconstruction_residual": resid,
        "effective_rank": erank,
        "atom_entropy": entropy,
        "temporal_redundancy": redundancy,
        "residual_score": resid_norm,
        "rank_score": erank_norm,
        "entropy_score": 1.0 - entropy,
        "redundancy_score": redundancy,
        "composite_compressibility": composite,
    }
