"""
Dictionary Trainer — MiniBatchDictionaryLearning 离线训练

Task 2 of Dictionary Learning pipeline.
Takes the N×M feature matrix X from MatrixBuilder, runs sparse
dictionary learning to discover K market-microstructure "atoms",
and exports the dictionary D for online inference.

Key math:
    X ≈ α @ D      (N×M ≈ N×K @ K×M)
    D ∈ ℝ^{K×M}    dictionary atoms (market regimes)
    α ∈ ℝ^{N×K}    sparse coefficients (alpha signals)

Uses sklearn.decomposition.MiniBatchDictionaryLearning for
memory-efficient training on 3.9M+ samples.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DictTrainerConfig:
    """Configuration for dictionary learning training."""

    # Dictionary
    n_components: int = 3          # number of atoms (market regimes)
    alpha: float = 1.0             # L1 sparsity penalty (higher → sparser)

    # Training
    batch_size: int = 2048         # samples per partial_fit call
    n_epochs: int = 5              # full passes over the dataset
    max_iter: int = 1000           # sklearn internal max_iter (fallback)

    # Evaluation
    val_size: int = 50_000         # subset for per-epoch error evaluation
    eval_fraction: float = 0.05    # fraction of X for final reconstruction eval

    # Export
    cache_dir: str = ""            # where to save dict_atoms_{K}.npy
    seed: int = 42

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = str(
                Path(__file__).resolve().parent / "cache"
            )


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class DictTrainer:
    """
    Train a sparse dictionary on the HFT feature matrix via
    MiniBatchDictionaryLearning.

    Usage
    -----
    >>> trainer = DictTrainer()
    >>> trainer.fit(X)
    >>> D, alpha = trainer.result()
    >>> print(D.shape)   # (3, 9)
    """

    def __init__(self, config: Optional[DictTrainerConfig] = None):
        self.config = config or DictTrainerConfig()
        self.model_: Optional[MiniBatchDictionaryLearning] = None
        self.D_: Optional[np.ndarray] = None          # K × M
        self.alpha_: Optional[np.ndarray] = None      # N × K (on eval subset)
        self.history_: dict = {}                       # per-epoch metrics

    # ── Core training ────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "DictTrainer":
        """
        Train the dictionary on X via mini-batch partial_fit loops.

        Parameters
        ----------
        X : (N, M) np.ndarray, float32 or float64
            Standardized feature matrix from MatrixBuilder.
        """
        cfg = self.config
        N, M = X.shape
        K = cfg.n_components

        print(f"{'═'*60}")
        print(f"Dictionary Trainer — MiniBatchDictionaryLearning")
        print(f"{'═'*60}")
        print(f"  Samples:       {N:,}")
        print(f"  Features:      {M}")
        print(f"  Atoms (K):     {K}")
        print(f"  L1 α:          {cfg.alpha}")
        print(f"  Batch size:    {cfg.batch_size:,}")
        print(f"  Epochs:        {cfg.n_epochs}")
        print(f"  Val size:      {cfg.val_size:,}")
        print(f"{'─'*60}")

        # Validation subset (fixed, for consistent per-epoch tracking)
        rng = np.random.RandomState(cfg.seed)
        val_idx = rng.choice(N, min(cfg.val_size, N), replace=False)
        X_val = X[val_idx]

        # Init learner
        self.model_ = MiniBatchDictionaryLearning(
            n_components=K,
            alpha=cfg.alpha,
            batch_size=cfg.batch_size,
            max_iter=cfg.max_iter,
            transform_algorithm="lasso_lars",
            random_state=cfg.seed,
            n_jobs=-1,
        )

        total_start = time.perf_counter()
        n_batches_per_epoch = max(1, N // cfg.batch_size)

        for epoch in range(1, cfg.n_epochs + 1):
            epoch_start = time.perf_counter()

            # Shuffle and feed batches
            perm = rng.permutation(N)
            for b in range(n_batches_per_epoch):
                start = b * cfg.batch_size
                end = start + cfg.batch_size
                batch = X[perm[start:end]]
                self.model_.partial_fit(batch)

            # Evaluate on validation subset
            alpha_val = self.model_.transform(X_val)
            X_recon = alpha_val @ self.model_.components_
            recon_error = np.mean((X_val - X_recon) ** 2)

            # Sparsity: fraction of entries that are ~zero
            nonzero_frac = np.mean(np.abs(alpha_val) > 1e-8) * 100

            epoch_time = time.perf_counter() - epoch_start
            self.history_[epoch] = {
                "recon_error": float(recon_error),
                "sparsity_pct": float(nonzero_frac),
                "time_s": epoch_time,
            }
            print(f"  Epoch {epoch}/{cfg.n_epochs} | "
                  f"Recon Error: {recon_error:.6f} | "
                  f"Sparsity: {nonzero_frac:.1f}% nonzero | "
                  f"Time: {epoch_time:.1f}s")

        total_time = time.perf_counter() - total_start
        print(f"{'─'*60}")
        print(f"  Total training time: {total_time:.1f}s "
              f"({total_time/60:.1f} min)")

        # Store dictionary
        self.D_ = self.model_.components_.copy().astype(np.float32)
        print(f"  Dictionary D: {self.D_.shape}  ({self.D_.nbytes / 1024:.1f} KB)")
        print(f"{'═'*60}\n")

        return self

    # ── Final evaluation ─────────────────────────────────────────────

    def evaluate(self, X: np.ndarray) -> dict:
        """
        Compute final reconstruction error and sparsity on a subset of X.

        Returns dict with keys: recon_error, sparsity_pct, explained_variance.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cfg = self.config
        rng = np.random.RandomState(cfg.seed + 1)
        n_eval = max(1, int(len(X) * cfg.eval_fraction))
        idx = rng.choice(len(X), n_eval, replace=False)
        X_sub = X[idx]

        alpha_sub = self.model_.transform(X_sub)
        X_recon = alpha_sub @ self.model_.components_

        mse = np.mean((X_sub - X_recon) ** 2)
        var_total = np.var(X_sub)
        explained_var = 1.0 - mse / var_total if var_total > 0 else 0.0
        nonzero_frac = np.mean(np.abs(alpha_sub) > 1e-8) * 100

        # Atom norms and pairwise similarities
        D = self.model_.components_
        atom_norms = np.linalg.norm(D, axis=1)
        atom_dot = D @ D.T
        np.fill_diagonal(atom_dot, 0)
        max_similarity = np.max(np.abs(atom_dot)) if self.config.n_components > 1 else 0.0

        print(f"Final Evaluation ({n_eval:,} samples):")
        print(f"  Reconstruction MSE:  {mse:.6f}")
        print(f"  Explained Variance:  {explained_var:.4f}"
              f"{' [OK]' if explained_var > 0 else ' [LOW]'}")
        print(f"  Sparsity:            {nonzero_frac:.1f}% nonzero")
        print(f"  Atom norms:          {np.round(atom_norms, 4).tolist()}")
        print(f"  Max atom cosine sim: {max_similarity:.4f}"
              f"{' [well-separated]' if max_similarity < 0.5 else ' [WARNING: overlapping]'}")

        self.alpha_ = alpha_sub.astype(np.float32)

        return {
            "recon_error": float(mse),
            "explained_variance": float(explained_var),
            "sparsity_pct": float(nonzero_frac),
            "atom_norms": atom_norms.tolist(),
            "max_atom_similarity": float(max_similarity),
        }

    # ── Export ───────────────────────────────────────────────────────

    def export(self) -> str:
        """
        Save the dictionary D to cache directory as .npy.

        Returns the file path.
        """
        if self.D_ is None:
            raise RuntimeError("No dictionary to export. Call fit() first.")

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        K = self.config.n_components
        path = cache_dir / f"dict_atoms_{K}.npy"
        np.save(str(path), self.D_)
        print(f"[DictTrainer] Dictionary exported → {path}")
        return str(path)

    # ── Result ───────────────────────────────────────────────────────

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (D, alpha_eval) — the dictionary and sparse coefficients
        on the evaluation subset.
        """
        if self.D_ is None:
            raise RuntimeError("No results. Call fit() first.")
        if self.alpha_ is None:
            raise RuntimeError("No alpha. Call evaluate() first.")
        return self.D_, self.alpha_


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def create_trainer(**kwargs) -> DictTrainer:
    """Factory: create a DictTrainer with optional config overrides."""
    config = DictTrainerConfig(**kwargs)
    return DictTrainer(config)


# ═══════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from modules.dictionary.matrix_builder import MatrixBuilder

    print("Step 1 — Building feature matrix X ...")
    builder = MatrixBuilder()
    X, _ = builder.assemble()

    print(f"\nStep 2 — Training dictionary (K=3) ...")
    trainer = DictTrainer()
    trainer.fit(X)

    print("Step 3 — Final evaluation ...")
    trainer.evaluate(X)

    print("Step 4 — Exporting dictionary ...")
    trainer.export()
