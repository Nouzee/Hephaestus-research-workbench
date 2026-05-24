"""
Hephaestus Training Framework — adapted from ExperimentC's battle-tested pipeline.

Feature layout (BTC L2):
    raw[:, 0]    = Time (ts_ms)
    raw[:, 1]    = TimeDiff (log1p duration_ms)
    raw[:, 2:]   = LOB features (variable count)

Training loop, metrics, feature ablation, and model orchestration.
"""
import json
import os
import random
import importlib
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


# ────────────────────────────────────────────
# Window Dataset
# ────────────────────────────────────────────
class WindowDataset(TorchDataset):
    """
    Returns:
        x_feat:   (1, T, n_features)
        x_dt:     (T, 1)
        y:        scalar float32, already scaled by target_scale
        end_idx:  concat row index corresponding to prediction target
    """

    def __init__(self, X: np.ndarray, dt: np.ndarray, y: np.ndarray, T: int):
        assert X.ndim == 2
        assert dt.ndim == 2
        assert len(X) == len(dt) == len(y)
        assert len(X) >= T

        self.T = int(T)

        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.dt = torch.from_numpy(dt.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

        y_np = np.asarray(y, dtype=np.float32)
        valid_end = np.where(np.isfinite(y_np))[0]
        valid_end = valid_end[valid_end >= (self.T - 1)]
        self.valid_end = torch.from_numpy(valid_end.astype(np.int64, copy=False))

    def __len__(self):
        return self.valid_end.numel()

    def __getitem__(self, i):
        end = int(self.valid_end[i].item())
        start = end - self.T + 1

        xw_feat = self.X[start:end + 1].unsqueeze(0)  # (1, T, n_feat)
        xw_dt = self.dt[start:end + 1]                 # (T, 1)
        yw = self.y[end]

        return xw_feat, xw_dt, yw, torch.tensor(end, dtype=torch.int64)


# ────────────────────────────────────────────
# Seeding
# ────────────────────────────────────────────
def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ────────────────────────────────────────────
# Model builder (dynamic import)
# ────────────────────────────────────────────
def build_model(model_name: str, module_name: str = "models.model_zoo"):
    models = importlib.import_module(module_name)
    return getattr(models, model_name)()


# ────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────
def log_line(log_path: Optional[str], msg: str):
    print(msg)
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def save_json(obj, path: Optional[str]):
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────
# Feature ablation (zero-masking)
# ────────────────────────────────────────────
# BTC feature groups (flexible, configured per-run)
# Default grouping:
#   group 0: price features (best_ask, best_bid, mid_px, micro_px, trade_px)
#   group 1: size/volume features (ask_sz, bid_sz, ask_depth_5, bid_depth_5, total_depth, trade_sz)
#   group 2: derived features (spread, spread_bps, imbalance, signed_imbalance, duration_ms)
#   group 3: side feature (trade_side)
#   group 4: total ask/bid depth (total_ask_depth, total_bid_depth)

VALID_FEATURE_SETS = {"PV", "PV+N", "PV+A", "PV+NA", "PV+NA+DT", "ALL"}


def safe_feature_name(feature_set: str) -> str:
    return str(feature_set).replace("+", "_").replace("/", "_").replace(" ", "")


def normalize_feature_set(feature_set: str) -> str:
    fs = str(feature_set).strip().upper()
    aliases = {
        "PVNA": "PV+NA",
        "PVNADT": "PV+NA+DT",
        "PV+N+A": "PV+NA",
        "FULL": "PV+NA+DT",
    }
    fs = aliases.get(fs, fs)
    if fs not in VALID_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set}. Valid={sorted(VALID_FEATURE_SETS)}")
    return fs


def apply_feature_ablation(X_all: np.ndarray, dt_all: np.ndarray,
                           feature_set: str, feature_groups: dict = None):
    """
    Zero-mask ablation. Model input shape stays the same; only data changes.

    feature_groups: dict mapping group name to column indices, e.g.:
        {"PV": [0,1,2,3], "N": [4,5,6], "A": [7,8], "DT": [9]}
    If None, uses a simple split: first 1/3 = PV, second 1/3 = N, last 1/3 = A.
    """
    feature_set = normalize_feature_set(feature_set)

    X = X_all.astype(np.float32, copy=True)
    dt = dt_all.astype(np.float32, copy=True)

    n_feat = X.shape[1]

    # Auto-group if not provided: split features into PV / N / A thirds
    if feature_groups is None:
        third = max(1, n_feat // 3)
        feature_groups = {
            "PV": list(range(0, third)),
            "N": list(range(third, min(2 * third, n_feat))),
            "A": list(range(min(2 * third, n_feat), n_feat)),
        }

    pv_cols = feature_groups.get("PV", [])
    n_cols = feature_groups.get("N", [])
    a_cols = feature_groups.get("A", [])

    keep_pv = True  # PV always kept
    keep_n = feature_set in {"PV+N", "PV+NA", "PV+NA+DT", "ALL"}
    keep_a = feature_set in {"PV+A", "PV+NA", "PV+NA+DT", "ALL"}
    keep_dt = feature_set in {"PV+NA+DT", "ALL"}

    if not keep_n:
        for c in n_cols:
            if 0 <= c < n_feat:
                X[:, c] = 0.0

    if not keep_a:
        for c in a_cols:
            if 0 <= c < n_feat:
                X[:, c] = 0.0

    if not keep_dt:
        dt[:, :] = 0.0

    return X, dt


# ────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────
def load_concat_data(data_path: str, y_path: str, target_scale: float,
                     feature_set: str = "ALL",
                     feature_groups: dict = None,
                     time_col: int = 0,
                     dt_col: int = 1,
                     feat_start_col: int = 2,
                     feat_end_col: int = None):
    """
    Load concatenated data parquet files.

    Returns:
        X_all:   (N, n_features) feature matrix
        dt_all:  (N, 1) time delta
        y_scaled: (N,) scaled target
        y_raw:   (N,) raw target
    """
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        raw = df.to_numpy(dtype=np.float32)
    else:
        raw = np.loadtxt(data_path, delimiter=",", dtype=np.float32, skiprows=1)

    # Extract columns
    dt_all = raw[:, dt_col:dt_col + 1]
    end_col = feat_end_col if feat_end_col is not None else raw.shape[1]
    X_all = raw[:, feat_start_col:end_col]

    X_all, dt_all = apply_feature_ablation(X_all, dt_all,
                                           feature_set=feature_set,
                                           feature_groups=feature_groups)

    # Labels
    if y_path.endswith(".parquet"):
        y_df = pd.read_parquet(y_path)
        y_raw = y_df.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)
    else:
        y_raw = np.load(y_path).reshape(-1).astype(np.float32)

    if len(X_all) != len(y_raw):
        raise ValueError(
            f"concat rows mismatch: X rows={len(X_all)}, y rows={len(y_raw)} | "
            f"data_path={data_path}, y_path={y_path}"
        )

    y_scaled = y_raw * float(target_scale)
    return X_all, dt_all, y_scaled, y_raw


# ────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────
def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return float("nan")
    x_std, y_std = np.std(x), np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return safe_pearsonr(rx, ry)


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst < 1e-12:
        return float("nan")
    return float(1.0 - sse / sst)


def safe_directional_accuracy(pred: np.ndarray, target: np.ndarray,
                              eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) > eps)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(target[mask])))


def regression_metrics(pred_scaled: np.ndarray, target_scaled: np.ndarray,
                       target_scale: float) -> dict:
    pred_scaled = np.asarray(pred_scaled, dtype=np.float64).reshape(-1)
    target_scaled = np.asarray(target_scaled, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred_scaled) & np.isfinite(target_scaled)
    pred_scaled = pred_scaled[mask]
    target_scaled = target_scaled[mask]

    pred_raw = pred_scaled / float(target_scale)
    target_raw = target_scaled / float(target_scale)

    diff_scaled = pred_scaled - target_scaled
    diff_raw = pred_raw - target_raw

    mae_scaled = float(np.mean(np.abs(diff_scaled))) if diff_scaled.size else float("nan")
    rmse_scaled = float(np.sqrt(np.mean(diff_scaled ** 2))) if diff_scaled.size else float("nan")
    mae_raw = float(np.mean(np.abs(diff_raw))) if diff_raw.size else float("nan")
    rmse_raw = float(np.sqrt(np.mean(diff_raw ** 2))) if diff_raw.size else float("nan")

    return {
        "test_pearson": safe_pearsonr(pred_scaled, target_scaled),
        "test_spearman": safe_spearmanr(pred_scaled, target_scaled),
        "test_mae_scaled": mae_scaled,
        "test_rmse_scaled": rmse_scaled,
        "test_mae_raw": mae_raw,
        "test_rmse_raw": rmse_raw,
        "directional_accuracy": safe_directional_accuracy(pred_raw, target_raw),
        "test_r2_scaled": safe_r2_score(target_scaled, pred_scaled),
    }


# ────────────────────────────────────────────
# Training / Evaluation loops
# ────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, (x_feat, x_dt, y, _end_idx) in enumerate(loader):
        x_feat = x_feat.to(device, dtype=torch.float32)
        x_dt = x_dt.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).reshape(-1)

        outputs = model(x_feat, x_dt).reshape(-1)
        loss = criterion(outputs, y)
        loss = loss / grad_accum_steps
        loss.backward()

        bs = y.size(0)
        diff = outputs - y
        # track unscaled loss for logging
        total_loss += (loss.item() * grad_accum_steps) * bs
        total_abs_err += torch.abs(diff).sum().item()
        total_sq_err += (diff ** 2).sum().item()
        total_samples += bs

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Flush remaining gradients at end of epoch
    if total_samples > 0 and ((batch_idx + 1) % grad_accum_steps != 0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(total_samples, 1)
    avg_mae = total_abs_err / max(total_samples, 1)
    avg_rmse = float(np.sqrt(total_sq_err / max(total_samples, 1)))
    return avg_loss, avg_mae, avg_rmse


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_feat, x_dt, y, _end_idx in loader:
            x_feat = x_feat.to(device, dtype=torch.float32)
            x_dt = x_dt.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).reshape(-1)

            outputs = model(x_feat, x_dt).reshape(-1)
            loss = criterion(outputs, y)

            bs = y.size(0)
            diff = outputs - y
            total_loss += loss.item() * bs
            total_abs_err += torch.abs(diff).sum().item()
            total_sq_err += (diff ** 2).sum().item()
            total_samples += bs

            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    avg_mae = total_abs_err / max(total_samples, 1)
    avg_rmse = float(np.sqrt(total_sq_err / max(total_samples, 1)))

    if all_preds:
        all_preds = np.concatenate(all_preds).astype(np.float32)
        all_targets = np.concatenate(all_targets).astype(np.float32)
    else:
        all_preds = np.array([], dtype=np.float32)
        all_targets = np.array([], dtype=np.float32)

    pearson = safe_pearsonr(all_preds, all_targets)
    spearman = safe_spearmanr(all_preds, all_targets)
    return avg_loss, avg_mae, avg_rmse, pearson, spearman


# ────────────────────────────────────────────
# Date assignment for daily metrics
# ────────────────────────────────────────────
def _date_by_end_index(end_indices: np.ndarray, day_offsets: List[dict]) -> np.ndarray:
    if not day_offsets:
        return np.array(["UNKNOWN"] * len(end_indices), dtype=object)
    end_rows = np.array([int(x["end_row"]) for x in day_offsets], dtype=np.int64)
    dates = np.array([str(x["date"]) for x in day_offsets], dtype=object)
    pos = np.searchsorted(end_rows, end_indices.astype(np.int64), side="right")
    pos = np.clip(pos, 0, len(dates) - 1)
    return dates[pos]


def build_daily_metrics(pred_df: pd.DataFrame, day_offsets: List[dict],
                        seed: int, feature_set: str, target_scale: float) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    pred_df = pred_df.copy()
    pred_df["date"] = _date_by_end_index(pred_df["end_index"].to_numpy(), day_offsets)

    rows = []
    for date, g in pred_df.groupby("date", sort=False):
        metrics = regression_metrics(
            pred_scaled=g["y_pred_scaled"].to_numpy(dtype=np.float64),
            target_scaled=g["y_true_scaled"].to_numpy(dtype=np.float64),
            target_scale=target_scale,
        )
        row = {
            "date": str(date),
            "feature_set": str(feature_set),
            "seed": int(seed),
            "samples": int(len(g)),
            "pearson_ic": metrics["test_pearson"],
            "spearman_ic": metrics["test_spearman"],
            "mae_scaled": metrics["test_mae_scaled"],
            "rmse_scaled": metrics["test_rmse_scaled"],
            "mae_raw": metrics["test_mae_raw"],
            "rmse_raw": metrics["test_rmse_raw"],
            "directional_accuracy": metrics["directional_accuracy"],
            "r2_scaled": metrics["test_r2_scaled"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────
# Main train + test pipeline
# ────────────────────────────────────────────
def train_and_test_pipeline(
    train_data_path,
    train_y_path,
    test_data_path,
    test_y_path,
    best_model_path,
    log_path,
    model_name,
    n_features=None,
    T=500,
    train_ratio=0.8,
    batch_size=16,
    num_workers=0,
    pin_memory=False,
    shuffle_train=True,
    epochs=30,
    learning_rate=1e-4,
    weight_decay=1e-4,
    grad_accum_steps=1,
    target_scale=1e4,
    device_str="auto",
    run_train=True,
    run_test=True,
    model_module_name="models.model_zoo",
    random_seed=None,
    feature_set="ALL",
    feature_groups=None,
    metrics_json_path=None,
    predictions_parquet_path=None,
    daily_metrics_csv_path=None,
    test_day_offsets=None,
    save_test_predictions=True,
    save_daily_metrics=True,
):
    feature_set = normalize_feature_set(feature_set)
    seed_everything(random_seed)

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    result = {
        "task": "regression",
        "device": str(device),
        "model_name": model_name,
        "model_module_name": model_module_name,
        "feature_set": feature_set,
        "seed": None if random_seed is None else int(random_seed),
        "best_model_path": best_model_path,
        "target_scale": float(target_scale),
    }

    loader_generator = None
    if random_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(random_seed))

    # ── Train / Validation ──
    if run_train:
        X_all, dt_all, y_scaled_all, _y_raw_all = load_concat_data(
            data_path=train_data_path, y_path=train_y_path,
            target_scale=target_scale, feature_set=feature_set,
            feature_groups=feature_groups,
        )

        split = int(len(y_scaled_all) * train_ratio)
        X_train, dt_train, y_train = X_all[:split], dt_all[:split], y_scaled_all[:split]
        X_val, dt_val, y_val = X_all[split:], dt_all[split:], y_scaled_all[split:]

        dataset_train = WindowDataset(X_train, dt_train, y_train, T)
        dataset_val = WindowDataset(X_val, dt_val, y_val, T)

        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            worker_init_fn=seed_worker if random_seed is not None else None,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )

        model = build_model(model_name=model_name, module_name=model_module_name).to(device)
        total_params = int(sum(p.numel() for p in model.parameters()))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)

        log_line(log_path, "=" * 80)
        log_line(log_path, "Hephaestus Regression Training")
        log_line(log_path, "=" * 80)
        log_line(log_path, f"feature_set: {feature_set}")
        log_line(log_path, f"seed: {random_seed}")
        log_line(log_path, f"device: {device}")
        log_line(log_path, f"model_name: {model_name}")
        log_line(log_path, f"target_scale: {target_scale}")
        log_line(log_path, f"total_params: {total_params:,}")
        log_line(log_path, f"raw_train_rows: {len(y_scaled_all)}")
        log_line(log_path, f"train_windows: {len(dataset_train)}")
        log_line(log_path, f"val_windows: {len(dataset_val)}")

        result["total_params"] = total_params
        result["raw_train_rows"] = int(len(y_scaled_all))
        result["train_windows"] = int(len(dataset_train))
        result["val_windows"] = int(len(dataset_val))

        best_val_loss = float("inf")
        best_val_mae = best_val_rmse = float("nan")
        best_val_pearson = best_val_spearman = float("nan")
        best_val_epoch = -1

        for epoch in range(epochs):
            train_loss, train_mae, train_rmse = train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, device=device,
                grad_accum_steps=grad_accum_steps,
            )
            val_loss, val_mae, val_rmse, val_pearson, val_spearman = eval_one_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device,
            )
            log_line(log_path,
                     f"Epoch {epoch + 1}/{epochs} | "
                     f"train_loss={train_loss:.6f} train_mae={train_mae:.6f} train_rmse={train_rmse:.6f} | "
                     f"val_loss={val_loss:.6f} val_mae={val_mae:.6f} val_rmse={val_rmse:.6f} "
                     f"val_pearson={val_pearson:.6f} val_spearman={val_spearman:.6f}")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), best_model_path)
                best_val_loss = float(val_loss)
                best_val_mae = float(val_mae)
                best_val_rmse = float(val_rmse)
                best_val_pearson = float(val_pearson)
                best_val_spearman = float(val_spearman)
                best_val_epoch = int(epoch + 1)
                log_line(log_path,
                         f"  -> New best model saved: val_loss={best_val_loss:.6f}, "
                         f"val_pearson={best_val_pearson:.6f}, epoch={best_val_epoch}")

        result["best_val_loss"] = best_val_loss
        result["best_val_mae_scaled"] = best_val_mae
        result["best_val_rmse_scaled"] = best_val_rmse
        result["best_val_pearson"] = best_val_pearson
        result["best_val_spearman"] = best_val_spearman
        result["best_val_epoch"] = best_val_epoch

        log_line(log_path, "")
        log_line(log_path, "Training complete.")
        log_line(log_path, f"Best val loss: {best_val_loss:.6f}")
        log_line(log_path, f"Best val MAE : {best_val_mae:.6f}")
        log_line(log_path, f"Best val RMSE: {best_val_rmse:.6f}")
        log_line(log_path, f"Best val Pearson : {best_val_pearson:.6f}")
        log_line(log_path, f"Best val Spearman: {best_val_spearman:.6f}")
        log_line(log_path, f"Best epoch: {best_val_epoch}")

    # ── Test ──
    if run_test:
        X_test, dt_test, y_test_scaled, _y_test_raw = load_concat_data(
            data_path=test_data_path, y_path=test_y_path,
            target_scale=target_scale, feature_set=feature_set,
            feature_groups=feature_groups,
        )

        dataset_test = WindowDataset(X_test, dt_test, y_test_scaled, T)
        test_loader = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )

        log_line(log_path, "")
        log_line(log_path, "=" * 80)
        log_line(log_path, "Hephaestus Regression Test")
        log_line(log_path, "=" * 80)
        log_line(log_path, f"feature_set: {feature_set}")
        log_line(log_path, f"seed: {random_seed}")
        log_line(log_path, f"test_windows: {len(dataset_test)}")

        model_test = build_model(model_name=model_name,
                                 module_name=model_module_name).to(device)
        state = torch.load(best_model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model_test.load_state_dict(state["state_dict"])
        else:
            model_test.load_state_dict(state)
        model_test.eval()

        criterion_test = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        all_preds, all_targets, all_end_indices = [], [], []

        with torch.no_grad():
            for x_feat, x_dt, y, end_idx in test_loader:
                x_feat = x_feat.to(device, dtype=torch.float32)
                x_dt = x_dt.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32).reshape(-1)

                outputs = model_test(x_feat, x_dt).reshape(-1)
                loss = criterion_test(outputs, y)

                bs = y.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
                all_end_indices.append(end_idx.cpu().numpy())

        if total_samples <= 0:
            raise RuntimeError("No valid test samples found.")

        all_preds = np.concatenate(all_preds).astype(np.float32)
        all_targets = np.concatenate(all_targets).astype(np.float32)
        all_end_indices = np.concatenate(all_end_indices).astype(np.int64)

        test_loss = float(total_loss / max(total_samples, 1))
        metrics = regression_metrics(pred_scaled=all_preds,
                                     target_scaled=all_targets,
                                     target_scale=target_scale)

        result.update(metrics)
        result["test_loss"] = test_loss
        result["test_samples"] = int(total_samples)
        result["pearson_ic"] = result["test_pearson"]
        result["spearman_ic"] = result["test_spearman"]
        result["mae"] = result["test_mae_scaled"]
        result["rmse"] = result["test_rmse_scaled"]
        result["mae_raw"] = result["test_mae_raw"]
        result["rmse_raw"] = result["test_rmse_raw"]

        log_line(log_path, "")
        log_line(log_path, "=" * 80)
        log_line(log_path, "Test Results")
        log_line(log_path, "=" * 80)
        log_line(log_path, f"Feature Set: {feature_set}")
        log_line(log_path, f"Seed: {random_seed}")
        log_line(log_path, f"Test Loss         : {test_loss:.6f}")
        log_line(log_path, f"Pearson IC        : {result['pearson_ic']:.6f}")
        log_line(log_path, f"Spearman IC       : {result['spearman_ic']:.6f}")
        log_line(log_path, f"MAE scaled        : {result['mae']:.6f}")
        log_line(log_path, f"RMSE scaled       : {result['rmse']:.6f}")
        log_line(log_path, f"MAE raw           : {result['mae_raw']:.10f}")
        log_line(log_path, f"RMSE raw          : {result['rmse_raw']:.10f}")
        log_line(log_path, f"Directional Acc   : {result['directional_accuracy']:.6f}")
        log_line(log_path, f"R2 scaled         : {result['test_r2_scaled']:.6f}")
        log_line(log_path, f"Samples           : {total_samples}")
        log_line(log_path, "=" * 80)

        pred_df = pd.DataFrame({
            "end_index": all_end_indices,
            "y_true_scaled": all_targets,
            "y_pred_scaled": all_preds,
            "y_true_raw": all_targets / float(target_scale),
            "y_pred_raw": all_preds / float(target_scale),
            "feature_set": feature_set,
            "seed": -1 if random_seed is None else int(random_seed),
        })

        if save_test_predictions and predictions_parquet_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(predictions_parquet_path)),
                        exist_ok=True)
            pred_df.to_parquet(predictions_parquet_path, index=False)
            result["predictions_parquet_path"] = predictions_parquet_path

        if save_daily_metrics and daily_metrics_csv_path is not None and test_day_offsets:
            daily_df = build_daily_metrics(
                pred_df=pred_df, day_offsets=test_day_offsets,
                seed=-1 if random_seed is None else int(random_seed),
                feature_set=feature_set, target_scale=target_scale,
            )
            if not daily_df.empty:
                os.makedirs(os.path.dirname(os.path.abspath(daily_metrics_csv_path)),
                            exist_ok=True)
                daily_df.to_csv(daily_metrics_csv_path, index=False)
                result["daily_metrics_csv_path"] = daily_metrics_csv_path
                result["num_test_days"] = int(daily_df["date"].nunique())

    save_json(result, metrics_json_path)
    return result
