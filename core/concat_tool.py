"""
Daily file concatenator — merges daily LOB parquets and label npy files
into single train/test concat files. Manifest-based for reproducibility.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def concat_lob_from_manifest(manifest: List[Dict], out_lob_parquet: str) -> int:
    """
    Concatenate daily LOB parquet files into one big parquet.

    Returns total row count.
    """
    if not manifest:
        raise ValueError("[LOB] empty manifest")

    header = None
    frames = []
    total_rows = 0

    for item in manifest:
        p = item["lob_parquet"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"[LOB] missing file: {p}")

        df = pd.read_parquet(p)
        if header is None:
            header = list(df.columns)
        else:
            if list(df.columns) != header:
                raise ValueError(f"[LOB] columns mismatch: {p} | "
                                 f"expected={header[:3]}..., got={list(df.columns)[:3]}...")

        expected_rows = int(item.get("lob_rows", len(df)))
        if len(df) != expected_rows:
            raise ValueError(f"[LOB] manifest rows mismatch: {p}, "
                             f"manifest={expected_rows}, actual={len(df)}")

        frames.append(df)
        total_rows += len(df)

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    _ensure_parent_dir(out_lob_parquet)
    df_all.to_parquet(out_lob_parquet, index=False)

    print(f"[ok] LOB concat: days={len(manifest)} rows={total_rows} -> {out_lob_parquet}")
    return total_rows


def concat_label_from_manifest(manifest: List[Dict], out_label_parquet: str) -> Tuple[int, Dict]:
    """
    Concatenate daily label npy files into a single parquet.

    Returns (total_rows, label_stats).
    """
    if not manifest:
        raise ValueError("[LABEL] empty manifest")

    parts = []
    total_rows = 0
    cols = None

    for item in manifest:
        p = item["label_npy"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"[LABEL] missing file: {p}")

        arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if cols is None:
            cols = arr.shape[1]
        elif arr.shape[1] != cols:
            raise ValueError(f"[LABEL] cols mismatch: {p}")

        expected_rows = int(item.get("label_rows", arr.shape[0]))
        if arr.shape[0] != expected_rows:
            raise ValueError(f"[LABEL] manifest rows mismatch: {p}, "
                             f"manifest={expected_rows}, actual={arr.shape[0]}")

        parts.append(arr)
        total_rows += arr.shape[0]

    y_all = np.vstack(parts).astype(np.float32, copy=False)
    _ensure_parent_dir(out_label_parquet)
    pd.DataFrame({"label": y_all.reshape(-1)}).to_parquet(out_label_parquet, index=False)

    y_flat = y_all.reshape(-1).astype(np.float64, copy=False)
    valid_mask = np.isfinite(y_flat)
    y_valid = y_flat[valid_mask]

    total = int(y_flat.size)
    valid_n = int(y_valid.size)
    nan_n = int(total - valid_n)
    nan_ratio = nan_n / total if total > 0 else float("nan")

    print(f"[ok] LABEL concat: days={len(manifest)} rows={total_rows} -> {out_label_parquet}")
    print(f"      total={total}, valid={valid_n}, nan={nan_n}, nan_ratio={nan_ratio:.6%}")

    if valid_n > 0:
        stats = {
            "total_count": total, "valid_count": valid_n,
            "nan_count": nan_n, "nan_ratio": float(nan_ratio),
            "mean": float(np.mean(y_valid)), "std": float(np.std(y_valid)),
            "min": float(np.min(y_valid)), "max": float(np.max(y_valid)),
            "q01": float(np.quantile(y_valid, 0.01)),
            "q05": float(np.quantile(y_valid, 0.05)),
            "q25": float(np.quantile(y_valid, 0.25)),
            "q50": float(np.quantile(y_valid, 0.50)),
            "q75": float(np.quantile(y_valid, 0.75)),
            "q95": float(np.quantile(y_valid, 0.95)),
            "q99": float(np.quantile(y_valid, 0.99)),
        }
    else:
        stats = {"total_count": total, "valid_count": 0, "nan_count": nan_n}
        print("      [warn] no valid label values")

    return total_rows, stats


def build_day_offsets(manifest: List[Dict]) -> List[Dict]:
    """
    Build concat row ranges for daily-block metrics.
    start_row is inclusive, end_row is exclusive.
    """
    offsets = []
    cur = 0
    for item in manifest:
        rows = int(item["lob_rows"])
        offsets.append({
            "date": str(item["date"]),
            "start_row": int(cur),
            "end_row": int(cur + rows),
            "rows": int(rows),
        })
        cur += rows
    return offsets
