"""
Hephaestus Full Training Pipeline — BTC L2 Regression with 高频项目架构.

Uses pre-built events_features.parquet (from 高频项目) as data source.
Model: HybridTransformerLSTM (from 高频项目, adapted for regression).
Training loop: from ExperimentC (WindowDataset / metrics / feature ablation).

Usage:
    python run_pipeline.py                    # full run
    python run_pipeline.py --skip-data        # reuse existing concat files
    python run_pipeline.py --epochs 2         # quick smoke test
    python run_pipeline.py --rebuild          # force rebuild data
"""
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import ctypes
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
except Exception:
    pass

from core.data_generator import (
    load_events_parquet,
    split_by_days,
    run_daily_generation,
    FEATURE_COLS,
)
from core.concat_tool import (
    concat_lob_from_manifest,
    concat_label_from_manifest,
    build_day_offsets,
)
from core.training_frame import (
    train_and_test_pipeline,
    safe_feature_name,
    save_json,
    load_json,
)


# ═══════════════════════════════════════════════════════════════
# Config — change these for each experiment
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_NAME = "hephaestus_btc_regression"

# Data source (高频项目 pre-built parquet)
EVENTS_PARQUET = r"C:\Users\ZaneLaw\Desktop\Zane\高频项目\刘子睿_HFT_Backtest\data\events_features.parquet"

# Chronological train/test split (70% train, 30% test)
TRAIN_RATIO_SPLIT = 0.7

# Label definition
LABEL_K = 500        # lookback ticks
LABEL_H = 10.0       # forward horizon in seconds
LABEL_TAG = f"{LABEL_K}-{LABEL_H}"
USE_LOG_RETURN = False
TARGET_SCALE = 100           # BTC return ~1e-5/s x 10s = 1e-4, x100 = 0.01 range

# Feature sets for ablation
FEATURE_SETS = ["PV+NA+DT"]

# Seeds for repeated runs
SEEDS = [0, 1, 2]

# Model (from 高频项目, regression-adapted)
MODEL_NAME = "HybridTransformerLSTM"
MODEL_MODULE_NAME = "models.model_zoo"

# Training params
T = LABEL_K                         # window size = label lookback
TRAIN_RATIO = 0.8                   # train/val split within training set
BATCH_SIZE = 256
GRAD_ACCUM_STEPS = 4     # effective batch = 256 x 4 = 1024
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE_STR = "auto"

# Pipeline switches
REBUILD_DATA = False
KEEP_TEMP_DAILY = False
RUN_TRAIN = True
RUN_TEST = True
SKIP_FINISHED = True
STOP_ON_ERROR = False
SAVE_TEST_PREDICTIONS = True
SAVE_DAILY_METRICS = False

# ═══════════════════════════════════════════════════════════════
# Paths (auto-generated from experiment name)
# ═══════════════════════════════════════════════════════════════
PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = str(PACKAGE_ROOT / "output" / EXPERIMENT_NAME / LABEL_TAG)
TMP_DIR        = os.path.join(PIPELINE_ROOT, "tmp_daily")
CONCAT_DIR     = os.path.join(PIPELINE_ROOT, "concat")
RUNS_DIR       = os.path.join(PIPELINE_ROOT, "runs")
SUMMARY_DIR    = os.path.join(PIPELINE_ROOT, "summary")
MASTER_LOG_DIR = os.path.join(PIPELINE_ROOT, "logs")

TRAIN_LOB_CONCAT   = os.path.join(CONCAT_DIR, "train_lob_concat.parquet")
TEST_LOB_CONCAT    = os.path.join(CONCAT_DIR, "test_lob_concat.parquet")
TRAIN_LABEL_CONCAT = os.path.join(CONCAT_DIR, f"train_label_concat_{LABEL_TAG}.parquet")
TEST_LABEL_CONCAT  = os.path.join(CONCAT_DIR, f"test_label_concat_{LABEL_TAG}.parquet")

MASTER_LOG_PATH       = os.path.join(MASTER_LOG_DIR, "master_log.txt")
CONFIG_PATH           = os.path.join(SUMMARY_DIR, "config.json")
TRAIN_MANIFEST_PATH   = os.path.join(SUMMARY_DIR, "train_manifest.json")
TEST_MANIFEST_PATH    = os.path.join(SUMMARY_DIR, "test_manifest.json")
TEST_DAY_OFFSETS_PATH = os.path.join(SUMMARY_DIR, "test_day_offsets.json")
ALL_RUNS_CSV          = os.path.join(SUMMARY_DIR, "all_runs.csv")
SUMMARY_CSV           = os.path.join(SUMMARY_DIR, "summary.csv")

# Feature groups for ablation (indices into FEATURE_COLS)
# 高频项目's FEATURE_COLS:
#   [0] mid_px, [1] spread, [2] imbalance, [3] total_depth,
#   [4] bid_px, [5] ask_px, [6] bid_sz, [7] ask_sz,
#   [8] signed_imbalance, [9] abs_imbalance, [10] spread_ticks
FEATURE_GROUPS = {
    "PV": [0, 3, 4, 5, 6, 7],      # price + volume
    "N":  [8, 9],                     # flow / imbalance
    "A":  [1, 2, 10],                # spread / microstructure
}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def ensure_dirs():
    for d in [PIPELINE_ROOT, TMP_DIR, CONCAT_DIR, RUNS_DIR, SUMMARY_DIR, MASTER_LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(os.path.abspath(MASTER_LOG_PATH)), exist_ok=True)
    with open(MASTER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# Data stage
# ═══════════════════════════════════════════════════════════════
def data_stage_needed() -> bool:
    required = [TRAIN_LOB_CONCAT, TEST_LOB_CONCAT,
                TRAIN_LABEL_CONCAT, TEST_LABEL_CONCAT]
    if REBUILD_DATA:
        return True
    return not all(os.path.exists(p) for p in required)


def run_data_stage() -> List[dict]:
    """Load events_features.parquet, split by date, generate daily files, concat."""
    log("Stage 1+2: Loading events_features.parquet + generating daily files")

    # Load full month
    df_all = load_events_parquet(EVENTS_PARQUET)
    log(f"Loaded {len(df_all):,} events from {EVENTS_PARQUET}")
    log(f"Columns: {FEATURE_COLS}")

    # Chronological 70/30 split (no date assumptions)
    split_idx = int(len(df_all) * TRAIN_RATIO_SPLIT)
    df_train = df_all.iloc[:split_idx].copy()
    df_test  = df_all.iloc[split_idx:].copy()
    log(f"Train: {len(df_train):,} events, Test: {len(df_test):,} events")

    if len(df_train) == 0 or len(df_test) == 0:
        raise RuntimeError("No data after split")

    # Split each into daily chunks
    train_days = split_by_days(df_train)
    test_days  = split_by_days(df_test)
    log(f"Train days: {len(train_days)}, Test days: {len(test_days)}")

    # Generate daily parquets (train normalizes with rolling stats; test inherits them)
    stats_history: List[Tuple] = []

    train_manifest, stats_history = run_daily_generation(
        daily_chunks=train_days,
        tmp_dir=TMP_DIR,
        stats_history=stats_history,
        label_k=LABEL_K,
        label_h=LABEL_H,
        use_log_return=USE_LOG_RETURN,
        split_label="train",
    )

    test_manifest, _ = run_daily_generation(
        daily_chunks=test_days,
        tmp_dir=TMP_DIR,
        stats_history=stats_history,
        label_k=LABEL_K,
        label_h=LABEL_H,
        use_log_return=USE_LOG_RETURN,
        split_label="test",
    )

    # Concat daily -> single train/test parquets
    train_rows = concat_lob_from_manifest(train_manifest, TRAIN_LOB_CONCAT)
    concat_label_from_manifest(train_manifest, TRAIN_LABEL_CONCAT)
    test_rows  = concat_lob_from_manifest(test_manifest, TEST_LOB_CONCAT)
    concat_label_from_manifest(test_manifest, TEST_LABEL_CONCAT)

    test_day_offsets = build_day_offsets(test_manifest)

    save_json(train_manifest, TRAIN_MANIFEST_PATH)
    save_json(test_manifest, TEST_MANIFEST_PATH)
    save_json(test_day_offsets, TEST_DAY_OFFSETS_PATH)

    log(f"Data stage done: train={train_rows} rows, test={test_rows} rows")
    return test_day_offsets


# ═══════════════════════════════════════════════════════════════
# Train + test one config
# ═══════════════════════════════════════════════════════════════
def run_one_feature_seed(feature_set: str, seed: int,
                         test_day_offsets: List[dict]) -> dict:
    safe_fs = safe_feature_name(feature_set)
    run_dir = os.path.join(RUNS_DIR, safe_fs, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    best_model_path  = os.path.join(run_dir, "best_model.pt")
    log_path          = os.path.join(run_dir, "training_log.txt")
    metrics_json_path = os.path.join(run_dir, "test_metrics.json")
    predictions_path  = os.path.join(run_dir, "test_predictions.parquet")
    daily_csv_path    = os.path.join(run_dir, "daily_metrics.csv")

    if SKIP_FINISHED and os.path.exists(metrics_json_path):
        log(f"[skip] fs={feature_set}, seed={seed}, already done")
        return load_json(metrics_json_path)

    log(f"Stage 3/4: fs={feature_set}, seed={seed}")

    result = train_and_test_pipeline(
        train_data_path=TRAIN_LOB_CONCAT,
        train_y_path=TRAIN_LABEL_CONCAT,
        test_data_path=TEST_LOB_CONCAT,
        test_y_path=TEST_LABEL_CONCAT,
        best_model_path=best_model_path,
        log_path=log_path,
        model_name=MODEL_NAME,
        T=T,
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle_train=SHUFFLE_TRAIN,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        target_scale=TARGET_SCALE,
        device_str=DEVICE_STR,
        run_train=RUN_TRAIN,
        run_test=RUN_TEST,
        model_module_name=MODEL_MODULE_NAME,
        random_seed=int(seed),
        feature_set=feature_set,
        feature_groups=FEATURE_GROUPS,
        metrics_json_path=metrics_json_path,
        predictions_parquet_path=predictions_path,
        daily_metrics_csv_path=daily_csv_path,
        test_day_offsets=test_day_offsets,
        save_test_predictions=SAVE_TEST_PREDICTIONS,
        save_daily_metrics=SAVE_DAILY_METRICS,
    )

    result.update({
        "experiment_name": EXPERIMENT_NAME,
        "symbol": "BTC-USDT-SWAP",
        "label_tag": LABEL_TAG,
        "feature_set": feature_set,
        "seed": int(seed),
        "run_dir": run_dir,
    })
    save_json(result, metrics_json_path)

    log(f"Done: fs={feature_set}, seed={seed} | "
        f"pearson={result.get('pearson_ic'):.4f}, "
        f"spearman={result.get('spearman_ic'):.4f}, "
        f"rmse={result.get('rmse'):.4f}, "
        f"dir_acc={result.get('directional_accuracy'):.4f}")
    return result


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
REPORT_METRICS = [
    ("pearson_ic", "Pearson IC"),
    ("spearman_ic", "Spearman IC"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("directional_accuracy", "Directional Acc"),
]


def summarize_results(rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(ALL_RUNS_CSV, index=False)

    # Only summarize successful runs
    ok_mask = df["status"] != "failed" if "status" in df.columns else pd.Series(True, index=df.index)
    if "pearson_ic" in df.columns:
        ok_mask = ok_mask & df["pearson_ic"].notna()
    df_ok = df[ok_mask]

    if df_ok.empty:
        log("[summary] No successful runs.")
        return

    summary_rows = []
    for fs in FEATURE_SETS:
        g = df_ok[df_ok["feature_set"] == fs]
        if g.empty:
            continue
        row = {"feature_set": fs, "num_seeds": len(g)}
        for key, name in REPORT_METRICS:
            if key not in g.columns:
                continue
            vals = pd.to_numeric(g[key], errors="coerce").dropna()
            row[f"{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{key}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log("[summary]\n" + summary_df.to_string(index=False))


def write_config():
    save_json({
        "experiment_name": EXPERIMENT_NAME,
        "events_parquet": EVENTS_PARQUET,
        "train_ratio_split": TRAIN_RATIO_SPLIT,
        "label_k": LABEL_K, "label_h": LABEL_H,
        "target_scale": TARGET_SCALE,
        "feature_sets": FEATURE_SETS, "seeds": SEEDS,
        "feature_cols": FEATURE_COLS,
        "feature_groups": {k: [FEATURE_COLS[i] for i in v] for k, v in FEATURE_GROUPS.items()},
        "model_name": MODEL_NAME, "model_module": MODEL_MODULE_NAME,
        "T": T, "train_ratio": TRAIN_RATIO,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
    }, CONFIG_PATH)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hephaestus BTC Training Pipeline")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    global REBUILD_DATA, EPOCHS
    if args.skip_data:
        REBUILD_DATA = False
    if args.rebuild:
        REBUILD_DATA = True
    if args.epochs:
        EPOCHS = args.epochs

    ensure_dirs()
    write_config()

    log("=" * 60)
    log("Hephaestus BTC Training Pipeline (高频项目 data + model)")
    log("=" * 60)
    log(f"Data: {EVENTS_PARQUET}")
    log(f"Train ratio: {TRAIN_RATIO_SPLIT}")
    log(f"Model: {MODEL_NAME}, Features: {len(FEATURE_COLS)}, K={LABEL_K}, H={LABEL_H}s")
    log(f"Feature sets: {FEATURE_SETS}, Seeds: {SEEDS}, Epochs: {EPOCHS}")

    all_rows = []

    try:
        if data_stage_needed():
            log("Running data stage...")
            test_day_offsets = run_data_stage()
        else:
            log("Reusing existing concat files.")
            test_day_offsets = load_json(TEST_DAY_OFFSETS_PATH) if os.path.exists(TEST_DAY_OFFSETS_PATH) else []

        for fs in FEATURE_SETS:
            for seed in SEEDS:
                try:
                    row = run_one_feature_seed(fs, int(seed), test_day_offsets)
                    all_rows.append(row)
                except Exception as e:
                    log(f"[FAIL] fs={fs}, seed={seed}: {e}")
                    log(traceback.format_exc())
                    all_rows.append({"feature_set": fs, "seed": seed, "status": "failed", "error": str(e)})
                    if STOP_ON_ERROR:
                        raise

        summarize_results(all_rows)

        if not KEEP_TEMP_DAILY and os.path.isdir(TMP_DIR):
            shutil.rmtree(TMP_DIR)

        log("=" * 60)
        log("Pipeline finished.")
        log(f"Output: {PIPELINE_ROOT}")
        log("=" * 60)

    except Exception:
        log("=" * 60)
        log("Pipeline FAILED")
        log(traceback.format_exc())
        log("=" * 60)
        raise


if __name__ == "__main__":
    main()
