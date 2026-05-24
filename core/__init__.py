"""
Hephaestus Core — Task Orchestrator, Logger, DataLoader, Training Framework
"""
from .orchestrator import TaskOrchestrator
from .logger import ExperimentLogger
from .data_loader import DataLoader
from .training_frame import (
    WindowDataset,
    train_and_test_pipeline,
    seed_everything,
    build_model,
)
from .data_generator import (
    load_events_parquet,
    run_daily_generation,
    FEATURE_COLS,
)
from .concat_tool import (
    concat_lob_from_manifest,
    concat_label_from_manifest,
    build_day_offsets,
)

__version__ = "2.0.0"
__all__ = [
    "TaskOrchestrator", "ExperimentLogger", "DataLoader",
    "WindowDataset", "train_and_test_pipeline", "seed_everything", "build_model",
    "load_events_parquet", "run_daily_generation",
    "FEATURE_COLS",
    "concat_lob_from_manifest", "concat_label_from_manifest", "build_day_offsets",
]
