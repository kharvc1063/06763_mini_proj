"""Shared choices for the TEP miniproject: paths, channels, the fixed run splits,
loading, and standardization.

Both detectors (PCA and forecast/ridge) import from here, so each fixed choice
in the project lives in one place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FAULT_FREE = DATA / "tep_fault_free_training.parquet"
FAULTY = DATA / "tep_faulty_training_runs01-20.parquet"
RESULTS = ROOT / "results"

RUN = "simulationRun"
KEYS = ["faultNumber", RUN, "sample"]
CHANNELS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
N_SAMPLES = 500

TRAIN_RUNS = (1, 300)   # fit the model
VAL_RUNS = (301, 400)   # set thresholds only
TEST_RUNS = (401, 500)  # measure false alarms only
FAULT_START = 20        # samples 1 to 20 of a faulty run are normal


def load_fault_free() -> pl.DataFrame:
    """The fault-free file, with faultNumber 0, sorted so rows within a run are in time order."""
    if not FAULT_FREE.exists():
        raise SystemExit(f"{FAULT_FREE} not found. Download the data into data/ first.")
    df = pl.read_parquet(FAULT_FREE)
    faulty_type = pl.read_parquet_schema(FAULTY)["faultNumber"]
    df = df.with_columns(pl.lit(0).cast(faulty_type).alias("faultNumber"))
    assert df[RUN].n_unique() == 500, "expected 500 fault-free runs"
    assert df.null_count().sum_horizontal().item() == 0, "unexpected nulls"
    return df.sort(KEYS)


def load_faulty() -> pl.DataFrame:
    """Faults 1 to 20, runs 1 to 20, sorted by fault, run and sample."""
    if not FAULTY.exists():
        raise SystemExit(f"{FAULTY} not found. Download the data into data/ first.")
    df = pl.read_parquet(FAULTY).filter(
        pl.col("faultNumber").is_between(1, 20) & pl.col(RUN).is_between(1, 20)
    )
    assert df.height == 20 * 20 * N_SAMPLES, "expected 20 faults x 20 runs x 500 samples"
    return df.sort(KEYS)


def split(df: pl.DataFrame, runs: tuple[int, int]) -> pl.DataFrame:
    """Rows whose run falls in an inclusive (first, last) range."""
    return df.filter(pl.col(RUN).is_between(*runs))


def fit_scaler(train: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std (ddof=1), from the training runs only."""
    x = train.select(CHANNELS).to_numpy()
    return x.mean(axis=0), x.std(axis=0, ddof=1)


def standardize(df: pl.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardized channel matrix, in the frame's row order."""
    return (df.select(CHANNELS).to_numpy() - mean) / std
