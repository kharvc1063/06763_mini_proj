from pathlib import Path

import sys

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

# Allow this script to import common.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHANNELS, KEYS, RUN, RESULTS
from common import TRAIN_RUNS, VAL_RUNS, TEST_RUNS
from common import load_fault_free, load_faulty, split, fit_scaler, standardize

channels = CHANNELS


# Build lag features separately inside each run
def make_lagged_data(df, mean, std):
    key_tables = []
    x_tables = []
    y_tables = []

    df = df.sort(KEYS)
    runs = df.partition_by(["faultNumber", RUN], maintain_order=True)

    for run in runs:
        samples = run["sample"].to_numpy()
        if not np.all(np.diff(samples) == 1):
            raise ValueError("Samples must be consecutive within each run.")
        if len(run) < 3:
            continue

        z = standardize(run, mean, std)

        # Predict sample t using sample t-1 and sample t-2
        X = np.hstack([z[1:-1], z[:-2]])
        y = z[2:]
        keys = run.select(KEYS).slice(2)

        key_tables.append(keys)
        x_tables.append(X)
        y_tables.append(y)

    if len(key_tables) == 0:
        raise ValueError("At least three samples are needed in a run.")

    return pl.concat(key_tables), np.vstack(x_tables), np.vstack(y_tables)


def main():
    # Load sorted data with faultNumber = 0 for normal runs.
    df_ff = load_fault_free()
    df_faulty = load_faulty()

    # Filter first 300 training runs
    train_df = split(df_ff, TRAIN_RUNS)

    # Standardize using all normal training rows
    mean, std = fit_scaler(train_df)

    # Fit one ridge model with 104 inputs and 52 targets
    train_keys, X_train, y_train = make_lagged_data(train_df, mean, std)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Calculate each channel's residual standard deviation on training data
    y_pred_train = ridge.predict(X_train)
    residual_train = y_train - y_pred_train
    residual_std = residual_train.std(axis=0, ddof=1)

    # Filter scoring data
    val_df = split(df_ff, VAL_RUNS)
    test_df = split(df_ff, TEST_RUNS)
    faulty_df = df_faulty

    all_dfs = [val_df, test_df, faulty_df]
    results = []

    # Score each dataset; the first two samples of every run have no score
    for df in all_dfs:
        keys, X, y = make_lagged_data(df, mean, std)
        y_pred = ridge.predict(X)
        residual = y - y_pred
        normalized_residual = residual / residual_std
        score = np.sum(normalized_residual ** 2, axis=1)

        res = keys.with_columns(pl.Series("score", score))
        results.append(res)

    # Save parquet file
    RESULTS.mkdir(exist_ok=True)
    final_df = pl.concat(results)
    final_df.write_parquet(RESULTS / "scores_ridge.parquet")
    print("saved results/scores_ridge.parquet")

    # Save parameters for channel contributions in the second week
    np.savez_compressed(
        RESULTS / "ridge_model.npz",
        channels=np.array(channels),
        mean=mean,
        std=std,
        coefficients=ridge.coef_,
        intercept=ridge.intercept_,
        residual_std=residual_std,
    )


if __name__ == "__main__":
    main()

