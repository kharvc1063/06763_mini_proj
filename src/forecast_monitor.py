from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

# 52 channel names list
channels = [f"xmeas_{i}" for i in range(1, 42)] + [
    f"xmv_{i}" for i in range(1, 12)
]


# Build lag features separately inside each run
def make_lagged_data(df, mean, std):
    key_tables = []
    x_tables = []
    y_tables = []

    df = df.sort(["faultNumber", "simulationRun", "sample"])
    runs = df.partition_by(["faultNumber", "simulationRun"], maintain_order=True)

    for run in runs:
        samples = run["sample"].to_numpy()
        if not np.all(np.diff(samples) == 1):
            raise ValueError("Samples must be consecutive within each run.")
        if len(run) < 3:
            continue

        x = run.select(channels).to_numpy()
        z = (x - mean) / std

        # Predict sample t using sample t-1 and sample t-2
        X = np.hstack([z[1:-1], z[:-2]])
        y = z[2:]
        keys = run.select(["faultNumber", "simulationRun", "sample"]).slice(2)

        key_tables.append(keys)
        x_tables.append(X)
        y_tables.append(y)

    if len(key_tables) == 0:
        raise ValueError("At least three samples are needed in a run.")

    return pl.concat(key_tables), np.vstack(x_tables), np.vstack(y_tables)


def main():
    df_ff = pl.read_parquet("data/tep_fault_free_training.parquet")
    df_faulty = pl.read_parquet("data/tep_faulty_training_runs01-20.parquet")

    # faultNumber = 0 for normal data
    df_ff = df_ff.with_columns(
        pl.lit(0).cast(df_faulty["faultNumber"].dtype).alias("faultNumber")
    )

    # Filter first 300 training runs
    train_df = df_ff.filter(
        (pl.col("simulationRun") >= 1) & (pl.col("simulationRun") <= 300)
    )

    # Standardize using all normal training rows
    x_train = train_df.select(channels).to_numpy()
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=1)

    # Fit one ridge model with 104 inputs and 52 targets
    train_keys, X_train, y_train = make_lagged_data(train_df, mean, std)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Calculate each channel's residual standard deviation on training data
    y_pred_train = ridge.predict(X_train)
    residual_train = y_train - y_pred_train
    residual_std = residual_train.std(axis=0, ddof=1)

    # Filter scoring data
    val_df = df_ff.filter(
        (pl.col("simulationRun") >= 301) & (pl.col("simulationRun") <= 400)
    )
    test_df = df_ff.filter(
        (pl.col("simulationRun") >= 401) & (pl.col("simulationRun") <= 500)
    )
    faulty_df = df_faulty.filter(
        (pl.col("faultNumber") >= 1)
        & (pl.col("faultNumber") <= 20)
        & (pl.col("simulationRun") >= 1)
        & (pl.col("simulationRun") <= 20)
    )

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
    Path("results").mkdir(exist_ok=True)
    final_df = pl.concat(results)
    final_df.write_parquet("results/scores_ridge.parquet")
    print("saved results/scores_ridge.parquet")

    # Save parameters for channel contributions in the second week
    np.savez_compressed(
        "results/ridge_model.npz",
        channels=np.array(channels),
        mean=mean,
        std=std,
        coefficients=ridge.coef_,
        intercept=ridge.intercept_,
        residual_std=residual_std,
    )


if __name__ == "__main__":
    main()

