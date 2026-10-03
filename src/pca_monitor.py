import numpy as np
import polars as pl
from sklearn.decomposition import PCA

df_ff = pl.read_parquet("data/tep_fault_free_training.parquet")
df_faulty = pl.read_parquet("data/tep_faulty_training_runs01-20.parquet")

# faultNumber = 0 for normal data
df_ff = df_ff.with_columns(
    pl.lit(0).cast(df_faulty["faultNumber"].dtype).alias("faultNumber")
)

# 52 channel names list
channels = [f"xmeas_{i}" for i in range(1, 42)]+[
    f"xmv_{i}" for i in range(1, 12)
]

# filter first 300 training runs
train_df = df_ff.filter(
    (pl.col("simulationRun") >= 1) & (pl.col("simulationRun") <= 300)
)

x_train = train_df.select(channels).to_numpy()
mean = x_train.mean(axis=0)
std = x_train.std(axis=0, ddof=1)

z_train = (x_train - mean)/std

# fit pca model on training data
pca = PCA().fit(z_train)

# k components for 90% variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
k = int(np.searchsorted(cum_var, 0.90) + 1)
print("k = ", k)

# loading matrix P and eigenvalues
P = pca.components_[:k].T
lambdas = pca.explained_variance_[:k]

# scoring data filering
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

# score each dataset
for df in all_dfs:
    x = df.select(channels).to_numpy()
    z = (x - mean)/std

    t = z @ P

    T2 = np.sum((t**2)/lambdas, axis=1)
    z_hat = t @ P.T
    SPE = np.sum((z - z_hat) ** 2, axis=1)

    res = df.select(["faultNumber", "simulationRun", "sample"]).with_columns(
        [
            pl.Series("T2", T2),
            pl.Series("SPE", SPE),
        ]
    )
    results.append(res)

# save parquet file
final_df = pl.concat(results)
final_df.write_parquet("results/scores_pca.parquet")
print("saved results/scores_pca.parquet")