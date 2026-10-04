import os

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

#channels and run ranges come from common.py, so PCA 
# and the ridge detector share the same choices
from common import CHANNELS, TRAIN_RUNS, VAL_RUNS, TEST_RUNS, split

df_ff = pl.read_parquet("data/tep_fault_free_training.parquet")
df_faulty = pl.read_parquet("data/tep_faulty_training_runs01-20.parquet")

# faultNumber = 0 for normal data
df_ff = df_ff.with_columns(
    pl.lit(0).cast(df_faulty["faultNumber"].dtype).alias("faultNumber")
)

# 52 channel names list
channels = CHANNELS

# filter first 300 training runs
train_df = split(df_ff, TRAIN_RUNS)

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
val_df = df_ff.filter(df_ff, VAL_RUNS)
test_df = df_ff.filter(df_ff, TEST_RUNS)
faulty_df = df_faulty.filter(
    (pl.col("faultNumber") >= 1)
    & (pl.col("faultNumber") <= 20)
    & (pl.col("simulationRun") >= 1)
    & (pl.col("simulationRun") <= 20)
)

#sanity check that compares our scores 
# with sklearn's transform/inverse_transform on the first 5000 validation rows

z_chk = (val_df.select(channels).to_numpy()[:5000] - mean)/std
pca_k = PCA(n_components=k).fit(z_train)
t_ref = pca_k.transform(z_chk)
T2_ref = np.sum(t_ref**2 / pca_k.explained_variance_, axis=1)
SPE_ref = np.sum((z_chk - pca_k.inverse_transform(t_ref)) ** 2, axis=1)
t_chk = z_chk @ P
assert np.allclose(np.sum(t_chk**2 / lambdas, axis=1), T2_ref)
assert np.allclose(np.sum((z_chk - t_chk @ P.T) ** 2, axis=1), SPE_ref)

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
os.makedirs("results", exist_ok=True)
final_df = pl.concat(results)
final_df.write_parquet("results/scores_pca.parquet")
print("saved results/scores_pca.parquet")

with open("results/pca_k.txt", "w") as f:
    f.write(f"{k}\n")