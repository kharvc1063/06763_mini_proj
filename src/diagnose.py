import argparse
from pathlib import Path

import numpy as np
import polars as pl

# Channels, paths and run ranges come from common.py
from common import (
    CHANNELS, FAULT_START, KEYS, N_SAMPLES, RESULTS, RUN, VAL_RUNS,
    load_faulty, split, standardize,
)
from src.forecast_monitor import make_lagged_data

CONTRIBUTION_SCHEMA = {
    "fault": pl.Int64,
    "detector": pl.String,
    "rank": pl.Int64,
    "channel": pl.String,
    "contribution": pl.Float64,
}


# Load saved parameters and check that they match the 52 channels
def load_model(path, detector):
    if not path.is_file():
        owner = "pca_monitor" if detector == "SPE" else "forecast_monitor"
        raise ValueError(f"Missing {path}; run uv run python -m src.{owner} first.")
    with np.load(path, allow_pickle=False) as saved:
        model = {name: saved[name] for name in saved.files}

    n = len(CHANNELS)
    shapes = {"mean": (n,), "std": (n,)}

    # PCA stores the loading matrix and number of components
    if detector == "SPE":
        k = model.get("k", np.array(np.nan))
        if k.shape != () or not np.isfinite(k):
            raise ValueError(f"{path}: k must be an integer from 1 to {n}.")
        k = float(k)
        if k < 1 or k > n or k != int(k):
            raise ValueError(f"{path}: k must be an integer from 1 to {n}.")
        shapes["P"] = (n, int(k))
    else:
        # Ridge stores the two-lag coefficients and residual scales
        shapes["coefficients"] = (n, 2 * n)
        shapes["intercept"] = (n,)
        shapes["residual_std"] = (n,)
        if "channels" not in model:
            raise ValueError(f"{path}: missing channel order.")

    if "channels" in model and model["channels"].tolist() != CHANNELS:
        raise ValueError(f"{path}: channel order does not match common.CHANNELS.")

    for name, shape in shapes.items():
        if name not in model or model[name].shape != shape:
            raise ValueError(f"{path}: {name} must be finite with shape {shape}.")
        if not np.isfinite(model[name]).all():
            raise ValueError(f"{path}: {name} must be finite with shape {shape}.")

    for name in ("std", "residual_std"):
        if name in model and np.any(model[name] <= 0):
            raise ValueError(f"{path}: {name} must be positive.")

    if detector == "SPE":
        P = model["P"]
        if not np.allclose(P.T @ P, np.eye(int(k)), atol=1e-8):
            raise ValueError(f"{path}: PCA loading vectors must be orthonormal.")

    return model


# Calculate one squared PCA residual per channel
def spe_contributions(df, model):
    df = df.sort(KEYS)
    z = standardize(df, model["mean"], model["std"])
    P = model["P"]

    t = z @ P
    z_hat = t @ P.T
    residual = z - z_hat
    contributions = residual ** 2

    return df.select(KEYS), contributions


# Rebuild the same lag features used in forecast_monitor.py
def ridge_contributions(df, model):
    keys, X, y = make_lagged_data(df, model["mean"], model["std"])

    y_pred = X @ model["coefficients"].T + model["intercept"]
    residual = y - y_pred
    normalized_residual = residual / model["residual_std"]
    contributions = normalized_residual ** 2

    return keys, contributions


# Set the threshold using only fault-free validation runs
def validation_threshold(scores, column, first_sample):
    df_ff = scores.filter(pl.col("faultNumber") == 0)
    val_df = split(df_ff, VAL_RUNS).sort(KEYS)

    # Every validation run must contain all of its scored samples
    samples = np.arange(first_sample, N_SAMPLES + 1)
    runs = np.arange(VAL_RUNS[0], VAL_RUNS[1] + 1)
    expected_runs = np.repeat(runs, len(samples))
    expected_samples = np.tile(samples, len(runs))
    if (
        val_df.height != len(expected_runs)
        or not np.array_equal(val_df[RUN].to_numpy(), expected_runs)
        or not np.array_equal(val_df["sample"].to_numpy(), expected_samples)
    ):
        raise ValueError(f"{column}: missing, duplicate or unexpected validation keys.")

    values = val_df[column].to_numpy()
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{column}: validation scores must be finite and nonnegative.")

    return float(np.quantile(values, 0.99))


# Use saved thresholds when available, after checking the validation quantiles
def diagnosis_thresholds(results_dir, pca, ridge):
    expected = {
        "SPE": validation_threshold(pca, "SPE", 1),
        "ridge": validation_threshold(ridge, "score", 3),
    }
    path = results_dir / "thresholds.csv"
    if not path.exists():
        print("thresholds.csv absent; using validation 99th percentiles in memory.")
        return expected

    table = pl.read_csv(path)
    if not {"detector", "threshold"} <= set(table.columns):
        raise ValueError(f"{path}: expected detector and threshold columns.")

    thresholds = {}
    for detector, value in expected.items():
        rows = table.filter(pl.col("detector") == detector)
        if rows.height != 1:
            raise ValueError(f"{path}: expected exactly one {detector} threshold.")

        threshold = float(rows["threshold"][0])
        matches = np.isclose(threshold, value, rtol=1e-10, atol=1e-12)
        if not np.isfinite(threshold) or not matches:
            raise ValueError(
                f"{path}: {detector} threshold differs from the validation 99th percentile."
            )
        thresholds[detector] = threshold

    return thresholds


# Three consecutive scores above the threshold make an alarm
# Scores must be sorted by fault, run and sample
def alarm_mask(scores, column, threshold):
    above = scores[column].to_numpy() > threshold
    alarm = np.zeros(scores.height, dtype=bool)
    if scores.height >= 3:
        fault = scores["faultNumber"].to_numpy()
        run = scores[RUN].to_numpy()
        sample = scores["sample"].to_numpy()

        # Consecutive samples must belong to the same fault and run
        same_fault = fault[1:] == fault[:-1]
        same_run = run[1:] == run[:-1]
        next_sample = np.diff(sample) == 1
        adjacent = same_fault & same_run & next_sample

        three_above = above[2:] & above[1:-1] & above[:-2]
        alarm[2:] = three_above & adjacent[1:] & adjacent[:-1]

    return alarm


# Average contributions over alarm samples and keep the top five channels
def summarize_contributions(keys, values, scores, column, threshold, detector):
    if (
        values.shape != (keys.height, len(CHANNELS))
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(f"{detector}: expected finite, nonnegative per-channel contributions.")

    # Sort the contributions and match saved scores using all three keys
    order = keys.with_row_index("_row").sort(KEYS)
    values = values[order["_row"].to_numpy()]
    keys = order.select(KEYS)
    if scores.height != keys.height or scores.select(KEYS).n_unique() != scores.height:
        raise ValueError(f"{detector}: missing, extra or duplicate faulty score keys.")
    aligned = keys.join(
        scores.select(KEYS + [column]),
        on=KEYS,
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if aligned[column].null_count():
        raise ValueError(f"{detector}: faulty score keys do not match the data.")

    # Channel contributions must add up to the detector's saved score
    saved_scores = aligned[column].to_numpy()
    total = np.sum(values, axis=1)
    matches = np.allclose(total, saved_scores, rtol=1e-7, atol=1e-9)
    if not np.isfinite(saved_scores).all() or not matches:
        raise ValueError(
            f"{detector}: contributions do not sum to saved scores; "
            "regenerate the model and scores together."
        )

    # Samples 19 and 20 can help establish an alarm at 21
    # Only samples after 20 enter the contribution mean
    alarm = alarm_mask(aligned, column, threshold)
    post_fault = aligned["sample"].to_numpy() > FAULT_START
    keep = alarm & post_fault
    faults = aligned["faultNumber"].to_numpy()

    rows = []
    for fault in range(1, 21):
        selected = keep & (faults == fault)
        if not selected.any():
            continue  # No channel ranking when there are no post-fault alarms

        # Pool all alarm samples across runs, giving each sample equal weight
        mean = values[selected].mean(axis=0)
        top_channels = np.argsort(-mean, kind="stable")[:5]

        # Tied contributions follow the channel order from common.py
        for rank, channel in enumerate(top_channels, start=1):
            rows.append({
                "fault": fault,
                "detector": detector,
                "rank": rank,
                "channel": CHANNELS[channel],
                "contribution": float(mean[channel]),
            })

    return pl.DataFrame(rows, schema=CONTRIBUTION_SCHEMA)


def diagnose(results_dir=RESULTS):
    # Load the saved models and scores
    pca_model = load_model(results_dir / "pca_model.npz", "SPE")
    ridge_model = load_model(results_dir / "ridge_model.npz", "ridge")
    pca_scores = pl.read_parquet(results_dir / "scores_pca.parquet")
    ridge_scores = pl.read_parquet(results_dir / "scores_ridge.parquet")

    # Get validation thresholds and load the faulty runs
    thresholds = diagnosis_thresholds(results_dir, pca_scores, ridge_scores)
    df_faulty = load_faulty()

    detectors = [
        ("SPE", pca_model, pca_scores, "SPE"),
        ("ridge", ridge_model, ridge_scores, "score"),
    ]
    results = []

    # Calculate channel contributions for each detector
    for detector, model, scores, column in detectors:
        if detector == "SPE":
            keys, values = spe_contributions(df_faulty, model)
        else:
            keys, values = ridge_contributions(df_faulty, model)

        scores = scores.filter(pl.col("faultNumber") != 0)
        threshold = thresholds[detector]
        res = summarize_contributions(keys, values, scores, column, threshold, detector)
        results.append(res)

        faults_with_alarms = res["fault"].to_list()
        absent = [fault for fault in range(1, 21) if fault not in faults_with_alarms]
        print(
            f"{detector}: threshold={threshold:.12g}; "
            f"faults without post-onset alarms: {absent}"
        )

    final_df = pl.concat(results).sort(["fault", "detector", "rank"])
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Rank SPE and ridge channel contributions over post-onset alarm samples."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory with saved models, scores and optional thresholds.csv.",
    )
    args = parser.parse_args()
    try:
        final_df = diagnose(args.results_dir)
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.exit(1, f"Diagnosis failed: {error}\n")

    # Save the top five channels per fault and detector
    output = args.results_dir / "contributions.csv"
    final_df.write_csv(output)
    print(f"Saved {output} ({final_df.height} rows).")


if __name__ == "__main__":
    main()
