"""Report figures from saved results; run from the project root.

uv run --with matplotlib python -m src.plot_report
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import polars as pl

from common import FAULT_START, RESULTS, RUN
from src.diagnose import alarm_mask


COLORS = {"T2": "#386cb0", "SPE": "#26906c", "ridge": "#bb5a30"}


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def monitoring_plot(results, output, fault, run):
    thresholds = dict(pl.read_csv(results / "thresholds.csv").select(
        "detector", "threshold"
    ).iter_rows())
    pca = pl.read_parquet(results / "scores_pca.parquet")
    ridge = pl.read_parquet(results / "scores_ridge.parquet")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True,
                             layout="constrained")
    for ax, (name, column, scores) in zip(axes, [
        ("T2", "T2", pca), ("SPE", "SPE", pca), ("ridge", "score", ridge)
    ]):
        frame = scores.filter(
            (pl.col("faultNumber") == fault) & (pl.col(RUN) == run)
        ).sort("sample")
        if frame.is_empty():
            raise ValueError(f"No {name} scores for fault {fault}, run {run}.")
        samples = frame["sample"].to_numpy()
        values = frame[column].to_numpy()
        threshold = thresholds[name]
        ax.plot(samples, values, color=COLORS[name], linewidth=1,
                label="Score")
        ax.axhline(threshold, color="#bd3030", linestyle="--", linewidth=1,
                   label=f"99th percentile = {threshold:.2f}")
        ax.axvline(FAULT_START, color="#555555", linestyle=":", linewidth=1,
                   label="Fault onset (after sample 20)")
        alarms = alarm_mask(frame, column, threshold)
        post_onset = alarms & (samples > FAULT_START)
        if post_onset.any():
            first = np.flatnonzero(post_onset)[0]
            ax.scatter(samples[first], values[first], color="black", s=24,
                       zorder=4, label=f"First alarm: sample {samples[first]}")
        # A symmetric log axis also represents zero scores faithfully.
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylabel("$T^2$" if name == "T2" else name)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
        ax.set_xlim(1, 500)
    axes[-1].set_xlabel("Sample (3 minutes per sample)")
    fig.suptitle(f"Fault {fault}, run {run}: monitoring statistics", fontsize=12)
    save_figure(fig, output, f"monitoring_fault{fault:02d}_run{run:02d}")


def contribution_plot(results, output, faults):
    table = pl.read_csv(results / "contributions.csv")
    fig, axes = plt.subplots(2, len(faults),
                             figsize=(3.3 * len(faults), 4.7),
                             squeeze=False, layout="constrained")
    for row, detector in enumerate(("SPE", "ridge")):
        for col, fault in enumerate(faults):
            ax = axes[row, col]
            frame = table.filter(
                (pl.col("fault") == fault) & (pl.col("detector") == detector)
            ).sort("rank")
            ax.set_title(f"Fault {fault} / {detector}", fontsize=10)
            if frame.is_empty():
                ax.text(0.5, 0.5, "No post-onset alarms", ha="center",
                        va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            ax.barh(frame["channel"].to_list(), frame["contribution"].to_numpy(),
                    color=COLORS[detector], height=0.65)
            ax.invert_yaxis()
            ax.set_xlabel("Mean channel contribution", fontsize=9)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
    fig.suptitle("Top five channels averaged over post-onset alarm samples",
                 fontsize=11)
    save_figure(fig, output, "diagnosis_top5_faults" + "_".join(map(str, faults)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--run", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--diagnosis-faults", type=int, nargs="+",
                        choices=range(1, 21), default=[1, 4, 10])
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none"})
    monitoring_plot(args.results_dir, args.output_dir, args.fault, args.run)
    contribution_plot(args.results_dir, args.output_dir, args.diagnosis_faults)


if __name__ == "__main__":
    main()
