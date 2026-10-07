import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import VAL_RUNS, TEST_RUNS, RESULTS, KEYS, RUN

STEP_MIN = 3.0  # 3 minutes per sample


def find_alarms(df: pl.DataFrame, col: str, thresh: float) -> pl.DataFrame:
    """Flag sample in alarm if it and the 2 samples before it exceed thresh."""
    df = df.sort(KEYS)
    over = (df[col] > thresh).to_numpy()
    runs = df[RUN].to_numpy()

    alarm = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        # Must be in the same run and all 3 consecutive samples must be over thresh
        if runs[i] == runs[i - 1] == runs[i - 2]:
            if over[i] and over[i - 1] and over[i - 2]:
                alarm[i] = True

    return df.with_columns(pl.Series("in_alarm", alarm))


def main():
    RESULTS.mkdir(exist_ok=True)

    # 1. load scores
    pca_df = pl.read_parquet(RESULTS / "scores_pca.parquet")
    ridge_df = pl.read_parquet(RESULTS / "scores_ridge.parquet")

    # filter validation data (runs 301-400, fault 0)
    v_pca = pca_df.filter((pl.col("faultNumber") == 0) & pl.col(RUN).is_between(*VAL_RUNS))
    v_ridge = ridge_df.filter((pl.col("faultNumber") == 0) & pl.col(RUN).is_between(*VAL_RUNS))

    # 2. 0.99 quantile thresholds
    t2_tr = float(np.quantile(v_pca["T2"], 0.99))
    spe_tr = float(np.quantile(v_pca["SPE"], 0.99))
    rg_tr = float(np.quantile(v_ridge["score"], 0.99))

    thresh_df = pl.DataFrame({
        "detector": ["T2", "SPE", "ridge"],
        "threshold": [t2_tr, spe_tr, rg_tr],
    })
    thresh_df.write_csv(RESULTS / "thresholds.csv")
    print("Saved results/thresholds.csv")

    # detectors: (dataframe, column_name, detector_label, threshold)
    detectors = [
        (pca_df, "T2", "T2", t2_tr),
        (pca_df, "SPE", "SPE", spe_tr),
        (ridge_df, "score", "ridge", rg_tr),
    ]

    rows = []

    # 3. compute metrics
    for df, col, name, tr in detectors:
        df_al = find_alarms(df, col, tr)

        # fault 0
        t_0 = df_al.filter((pl.col("faultNumber") == 0) & pl.col(RUN).is_between(*TEST_RUNS))
        far = float(t_0["in_alarm"].mean()) if len(t_0) > 0 else 0.0

        rows.append({
            "fault": 0,
            "detector": name,
            "detection_rate": far,
            "median_delay_min": None,
            "runs_missed": None,
        })

        # faults 1-20
        for f in range(1, 21):
            f_df = df_al.filter(pl.col("faultNumber") == f)
            rates = []
            delays = []
            missed = 0

            for r in range(1, 21):
                run_df = f_df.filter(pl.col(RUN) == r)
                post = run_df.filter(pl.col("sample") > 20)

                if len(post) > 0:
                    rates.append(float(post["in_alarm"].mean()))
                else:
                    rates.append(0.0)

                # delay calculation
                al_post = post.filter(pl.col("in_alarm"))
                if len(al_post) > 0:
                    first_sample = al_post["sample"].min()
                    delays.append((first_sample - 20) * STEP_MIN)
                else:
                    missed += 1

            rows.append({
                "fault": f,
                "detector": name,
                "detection_rate": float(np.mean(rates)) if rates else 0.0,
                "median_delay_min": float(np.median(delays)) if delays else None,
                "runs_missed": missed,
            })

    # 4. results/detection.csv
    det_df = pl.DataFrame(rows).select([
        "fault", "detector", "detection_rate", "median_delay_min", "runs_missed"
    ])
    det_df.write_csv(RESULTS / "detection.csv")
    print("Saved results/detection.csv")


if __name__ == "__main__":
    main()