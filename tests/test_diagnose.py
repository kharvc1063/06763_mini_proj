"""Regression cases for diagnosis alignment, alarm timing and pooled means."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from common import CHANNELS, KEYS, N_SAMPLES, RUN, VAL_RUNS
from src.diagnose import (
    alarm_mask, diagnosis_thresholds, load_model, ridge_contributions,
    spe_contributions, summarize_contributions, validation_threshold,
)


def keys(rows):
    return pl.DataFrame(rows, schema=KEYS, orient="row")


class AlarmTests(unittest.TestCase):
    def test_strict_threshold_and_overlapping_alarms(self):
        scores = keys([(1, 1, i) for i in range(1, 9)]).with_columns(
            pl.Series("score", [2, 2, 1, 2, 2, 2, 2, 0])
        )
        np.testing.assert_array_equal(
            alarm_mask(scores, "score", 1),
            [False, False, False, False, False, True, True, False],
        )

    def test_never_cross_run_fault_or_sample_gap(self):
        scores = keys([
            (1, 1, 499), (1, 1, 500), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (2, 2, 4), (2, 2, 5), (2, 2, 6), (2, 2, 8), (2, 2, 9), (2, 2, 10),
        ]).with_columns(pl.lit(2.0).alias("score"))
        np.testing.assert_array_equal(np.flatnonzero(alarm_mask(scores, "score", 1)), [4, 7, 10])

    def test_short_runs_cannot_alarm(self):
        for n in range(3):
            scores = keys([(1, 1, i) for i in range(1, n + 1)]).with_columns(pl.lit(2.0).alias("score"))
            self.assertFalse(alarm_mask(scores, "score", 1).any())


class ContributionTests(unittest.TestCase):
    def test_spe_uses_saved_scaler_and_projection(self):
        n = len(CHANNELS)
        mean, std = np.arange(n), np.arange(n) + 1
        z = np.zeros((2, n))
        z[:, :3] = [[3, 4, 5], [-3, -4, -2]]
        raw = z * std + mean
        frame = keys([(1, 1, 21), (1, 1, 22)]).hstack(pl.DataFrame(raw, schema=CHANNELS))
        projection = np.zeros((n, 1))
        projection[:2, 0] = 1 / np.sqrt(2)
        result_keys, values = spe_contributions(frame.reverse(), {"mean": mean, "std": std, "P": projection})
        self.assertTrue(result_keys.equals(frame.select(KEYS)))
        np.testing.assert_allclose(values[:, :3], [[0.25, 0.25, 25], [0.25, 0.25, 4]])
        np.testing.assert_allclose(values[:, 3:], 0)

    def test_ridge_lag_order_intercept_and_residual_scaling(self):
        n = len(CHANNELS)
        # Reused run numbers across faults expose accidental cross-fault lags.
        frame_keys = keys([(fault, run, t) for fault, run in [(1, 1), (1, 2), (2, 1)] for t in range(1, 5)])
        z = np.zeros((12, n))
        z[:, 0] = [1, 3, 8, 20, 100, 103, 108, 120, 200, 203, 208, 220]
        mean, std = np.ones(n) * 7, np.ones(n) * 2
        frame = frame_keys.hstack(pl.DataFrame(z * std + mean, schema=CHANNELS)).reverse()
        coefficients = np.zeros((n, 2 * n))
        coefficients[0, 0], coefficients[0, n] = 2, -1
        intercept = np.zeros(n)
        intercept[0] = 0.5
        residual_std = np.ones(n)
        residual_std[0] = 2
        result_keys, values = ridge_contributions(frame, {
            "mean": mean, "std": std, "coefficients": coefficients,
            "intercept": intercept, "residual_std": residual_std,
        })
        self.assertEqual(result_keys.rows(), [(f, r, t) for f, r in [(1, 1), (1, 2), (2, 1)] for t in (3, 4)])
        np.testing.assert_allclose(values[:, 0], np.array([2.5, 6.5, 1.5, 6.5, 1.5, 6.5]) ** 2 / 4)
        np.testing.assert_allclose(values[:, 1:], 0)

    def pooled_case(self):
        frame_keys = keys([(1, 1, t) for t in (19, 20, 21)] + [(1, 2, t) for t in (19, 20, 21, 22, 23)])
        values = np.zeros((8, len(CHANNELS)))
        values[:3, 0], values[3:, 1] = 100, 50
        scores = frame_keys.with_columns(pl.Series("SPE", values.sum(axis=1)))
        return frame_keys, values, scores

    def test_pool_samples_not_run_means_and_include_alarm_at_21(self):
        frame_keys, values, scores = self.pooled_case()
        result = summarize_contributions(frame_keys.reverse(), values[::-1], scores.reverse(), "SPE", 10, "SPE")
        # One sample from run 1 and three from run 2: 100/4 and 150/4.
        self.assertEqual(result["channel"][:2].to_list(), ["xmeas_2", "xmeas_1"])
        self.assertEqual(result["contribution"][:2].to_list(), [37.5, 25.0])
        self.assertEqual(result["rank"].to_list(), [1, 2, 3, 4, 5])
        self.assertEqual(result["fault"].unique().to_list(), [1])

    def test_no_post_onset_alarms_produces_header_only_table(self):
        frame_keys, values, scores = self.pooled_case()
        result = summarize_contributions(frame_keys, values, scores, "SPE", 100, "SPE")
        self.assertEqual(result.height, 0)
        self.assertEqual(result.columns, ["fault", "detector", "rank", "channel", "contribution"])

    def test_excludes_pre_onset_alarm_contributions(self):
        frame_keys = keys([(1, 1, t) for t in range(18, 23)])
        values = np.zeros((5, len(CHANNELS)))
        values[:, 0] = [9, 9, 9, 1, 1]
        scores = frame_keys.with_columns(pl.Series("SPE", values.sum(axis=1)))
        result = summarize_contributions(frame_keys, values, scores, "SPE", 0.5, "SPE")
        # Sample 20 is in alarm but its contribution of 9 must not enter the mean.
        self.assertEqual(result["contribution"][0], 1.0)

    def test_reject_stale_scores_and_duplicate_or_missing_keys(self):
        frame_keys, values, scores = self.pooled_case()
        for bad in [scores.with_columns(pl.col("SPE") + 1), scores.head(7),
                    pl.concat([scores.head(7), scores.head(1)]),
                    scores.with_columns(pl.col("sample") + 1)]:
            with self.subTest(bad=bad.shape), self.assertRaises(ValueError):
                summarize_contributions(frame_keys, values, bad, "SPE", 10, "SPE")


class ThresholdAndModelTests(unittest.TestCase):
    def validation_scores(self, first, column):
        runs = np.arange(VAL_RUNS[0], VAL_RUNS[1] + 1)
        samples = np.arange(first, N_SAMPLES + 1)
        return pl.DataFrame({
            "faultNumber": np.zeros(len(runs) * len(samples), dtype=int),
            RUN: np.repeat(runs, len(samples)), "sample": np.tile(samples, len(runs)),
            column: np.arange(len(runs) * len(samples), dtype=float),
        })

    def test_quantile_excludes_training_test_and_faulty_scores(self):
        for first, column in [(1, "SPE"), (3, "score")]:
            scores = self.validation_scores(first, column)
            extras = keys([(0, 1, 1), (0, 401, 1), (1, 301, 21)]).with_columns(pl.lit(1e12).alias(column))
            mixed = pl.concat([extras, scores], how="vertical_relaxed").reverse()
            self.assertEqual(validation_threshold(mixed, column, first), np.quantile(scores[column].to_numpy(), 0.99))
            with self.assertRaisesRegex(ValueError, "validation keys"):
                validation_threshold(scores.head(scores.height - 1), column, first)

    def test_threshold_file_must_match_and_is_not_created_by_fallback(self):
        pca, ridge = self.validation_scores(1, "SPE"), self.validation_scores(3, "score")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expected = diagnosis_thresholds(folder, pca, ridge)
            self.assertFalse((folder / "thresholds.csv").exists())
            table = pl.DataFrame({"detector": ["T2", "SPE", "ridge"], "threshold": [1.0, expected["SPE"], expected["ridge"]]})
            table.write_csv(folder / "thresholds.csv")
            self.assertEqual(diagnosis_thresholds(folder, pca, ridge), expected)
            table.with_columns(pl.col("threshold") + 1).write_csv(folder / "thresholds.csv")
            with self.assertRaisesRegex(ValueError, "99th percentile"):
                diagnosis_thresholds(folder, pca, ridge)

    def test_saved_model_contract_and_positive_scales(self):
        n = len(CHANNELS)
        model = dict(P=np.eye(n)[:, :2], mean=np.zeros(n), std=np.ones(n), k=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            np.savez(path, **model)
            self.assertEqual(int(load_model(path, "SPE")["k"]), 2)
            for change in [dict(std=np.zeros(n)), dict(k=2.5), dict(P=np.ones((n, 2))),
                           dict(channels=np.array(CHANNELS[::-1]))]:
                np.savez(path, **(model | change))
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    load_model(path, "SPE")


if __name__ == "__main__":
    unittest.main()
