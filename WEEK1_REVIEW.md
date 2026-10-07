# First-week miniproject review

Reviewed October 4, 2026, at repository HEAD `fbcc1b3`. Scope: the shared setup and the two first-week detectors, with emphasis on the forecast monitor. This review makes no changes to the implementation, saved models, score files, or evidence PDF.

**Assessment:** the forecast monitor implements the prescribed first-week algorithm correctly by source inspection and isolated synthetic execution. Its saved output has the exact required keys and valid numeric values. Official numerical agreement on the TEP data remains unverified because the two input data files are absent. The team repository is not yet reproducible: the current PCA script has a confirmed execution error, and the shared preprocessing is only partially adopted.

Sources: the supplied *Miniproject: Detecting faults in a chemical plant without examples of faults* PDF, especially pages 3–5 and 7–8; `miniproject-evidence.pdf`, especially page 1; current source files and saved artifacts. The assignment is used as the evaluation rubric, not as an instruction to submit or publish anything.

## What belongs in week one

| Requirement | Assessment |
| --- | --- |
| Obtain the two published data files and verify their checksums | Incomplete in this checkout: no `data/` directory. The evidence PDF also reports both files missing. |
| Agree on 52 channels, run splits, and training-only standardization with `ddof=1`; put shared preprocessing in one place | Numerical choices agree in the current source, but shared implementation is incomplete. |
| PCA model retaining the smallest number of components explaining at least 90% of training variance; T2/SPE scores | Model/scoring formulas follow the recipe, and a structurally complete output exists. Current source cannot regenerate it because its split calls fail. |
| Two-lag, 104-input/52-output `Ridge(alpha=1.0)` trained only on fault-free runs 1–300 | Implemented correctly; independently checked on synthetic data. |
| Training residual standard deviations with `ddof=1`; sum of squared normalized residuals | Implemented correctly; independently checked on synthetic data. |
| Required score files with precisely the prescribed rows and columns | Both pass the local structural audit. Numerical agreement with the official data rebuild is still unverified. |

The assignment explicitly places thresholds, alarms, detection metrics, and channel contributions in week two. The combined report follows that evaluation and diagnosis work. Missing `thresholds.csv`, `detection.csv`, `contributions.csv`, and `REPORT.pdf` therefore do not count as first-week deficiencies in this review. The pair split and schedule are suggested workflow, not a separately published week-one grading rubric.

## Findings, in priority order

### 1. The current PCA script cannot regenerate the team's results

At `src/pca_monitor.py:44–45`, these calls are invalid:

```python
val_df = df_ff.filter(df_ff, VAL_RUNS)
test_df = df_ff.filter(df_ff, TEST_RUNS)
```

Polars expects filter predicates, not a DataFrame and a run-range tuple. Reproduced with the project's Polars version: `TypeError: invalid predicate for filter`. The already-imported shared helper provides the intended operation:

```python
val_df = split(df_ff, VAL_RUNS)
test_df = split(df_ff, TEST_RUNS)
```

There is also an invocation issue: executing `python src/pca_monitor.py` from the repository root fails at line 9 with `ModuleNotFoundError: No module named 'common'` in a clean import environment. A root-level module command such as `uv run python -m src.pca_monitor` resolves that import layout, but the split error still needs correction. The README currently documents no supported command.

The saved PCA output was last changed in `e7ea933`; the source was subsequently changed. The assignment explicitly says the evidence script does not execute the team's code, so even numerically correct saved output would not establish that the current source runs. This is a team integration blocker, not a defect in the forecast equations.

### 2. Missing input data prevents first-week numerical verification

`miniproject-evidence.pdf` reports both input files missing, not a demonstrated score mismatch. Both monitors require those files. Restoring the published inputs under `data/` and checking their published SHA-256 checksums is necessary before a real-data rerun and official comparison.

Keeping `data/` in `.gitignore` is appropriate; the data need to exist locally, not be committed. After correcting the PCA execution issue, regenerate both outputs and rerun the evidence script from the project root. Preserve the existing evidence PDF if a before/after comparison is useful. Do not interpret skipped reference checks as passes.

### 3. The forecast monitor has not adopted the shared preprocessing

`common.py:1–5` says both detectors import the shared choices, but `src/forecast_monitor.py` does not import `common` at all. It duplicates channel names, paths, run ranges, and standardization. PCA imports channel/split constants but still duplicates loading and scaling; neither monitor uses the shared loading/scaling helpers.

The copies currently agree: there is no observed channel-order, split, or `ddof` discrepancy. The issue is the first-week integration expectation from assignment page 3 and the risk that a future correction reaches only one detector. Adopt the existing shared constants and preprocessing functions in both scripts, keeping forecast-specific lag construction in the forecast module. Use and document a consistent module invocation so introducing a `common` import does not create another direct-script import failure.

### 4. The team handoff needs minimal run documentation

`README.md` contains only a blank line. Add the data filenames/download and checksum procedure, environment setup (`uv sync --locked`), supported commands from the repository root, expected outputs, and the score-table contract below. Document the fact that ridge omits samples 1–2 of each run.

This is a reproducibility and collaboration gap, not a separately listed automatic grading deduction. A full framework, CLI, or extensive new test suite is unnecessary for this checkpoint.

## Forecast monitor: detailed assessment

| Recipe item | Source | Result |
| --- | --- | --- |
| All 52 channels, in prescribed order | `src/forecast_monitor.py:8–10` | Correct; saved model channel order also matches `common.CHANNELS`. |
| Fit mean and sample standard deviation using only normal runs 1–300 | `:57–64` | Correct; samples 1–2 properly participate in scaling even though they have no forecast target score. |
| Build lag features separately within each fault/run and sort by sample | `:19–35` | Correct. Partitioning by both `faultNumber` and `simulationRun` prevents leakage between different faults sharing a run number. |
| Use `[z[t-1], z[t-2]]` to predict `z[t]` | `:33–35` | Correct feature order and target/key alignment; no future-row input. |
| One multioutput `Ridge(alpha=1.0)` | `:67–69` | Correct, with the default fitted intercept. |
| Compute per-channel residual scale on training predictions, `ddof=1` | `:71–74` | Correct; validation/test/faulty data do not enter fitting or normalization. |
| Score normal validation/test runs and all required faulty runs | `:77–99` | Correct. Normal validation and test roles remain distinct by run number. |
| Sum squared residuals divided by each training residual standard deviation | `:97–99` | Correct. |
| Emit keys plus `score`, skipping each run's first two samples | `:93–107` | Correct; saved output verified independently. |
| Support later per-channel diagnosis | `:110–119` | Useful extra: the saved scaler, coefficient matrix, intercept, residual scales, and channel order are sufficient to recompute contributions from the original inputs. |

No forecast algorithm defect was found in the inspected paths or exercised synthetic cases. This does not substitute for the official numerical comparison on the published data.

## Saved artifact checks

The audit compared the full set of key tuples against the assignment's prescribed runs and samples, rather than checking row counts alone.

| Artifact | Validation rows | Test rows | Faulty rows | Total | Samples per run |
| --- | ---: | ---: | ---: | ---: | --- |
| `results/scores_pca.parquet` | 50,000 | 50,000 | 200,000 | 300,000 | 1–500 |
| `results/scores_ridge.parquet` | 49,800 | 49,800 | 199,200 | 298,800 | 3–500 |

Both files have the exact required columns, zero missing/extra/duplicate keys, zero nulls, and finite, nonnegative scores. Neither includes training rows. Fault-free rows use `faultNumber=0`.

`results/ridge_model.npz` has a `(52, 104)` coefficient matrix, `(52,)` intercept/scaler/residual-scale arrays, the correct channel ordering, and entirely finite numeric parameters. All saved channel and residual standard deviations are positive.

For the next week's handoff:

- Identify rows by all three keys: `(faultNumber, simulationRun, sample)`. Never join by run number alone or concatenate by row position.
- Expect the 1,200-row difference between PCA and ridge: 600 scored runs × 2 unavailable lag targets. Do not pad ridge samples 1–2 with zero scores or remove those PCA samples from its required output.
- Retain the normal pre-fault scores; faults begin at sample 21. Filter by `sample > 20` when calculating fault detection metrics and diagnosis later.
- Reset future alarm logic for every fault/run. Use each detector's own normal validation scores to set its threshold.
- Recompute channel-level squared ridge residuals using the saved parameters and original inputs; the scalar score file alone cannot recover channel contributions. Contribution aggregation must use the alarm rule specified for week two.

## What the evidence PDF establishes

The PDF reports **1.9/10 automatic overall**, consisting of **1.9/5 for detectors** and **0/5 for evaluation/diagnosis**. That is a whole-project intermediate score, not an appropriate percentage-complete measure for week one.

Of eight detector checks, three passed: PCA score file present, ridge score file present, and a ridge fit found in the code. Five were skipped because the inputs were unavailable: PCA row coverage, T2 agreement, SPE agreement, ridge row coverage, and ridge score agreement. The separate local audit establishes structural row coverage against the stated recipe, but does not retroactively change the PDF or verify reference-score agreement.

The evaluation/diagnosis failures concern second-week artifacts. The PDF still contains placeholder team details (`Team name`, `id1`–`id4`); use actual team details for the eventual submission. No submission is required as part of this review.

## Validation performed and limits

- Read both supplied PDFs and inspected the current source, dependency lock, Git history, and saved artifacts.
- Used cached NumPy 2.5.3, Polars 1.44.2, and scikit-learn 1.9.1 with Python 3.12, matching the project package versions. No dependency files were changed.
- Checked shuffled synthetic runs with distinct fault/run/time values: exact lag order, correct target keys, isolated run boundaries, and rejection of missing or duplicate samples.
- Ran the complete forecast pipeline in a temporary directory on synthetic inputs spanning all specified run ranges. Compared its fitted parameters and output scores to an independent, centered ridge normal-equation solution. Deliberately offset held-out data to expose accidental training leakage. All comparisons passed.
- Reproduced both the direct-script PCA import failure and, with import paths and synthetic data supplied, the PCA filter failure.
- Did not download the raw TEP data, rebuild real-data scores, regenerate evidence, or run second-week evaluation. Thus full numerical compliance with the official reference remains unverified.

Recommended order before closing week one: correct the PCA execution error; consolidate the shared preprocessing and document run commands; restore/checksum the local input data; regenerate both detectors and rerun the evidence check. The forecast algorithm itself does not currently need redesign.
