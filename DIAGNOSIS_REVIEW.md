# Diagnosis integration check — October 4, 2026

Scope: implement the assignment's second-week diagnosis script and verify its
output on the published TEP data. Changes remain local; nothing was committed
or pushed. `WEEK1_REVIEW.md` and the earlier evidence PDF remain historical
records and were not edited.

## Delivered

- `src/diagnose.py` reconstructs SPE and normalized ridge residual contributions
  from saved models, checks their sums against the saved scores, applies the
  three-consecutive-sample alarm rule within each fault/run, pools post-onset
  alarm samples, and writes the top five channel means.
- `results/contributions.csv` contains 170 rows. Faults 3, 9 and 15 produce no
  post-onset alarms for either detector and therefore have no contribution rows.
- The PCA script's two invalid split calls were repaired, its loading/scaling
  now uses `common.py`, and it saves `pca_model.npz` with `P, mean, std, k`.
  `pca_k.txt` records **31**. These changes unblock diagnosis; the PCA owner
  should include them when integrating their work.
- `README.md`, `data.sha256`, and regression tests document and check the handoff.

The diagnosis script reuses `make_lagged_data` from the forecast module. The
forecast source, saved model, and saved score file were not changed. It accepts
the evaluation owner's `thresholds.csv` when present and checks consistency
with the validation scores. In this run that file was absent, so it computed
the prescribed quantiles in memory without writing evaluation artifacts.

## Data and numerical verification

Both existing Parquet files matched the
[published checksum manifest](https://kitchin-services.cheme.cmu.edu/f26-06763/data/SHA256SUMS):

```text
bd98fb16e4c129ce9bdec6455016128af6392cb17e84b72a5bdd3aea0ce680c2  tep_fault_free_training.parquet
e3966fccd6598ff0471c7c939d9203b0d88b4b33536834ac52f7b0d0dabe1ed6  tep_faulty_training_runs01-20.parquet
```

Commands run from the repository root, using the locked Python 3.12 environment:

```bash
sha256sum -c data.sha256
uv run python -m src.pca_monitor
uv run python -m src.diagnose
uv run python -m unittest discover -s tests -v
```

The PCA rebuild kept all original keys. Compared with the original saved
scores, the maximum absolute changes were `6.92e-11` for T² and `5.83e-11` for
SPE (rounded upward); both pass `rtol=1e-10, atol=1e-10`.

Diagnosis thresholds were **11.7745273394** for SPE and **112.442309862** for
ridge. Twelve regression tests pass, including sample-weighted pooling,
pre-onset exclusion, alarm timing at sample 21, fault/run isolation, strict
threshold comparison, lag order, scaling, and stale/misaligned artifact checks.

## Official reference check

Downloaded the
[official evidence script](https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py)
and verified its SHA-256 against the
[published script hash](https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py.sha256):

```text
744445fa94498271372f89d7a8053499fd20999c7754842f770eb7bd5f4a157a
```

Invoked the unmodified script's discovery/collection checks programmatically
against this repository. This was a local check, not an evidence PDF run with
the team's identities; the existing PDF was preserved.

| Check | Result |
| --- | --- |
| All eight detector checks | Pass; 100% required row coverage and numerical agreement within the official tolerance |
| Contributions file and top-channel checks | Pass; **34/34** fault/detector top channels agree |
| Additional comparison of all five ranks | **170/170** fault/detector/rank/channel rows agree |
| Additional comparison of contribution means | All pass `rtol=1e-7, atol=1e-9`; maximum absolute difference `3.25e-6` |
| Six threshold/detection checks | Fail because `thresholds.csv` and `detection.csv` are absent |
| Report check | Skipped by the script; the TA reads it separately |

The additional comparison used the official script's independently fitted
reference models and reference validation thresholds. Production diagnosis
does not import the evidence script or depend on any reference outputs.

## Team handoff

The evaluation owner still supplies thresholds and detection metrics. Once
those files arrive, rerun diagnosis and the official evidence command in the
README with all four real Andrew IDs. The false-alarm denominator and fault-0
fields do not enter diagnosis; their resolution belongs in evaluation.

The forecast owner's shared-preprocessing refactor and the report/figures are
still pending. For section 6, use the contribution rows to check three faults
against the physical fault list, distinguishing the affected channels from the
fault's origin. The diagnosis results alone do not establish causal location.

For the later AI-use disclosure: Codex generated the diagnosis implementation,
regression tests, README and this integration note; it also repaired the PCA
handoff and ran the checks described here. Team members should review and be
able to explain these changes before using them in the submission.
