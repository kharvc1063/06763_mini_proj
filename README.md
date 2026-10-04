# 06-763 Mini Project: TEP Fault Detection

Team project comparing two fault detectors for the Tennessee Eastman Process (TEP), trained only on normal operating data:

- PCA monitoring with T2 and SPE statistics.
- A forecast-residual monitor using Ridge regression.

The first-week implementation generates detector scores. Second-week work will combine the scores to set thresholds, evaluate alarms, compare faults, and analyze channel contributions.

## Project layout

```text
src/
    pca_monitor.py           # PCA training and scoring
    forecast_monitor.py     # Ridge training and scoring
common.py                   # Shared paths, channels, run splits, and preprocessing
data/                       # Downloaded course data; excluded from Git
results/                    # Detector scores and saved forecast parameters
tests/                      # Optional forecast verification
pyproject.toml              # Project metadata and dependencies
uv.lock                     # Resolved dependency versions
.python-version             # Python version
.gitignore                  # Git exclusions
README.md                   # Setup and running instructions
```

The detector scripts import shared choices from common.py. Forecast also uses its loading, run-splitting, and standardization functions. Run the documented commands from the repository root. Forecast resolves its data and output paths through common.py relative to the project root; the PCA script uses relative file paths.

## Environment

Requires Python 3.12 or newer and uv:

```powershell
uv sync --locked
```

uv manages dependencies for this script-based project; no Python package installation or custom console entry point is needed.

## Data

Place these course-provided files in data/:

```text
data/
    tep_fault_free_training.parquet
    tep_faulty_training_runs01-20.parquet
    SHA256SUMS
```

Download with Windows PowerShell from the repository root:

```powershell
New-Item -ItemType Directory -Path data -Force | Out-Null
foreach ($file in @(
    "tep_fault_free_training.parquet"
    "tep_faulty_training_runs01-20.parquet"
    "SHA256SUMS"
)) {
    curl.exe -fL -o "data/$file" "https://kitchin-services.cheme.cmu.edu/f26-06763/data/$file"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $file" }
}
```

Verify the downloaded data against SHA256SUMS before use:

```powershell
Get-Content data/SHA256SUMS | ForEach-Object {
    if ($_ -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
        $expected = $matches[1]
        $file = $matches[2].Trim()
        $actual = (Get-FileHash -LiteralPath "data/$file" -Algorithm SHA256).Hash
        if ($actual -ne $expected) { throw "SHA256 mismatch: $file" }
        Write-Host "${file}: OK"
    }
}
```

Data, virtual environments, and caches are excluded from Git. Commit source code, dependency files, documentation, and the required result files.

## Run the detectors

Create the output directory before running PCA:

```powershell
New-Item -ItemType Directory -Path results -Force | Out-Null
uv run python src/pca_monitor.py
uv run python src/forecast_monitor.py
```

The scripts can also be run independently. Each trains on normal runs and writes its own score file.

## Shared preprocessing

| Setting | Value |
|---|---|
| Channels | xmeas_1 to xmeas_41, then xmv_1 to xmv_11 |
| Training | Normal runs 1-300 |
| Validation | Normal runs 301-400; second-week threshold calibration |
| Test | Normal runs 401-500; second-week false-alarm evaluation |
| Fault evaluation | Faults 1-20, runs 1-20 per fault |
| Standardization | Channel means and standard deviations (ddof=1) from all 150,000 normal training rows |

Validation, test, and faulty data reuse the training standardization parameters. Training rows are not included in the exported score files.

## Detector methods

PCA retains the smallest number of components explaining at least 90% of training variance. It prints the retained component count and computes T2 and SPE for each held-out row.

Forecast uses the standardized rows at t-1 and t-2 as 104 inputs to predict the 52 channels at t with Ridge(alpha=1.0). It fits 149,400 training pairs and computes each channel's training residual standard deviation (ddof=1). The anomaly score is the sum of squared residuals divided by their per-channel residual variances.

Forecast constructs lag features separately for each fault and simulation run. The first two samples of every run have no forecast score.

## Outputs

| File | Columns / contents | Rows |
|---|---|---|
| results/scores_pca.parquet | faultNumber, simulationRun, sample, T2, SPE | 300,000 |
| results/scores_ridge.parquet | faultNumber, simulationRun, sample, score | 298,800 |
| results/ridge_model.npz | Channel order, standardization parameters, Ridge coefficients and intercept, residual standard deviations | Model parameters |

Normal rows use faultNumber 0. PCA scores samples 1-500 of each held-out run; forecast scores samples 3-500. The saved Ridge parameters support second-week contribution analysis.

## Optional forecast verification

The tests are development tools, not required first-week deliverables:

```powershell
uv run python -m unittest discover -s tests -v
uv run python tests/verify_forecast_data.py
```

The unit tests check lag order, run boundaries, and use of training standardization parameters. Dataset verification checks checksums and independently rebuilds Ridge to compare all forecast sample keys and scores. It requires the downloaded data and generated forecast outputs.

These checks do not validate the complete team submission. The course evidence script will check both detectors and second-week outputs when the full pipeline is ready.

## AI use

Generative AI assisted with the forecast implementation, verification, and documentation. Team members should disclose their actual AI use in the final report and be able to explain the code and results.
