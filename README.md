# timeseries-modeling

Experiments for the paper *Evaluating time series forecasts with value-distribution Earth mover's distance*. The core metric implementations are in [pemd.py](pemd.py) and [metrics.py](metrics.py); the main evaluation entry point is [viz_models.py](viz_models.py).

## Setup

```console
$ pip install -r requirements.local.txt
```

Or with Poetry:

```console
$ poetry install
```

## Running tests

```console
$ python -m pytest tests/
```

Both test modules should pass cleanly:

- `tests/test_pemd.py` — unit tests for `value_distribution_emd` and `probabilistic_value_distribution_emd` in [pemd.py](pemd.py)
- `tests/test_metrics_gluonts.py` — integration tests for the GluonTS `EMD` and `PEMD` metric wrappers in [metrics.py](metrics.py)

## Running the evaluation

The main script evaluates all nine forecasters on the July bike-demand series (casual and registered riders, 3:1 context-to-horizon ratio) and writes results to `./out/results_metrics.csv` alongside per-model forecast plots.

```console
$ python viz_models.py
```

Environment variables for overrides:

| Variable | Default | Description |
|---|---|---|
| `CONFIG_FILE_PATH` | `./evaluation_configs/bike-zero-shot.yaml` | Backtest config |
| `DATA_DIR_PATH` | `./data/` | Root directory for HuggingFace-format datasets |
| `RESULTS_FILE_PATH` | `./out/results_metrics.csv` | Output CSV (deleted and rewritten on each run) |
| `PLOT_DIR` | `./out` | Directory for forecast PNGs |

Running with defaults should produce results consistent with [results_metrics_pemd.csv](results_metrics_pemd.csv) (the canonical reference output committed to the repo).

## Processing results into LaTeX tables

```console
$ python process_results.py
```

By default this reads `./out/results_metrics.csv`. To use the committed reference results instead:

```console
$ RESULTS_FILE_PATH=./results_metrics_pemd.csv python process_results.py
```

The script prints one LaTeX `table*` block per unique `ratio` value found in the CSV, with best values bolded per metric.

## Sparse-series toy example

[emd_plots.py](emd_plots.py) generates the two illustrative sparse-series figures (Figures 1–2 in the paper). The output paths are currently hardcoded; edit the `plt.savefig(...)` calls at the bottom of the file before running:

```console
$ python emd_plots.py
```

## Repo layout

```
pemd.py              — core EMD / pEMD functions (no dependencies)
metrics.py           — GluonTS metric wrappers (EMD, PEMD, helpers)
predictors.py        — all nine forecaster classes
viz_models.py        — main evaluation loop
process_results.py   — CSV → LaTeX table
emd_plots.py         — sparse-series toy plots
evaluation_configs/  — YAML backtest configs
data/                — preprocessed HuggingFace-format bike datasets
out/                 — generated plots and results CSV (gitignored except reference CSV)
tests/               — unit + integration tests
```
