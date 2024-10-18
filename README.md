# timeseries-modeling

# Set up environment

```console
timeseries-modeling/ $ python3 -m venv venv
timeseries-modeling/ (venv) $ . venv/bin/activate[.zsh|.fish]
timeseries-modeling/ (venv) $ pip install requirements.local.txt

# or, if you have a GPU
# pip install requirements.cuda.txt
```

# Evaluate local-only models

timeseries-modeling/ (venv) $ python evaluate

# Evaluate Chronos

# Notes for Developers

## Prepare evaluation data from source dataset

```console
timeseries-modeling/ $ cd data
timeseries-modeling/data/ $ pythpn reshape_data.py
```
