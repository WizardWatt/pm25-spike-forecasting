"""Write/read prediction files in Person A's format, so xgb / gru / baselines are interchangeable.

Columns: Datetime, StationId, split, model, y_true, y_pred, lower, upper, spike_prob, y_spike, alert,
pm25_now, is_diwali, is_lockdown, target_imputed.
`model` is the fit used: 'train_only' (fit 2015-17, rows from 2018) or 'refit' (fit 2015-18, 2019+).
One file per model and station: outputs/predictions/{name}_{station}.csv
"""
import os
import numpy as np
import pandas as pd
import eval_config as C

SHARED_COLUMNS = ["Datetime", "StationId", "split", "model", "y_true", "y_pred", "lower", "upper",
                  "spike_prob", "y_spike", "alert", "pm25_now", "is_diwali", "is_lockdown", "target_imputed"]
_EXTRA = ["y_spike", "split", "PM2.5", "is_diwali", "is_lockdown", "target_imputed"]


def to_shared(frame, clean, label, thr):
    """frame: the 7 core columns (Datetime, StationId, y_true, y_pred, lower, upper, spike_prob)."""
    f = frame.copy()
    f["Datetime"] = pd.to_datetime(f["Datetime"])
    extra = clean.reindex(columns=["Datetime", "StationId"] + _EXTRA).rename(columns={"PM2.5": "pm25_now"})
    f = f.merge(extra, on=["Datetime", "StationId"], how="left")
    f["model"] = label
    f["alert"] = (f["spike_prob"] >= thr).astype(int)            # no probability -> no alert
    for c in ("is_diwali", "is_lockdown", "target_imputed"):
        f[c] = f[c].fillna(False).astype(bool)
    return f[SHARED_COLUMNS]


def _path(name, station, out_dir=None):
    return os.path.join(out_dir or C.PRED_DIR, f"{name}_{station}.csv")


def save_model_preds(shared, name, station, out_dir=None):
    """Write rows for this model+station, replacing earlier rows with the same `model` label."""
    path = _path(name, station, out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["Datetime"])
        shared = pd.concat([old[~old["model"].isin(shared["model"].unique())], shared], ignore_index=True)
    shared["Datetime"] = pd.to_datetime(shared["Datetime"])
    shared.sort_values("Datetime").to_csv(path, index=False)
    return path


def load_preds(name, station, out_dir=None):
    return pd.read_csv(_path(name, station, out_dir))
