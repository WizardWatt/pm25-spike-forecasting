"""GRU data pipeline: sliding windows, leakage-safe scaling, shared-format prediction frames.

Per sample (issue time t): X = last `lookback` hours up to and including t, channels:
  [PM2.5 (scaled, 0 where missing), observed-mask, hour/dow/month sin-cos of that step,
   sin-cos of the TARGET hour (t+6) -- calendar, known in advance, so not leakage]
Targets: y_reg (PM2.5 at t+6, raw units), y_spike (any > threshold in t+1..t+6).

Leakage rules enforced here:
  * scaler fitted on the fit period only;
  * a sample belongs to a split by issue time t AND its target time t+6 must lie inside the split
    (train targets never reach into validation/test);
  * look-back may reach into earlier periods (that is past data, which is allowed).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import eval_config as C


class Scaler:
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit(cls, pm25_values):
        v = np.asarray(pm25_values, float)
        v = v[~np.isnan(v)]
        return cls(v.mean(), v.std() + 1e-8)

    def transform(self, x):
        return (np.asarray(x, float) - self.mean) / self.std

    def inverse(self, x):
        return np.asarray(x, float) * self.std + self.mean


def _cyc(values, period):
    a = 2 * np.pi * np.asarray(values, float) / period
    return np.sin(a), np.cos(a)


def hourly_frame(station_df):
    d = station_df.drop_duplicates("Datetime").set_index("Datetime").sort_index()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))


def build_windows(hf, scaler, lookback=24, start=None, end=None, max_missing_frac=0.25, horizon=C.HORIZON):
    idx = hf.index
    pm = hf["PM2.5"].to_numpy(float)
    obs = ~np.isnan(pm)
    pm_s = np.where(obs, scaler.transform(pm), 0.0)
    hs, hc = _cyc(idx.hour, 24); ds, dc = _cyc(idx.dayofweek, 7); ms, mc = _cyc(idx.month - 1, 12)
    th = (idx + pd.Timedelta(hours=horizon)).hour
    ts, tc = _cyc(th, 24)
    feats = np.stack([pm_s, obs.astype(float), hs, hc, ds, dc, ms, mc, ts, tc], axis=1).astype(np.float32)
    names = ["pm25", "observed", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
             "month_sin", "month_cos", "tgt_hour_sin", "tgt_hour_cos"]

    X = sliding_window_view(feats, lookback, axis=0).transpose(0, 2, 1)       # (N, L, F)
    miss = 1.0 - sliding_window_view(obs, lookback).mean(axis=1)
    times = idx[lookback - 1:]
    y_reg = hf["y_reg"].to_numpy(float)[lookback - 1:]
    y_spike = hf["y_spike"].to_numpy(float)[lookback - 1:]

    keep = np.isfinite(y_reg) & np.isfinite(y_spike) & (miss <= max_missing_frac)
    if start is not None:
        keep &= times >= pd.Timestamp(start)
    if end is not None:
        keep &= (times + pd.Timedelta(hours=horizon)) <= pd.Timestamp(end)
    return {"X": np.ascontiguousarray(X[keep]), "y_reg": y_reg[keep].astype(np.float32),
            "y_spike": y_spike[keep].astype(np.float32), "times": times[keep],
            "miss_frac": miss[keep], "feature_names": names}


def make_splits(clean, station, mode="val", lookback=24, max_missing_frac=0.25):
    """mode='val'  : scaler+train on 2015-17, eval on 2018   (all tuning happens here)
       mode='test' : scaler+train on 2015-18, eval on 2019 and later   (run once at the end)"""
    if mode == "val":
        fit_end, ev = C.TRAIN_END, (C.VAL_START, C.VAL_END)
    elif mode == "test":
        fit_end, ev = C.REFIT_END, (C.TEST_START, None)   # 2019 and later (2020 lockdown shift test)
    else:
        raise ValueError(mode)
    st = clean[clean["StationId"] == station]
    hf = hourly_frame(st)
    scaler = Scaler.fit(hf.loc[C.TRAIN_START:fit_end, "PM2.5"])
    kw = dict(lookback=lookback, max_missing_frac=max_missing_frac)
    return {"station": station, "scaler": scaler,
            "train": build_windows(hf, scaler, start=C.TRAIN_START, end=fit_end, **kw),
            "eval": build_windows(hf, scaler, start=ev[0], end=ev[1], **kw)}


def predictions_frame(data, station, y_pred, spike_prob, lower=None, upper=None):
    """data = a split dict (e.g. out['eval']); y_pred/lower/upper in ORIGINAL units
    (apply scaler.inverse first if the net predicts scaled values)."""
    n = len(data["times"])
    nan = np.full(n, np.nan)
    return pd.DataFrame({"Datetime": data["times"], "StationId": station, "y_true": data["y_reg"],
                         "y_pred": np.asarray(y_pred, float),
                         "lower": nan if lower is None else np.asarray(lower, float),
                         "upper": nan if upper is None else np.asarray(upper, float),
                         "spike_prob": np.asarray(spike_prob, float)})[C.PRED_COLUMNS]
