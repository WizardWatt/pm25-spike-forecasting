"""Evaluation harness. Input: prediction file (PRED_COLUMNS) + cleaned file (CLEAN_COLUMNS).

Definitions
- Row-level spike metrics use y_spike (any PM2.5 > threshold in the next `horizon` hours).
  recall = TP/(TP+FN); precision = TP/(TP+FP);
  false_alarm_rate = FP/(FP+TN)  (share of calm hours that raised an alert);
  false_alarm_ratio = FP/(FP+TP) (share of alerts that were wrong = 1 - precision).
- Episode-level metrics use raw PM2.5: an episode starts at the first hour above threshold
  after a non-exceeding hour. It is "detected" if an alert was issued in the `horizon` hours
  before its onset. Lead time = onset - earliest such alert (hours, max = horizon).
"""
import numpy as np
import pandas as pd

KEYS = ["Datetime", "StationId"]


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "n": 0}
    e = y_pred[m] - y_true[m]
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "n": int(m.sum())}


def interval_metrics(y_true, lower, upper):
    y, lo, hi = (np.asarray(a, float) for a in (y_true, lower, upper))
    m = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    if m.sum() == 0:
        return {"coverage": np.nan, "mean_width": np.nan}
    return {"coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
            "mean_width": float(np.mean(hi[m] - lo[m]))}


def _div(a, b):
    return float(a / b) if b else np.nan


def spike_metrics(y_spike, alert):
    y, a = np.asarray(y_spike).astype(bool), np.asarray(alert).astype(bool)
    tp, fp = int((y & a).sum()), int((~y & a).sum())
    fn, tn = int((y & ~a).sum()), int((~y & ~a).sum())
    return {"recall": _div(tp, tp + fn), "precision": _div(tp, tp + fp),
            "false_alarm_rate": _div(fp, fp + tn), "false_alarm_ratio": _div(fp, fp + tp),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def episode_metrics(pm25: pd.Series, alert: pd.Series, threshold=90, horizon=6, start=None, end=None,
                    short_max_h=2):
    """pm25, alert: Series indexed by Datetime (one station). alert is bool.
    start/end = first/last forecast issue time evaluated: only episodes whose whole alert window
    [onset-horizon, onset-1h] lies inside that range are counted (otherwise episodes from years
    that were never forecast would count as misses). Episodes of <= short_max_h hours are 'short'."""
    idx = pd.date_range(pm25.index.min(), pm25.index.max(), freq="h")
    pm = pm25.reindex(idx)
    al = alert.reindex(idx).fillna(False).astype(bool)   # no forecast -> no alert
    exceed = pm > threshold                               # NaN -> False
    onset = exceed & ~exceed.shift(1, fill_value=False)
    run_len = exceed.groupby((exceed != exceed.shift(fill_value=False)).cumsum()).transform("sum")
    if start is not None:
        onset &= idx >= pd.Timestamp(start) + pd.Timedelta(hours=horizon)
    if end is not None:
        onset &= idx <= pd.Timestamp(end) + pd.Timedelta(hours=1)
    leads, lens = [], []
    for s in idx[onset.values]:
        win = al.loc[s - pd.Timedelta(hours=horizon): s - pd.Timedelta(hours=1)]
        hit = win[win]
        leads.append((s - hit.index[0]).total_seconds() / 3600 if len(hit) else np.nan)
        lens.append(int(run_len.loc[s]))
    leads, lens = np.array(leads, float), np.array(lens, int)
    n = len(leads)
    det = ~np.isnan(leads)
    short = lens <= short_max_h
    return {"n_episodes": n, "episodes_detected": int(det.sum()),
            "episode_recall": _div(det.sum(), n),
            "mean_lead_h": float(np.mean(leads[det])) if det.any() else np.nan,
            "median_lead_h": float(np.median(leads[det])) if det.any() else np.nan,
            "short_n": int(short.sum()), "short_recall": _div(det[short].sum(), short.sum()),
            "long_n": int((~short).sum()), "long_recall": _div(det[~short].sum(), (~short).sum())}


def alert_run_metrics(alert: pd.Series, y_spike: pd.Series):
    """Alerts as RUNS of consecutive alert hours (one long alert counts once, not once per hour).
    A run is false if none of its hours is followed by a spike within the horizon (y_spike == 0 throughout).
    Rates are per calendar week of the evaluated period."""
    idx = pd.date_range(alert.index.min(), alert.index.max(), freq="h")
    al = alert.reindex(idx).fillna(False).astype(bool)
    ys = y_spike.reindex(idx).fillna(0)
    run = (al != al.shift(fill_value=False)).cumsum()
    on_ids = run[al].unique()
    spike_in_run = ys.groupby(run).max()
    false_runs = int((spike_in_run.loc[on_ids] == 0).sum()) if len(on_ids) else 0
    weeks = len(idx) / 168.0
    return {"alert_runs": int(len(on_ids)), "false_alert_runs": false_runs,
            "alert_runs_per_week": float(len(on_ids) / weeks),
            "false_alert_runs_per_week": float(false_runs / weeks)}


def pick_threshold(y_spike, prob, beta=2.0):
    """Probability cut-off maximising F-beta (beta=2 favours recall) on the given (2018) data.
    Same rule as Person A's xgb.py, so all models are compared under one rule."""
    y, p = np.asarray(y_spike).astype(bool), np.asarray(prob, float)
    if y.sum() == 0:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        a = p >= t
        tp, fp, fn = (y & a).sum(), (~y & a).sum(), (y & ~a).sum()
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f = (1 + beta ** 2) * pr * rc / (beta ** 2 * pr + rc)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t


PRED_COLS = ["Datetime", "StationId", "y_true", "y_pred", "lower", "upper", "spike_prob"]


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, prob_threshold=0.5,
             spike_threshold=90, horizon=6, spike=True, exclude_imputed=True):
    """Return {station: {metric: value}}. Set spike=False for shift tests
    (judge forecast error + false alarms only).

    Extra columns in `pred` (split, model, y_spike, alert, ...) are ignored; y_spike always comes
    from `truth`. If `pred` has target_imputed and exclude_imputed is True, rows whose target hour
    was imputed are left out of all row-level metrics (same as Person A's scoring). Episode
    detection still uses every alert, because an alert issued at such a row did happen."""
    pred, truth = pred.copy(), truth.copy()
    pred["Datetime"], truth["Datetime"] = pd.to_datetime(pred["Datetime"]), pd.to_datetime(truth["Datetime"])
    cols = [c for c in PRED_COLS if c in pred.columns] + (["target_imputed"] if "target_imputed" in pred.columns else [])
    pred = pred[cols]
    out = {}
    for st, p_all in pred.groupby("StationId"):
        t = truth[truth["StationId"] == st]
        p = p_all
        if exclude_imputed and "target_imputed" in p_all.columns:
            p = p_all[~p_all["target_imputed"].fillna(False).astype(bool)]
        res = {}
        res.update(regression_metrics(p["y_true"], p["y_pred"]))
        if {"lower", "upper"} <= set(p.columns):
            res.update(interval_metrics(p["y_true"], p["lower"], p["upper"]))
        if "spike_prob" in p and p["spike_prob"].notna().any():
            m = p.merge(t[KEYS + ["y_spike"]], on=KEYS, how="inner").dropna(subset=["spike_prob", "y_spike"])
            alert = m["spike_prob"] >= prob_threshold
            fa = spike_metrics(m["y_spike"], alert)
            if spike:
                res.update(fa)
                m_all = p_all.dropna(subset=["spike_prob"])
                alert_s = pd.Series((m_all["spike_prob"] >= prob_threshold).values,
                                    index=pd.DatetimeIndex(m_all["Datetime"]))
                pm = t.set_index("Datetime")["PM2.5"]
                res.update(episode_metrics(pm, alert_s, spike_threshold, horizon,
                                           start=m_all["Datetime"].min(), end=m_all["Datetime"].max()))
                m2 = m_all.merge(t[KEYS + ["y_spike"]], on=KEYS, how="left")
                ix = pd.DatetimeIndex(m2["Datetime"])
                res.update(alert_run_metrics(pd.Series((m2["spike_prob"] >= prob_threshold).values, index=ix),
                                             pd.Series(m2["y_spike"].values, index=ix)))
            else:
                res.update({k: fa[k] for k in ("false_alarm_rate", "false_alarm_ratio", "FP", "TN")})
        out[st] = res
    return out


def evaluate_file(pred_path, clean_path, **kw):
    return evaluate(pd.read_csv(pred_path), pd.read_csv(clean_path), **kw)


def common_rows(preds: dict):
    """Restrict several prediction frames {name: df} to the (Datetime, StationId) rows ALL have,
    so models are compared on identical hours. Rows with NaN y_true/y_pred are dropped first."""
    clean = {k: v.dropna(subset=["y_true", "y_pred"]) for k, v in preds.items()}
    keys = None
    for v in clean.values():
        k = pd.MultiIndex.from_frame(v[KEYS].assign(Datetime=pd.to_datetime(v["Datetime"])))
        keys = k if keys is None else keys.intersection(k)
    out = {}
    for name, v in clean.items():
        k = pd.MultiIndex.from_frame(v[KEYS].assign(Datetime=pd.to_datetime(v["Datetime"])))
        out[name] = v[k.isin(keys)].reset_index(drop=True)
    return out
