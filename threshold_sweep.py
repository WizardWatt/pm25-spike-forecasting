"""Sweep alert thresholds on a VALIDATION prediction file and compare with a chance baseline.

Why the chance baseline: an episode counts as detected if ANY alert fired in the 6 hours before it.
A model that alerts often will catch many episodes by luck, and its "lead time" looks long for the
same reason. The chance column re-scores the same alerts after shifting them in time by a random
number of hours (keeps the alert rate and clustering, destroys any link to the real spikes).

Use on 2018 only, to choose an operating point. Do not tune on the 2019 file.
  python threshold_sweep.py --pred preds/gru_KA003_val.csv --station KA003
"""
import argparse
import numpy as np
import pandas as pd
import eval_config as C
from data_io import load_clean
from evaluate import spike_metrics, episode_metrics, KEYS

DEFAULT_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def sweep(pred, truth, station, thresholds=DEFAULT_THRESHOLDS, n_null=100, seed=0,
          horizon=C.HORIZON, pm_threshold=C.SPIKE_THRESHOLD):
    p = pred[pred["StationId"] == station].copy()
    p["Datetime"] = pd.to_datetime(p["Datetime"])
    t = truth[truth["StationId"] == station]
    m = p.merge(t[KEYS + ["y_spike"]], on=KEYS).dropna(subset=["spike_prob", "y_spike"])
    start, end = m["Datetime"].min(), m["Datetime"].max()
    pm = t.set_index("Datetime")["PM2.5"].loc[start:end + pd.Timedelta(hours=horizon)]
    grid = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(seed)
    rows = []
    for th in thresholds:
        alert = m["spike_prob"].to_numpy() >= th
        sm = spike_metrics(m["y_spike"], alert)
        a_s = pd.Series(alert, index=pd.DatetimeIndex(m["Datetime"]))
        ep = episode_metrics(pm, a_s, pm_threshold, horizon, start, end)
        base = a_s.reindex(grid).fillna(False).to_numpy()
        rec, lead = [], []
        for _ in range(n_null):
            s = int(rng.integers(24 * 7, len(grid) - 24 * 7))
            r = episode_metrics(pm, pd.Series(np.roll(base, s), index=grid), pm_threshold, horizon, start, end)
            rec.append(r["episode_recall"]); lead.append(r["mean_lead_h"])
        rows.append({"threshold": th, "alert_rate": float(alert.mean()), "recall": sm["recall"],
                     "precision": sm["precision"], "false_alarm_rate": sm["false_alarm_rate"],
                     "episodes": ep["n_episodes"], "episode_recall": ep["episode_recall"],
                     "ep_recall_chance": float(np.nanmean(rec)),
                     "mean_lead_h": ep["mean_lead_h"], "lead_chance_h": float(np.nanmean(lead))})
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--station", default=C.PRIMARY_STATION)
    ap.add_argument("--clean", default=str(C.FEATURES_PATH))
    ap.add_argument("--n-null", type=int, default=100)
    a = ap.parse_args()
    print(sweep(pd.read_csv(a.pred), load_clean(a.clean), a.station, n_null=a.n_null).to_string(index=False))
