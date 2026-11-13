"""Step 3 deliverable: historical alert replay.

Simulates a past pollution episode and shows when the system would have warned
versus when pollution actually peaked. Uses the SAME onset / lead-time definition
as evaluate.episode_metrics (onset = first hour PM2.5 crosses the threshold after a
calm hour; lead time = onset minus the earliest alert issued in the `horizon` hours
before it), so the numbers shown here are consistent with compare_test.csv.

Reads only: outputs/predictions/{model}_{station}.csv  (written by xgb.py / train_rf.py / train_gru.py)
Does not retrain or change any saved file.

Usage:
  python alert_replay.py                                    # top 3 biggest 2019 episodes, KA003, xgb_ci
  python alert_replay.py --station KA009 --model gru_ci --n 5
  python alert_replay.py --station KA003 --date 2019-03-28   # replay the episode containing this date
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_config as C
from shared_format import load_preds

THRESHOLD = C.SPIKE_THRESHOLD
HORIZON = C.HORIZON
OUT_DIR = os.path.join(C.ROOT, "outputs", "reports")


def load_test_series(model, station):
    """Actual PM2.5 (pm25_now) and alert, indexed by hour, for the 2019 test window only."""
    df = load_preds(model, station, C.PRED_DIR)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["model"] == "refit") & (df["Datetime"] >= C.TEST_START) & (df["Datetime"] <= C.TEST_END)]
    df = df.sort_values("Datetime").set_index("Datetime")
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    pm = df["pm25_now"].reindex(idx)
    alert = df["alert"].reindex(idx).fillna(False).astype(bool)
    prob = df["spike_prob"].reindex(idx)
    return pm, alert, prob


def find_episodes(pm, threshold=THRESHOLD):
    """Consecutive runs where pm > threshold. Returns list of (start, end, peak_time, peak_value)."""
    exceed = (pm > threshold).fillna(False)
    grp = (exceed != exceed.shift(fill_value=False)).cumsum()
    episodes = []
    for _, g in pm[exceed].groupby(grp[exceed]):
        episodes.append((g.index[0], g.index[-1], g.idxmax(), float(g.max())))
    return episodes


def lead_time(onset, alert, horizon=HORIZON):
    """Hours between onset and the earliest alert in [onset-horizon, onset-1h]. NaN if none."""
    win = alert.loc[onset - pd.Timedelta(hours=horizon): onset - pd.Timedelta(hours=1)]
    hit = win[win]
    return (onset - hit.index[0]).total_seconds() / 3600 if len(hit) else np.nan


def plot_episode(pm, alert, prob, onset, peak_time, peak_value, lead_h, station, model, out_path):
    lo = onset - pd.Timedelta(hours=18)
    hi = peak_time + pd.Timedelta(hours=18)
    w_pm, w_alert = pm.loc[lo:hi], alert.loc[lo:hi]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(w_pm.index, w_pm.values, color="black", linewidth=1.3, label="Actual PM2.5")
    ax1.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD:g})")
    ax1.scatter(w_alert[w_alert].index, w_pm.reindex(w_alert[w_alert].index),
                color="orange", s=18, zorder=3, label="Hour with an active alert")
    ax1.axvline(onset, color="red", linestyle=":", linewidth=1.5, label="Actual onset")
    lead_label = f"lead = {lead_h:.0f}h" if not np.isnan(lead_h) else "not warned in time"
    ax1.annotate(lead_label, xy=(onset, THRESHOLD), xytext=(10, 25), textcoords="offset points",
                 color="red", fontsize=9, fontweight="bold")
    ax1.set_ylabel("PM2.5 (\u00b5g/m\u00b3)")
    ax1.set_title(f"{station} {model}: alert replay around {onset:%Y-%m-%d %H:%M} (peak {peak_value:.0f})")
    ax1.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def replay(station, model, n=3, date=None):
    pm, alert, prob = load_test_series(model, station)
    episodes = find_episodes(pm)
    if not episodes:
        print("No episodes above threshold found in the test window.")
        return pd.DataFrame()

    if date is not None:
        d = pd.Timestamp(date)
        chosen = [e for e in episodes if e[0] <= d <= e[1]] or \
                 sorted(episodes, key=lambda e: abs((e[2] - d).total_seconds()))[:1]
    else:
        chosen = sorted(episodes, key=lambda e: -e[3])[:n]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for start, end, peak_time, peak_value in chosen:
        onset = start
        lead_h = lead_time(onset, alert)
        fname = f"alert_replay_{station}_{model}_{onset:%Y%m%d_%H%M}.png"
        plot_episode(pm, alert, prob, onset, peak_time, peak_value, lead_h, station, model,
                     os.path.join(OUT_DIR, fname))
        rows.append({"station": station, "model": model, "onset": onset, "peak_time": peak_time,
                     "peak_value": round(peak_value, 1), "episode_hours": int((end - start).total_seconds() // 3600) + 1,
                     "lead_h": round(lead_h, 1) if not np.isnan(lead_h) else np.nan,
                     "warned_in_time": bool(not np.isnan(lead_h)), "plot": fname})

    out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"alert_replay_{station}_{model}.csv")
    out.to_csv(csv_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} plot(s) and {csv_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default=C.PRIMARY_STATION)
    ap.add_argument("--model", default="xgb_ci")
    ap.add_argument("--n", type=int, default=3, help="number of episodes to replay (ignored if --date given)")
    ap.add_argument("--date", default=None, help="replay the episode containing/closest to this date")
    a = ap.parse_args()
    replay(a.station, a.model, n=a.n, date=a.date)