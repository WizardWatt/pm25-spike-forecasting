"""Baselines: persistence, historical average, naive alarm. Saved in Person A's prediction format.

Row convention: a row is the ISSUE time t. y_true = PM2.5 at t+6 (y_reg), y_pred = forecast for t+6.
 persistence : y_pred = PM2.5 now; alert if the forecast exceeds the spike threshold.
 hist_avg    : mean PM2.5 by (station, month, hour) of the target time, fit on training years only.
 naive_alarm : y_pred = PM2.5 now; spike_prob = PM2.5 / 90 (capped at 1); alert if above a fraction f of
               the threshold, f chosen on 2018 with the same F2 rule as the models (per station).
Fits: 2018 rows use 2015-17 statistics ('train_only'); 2019+ rows use 2015-18 ('refit').
Nothing here is tuned on 2019. The summary printed is for 2018 only.

Usage:  python baselines.py          # writes outputs/predictions/{persistence,hist_avg,naive_alarm}_{station}.csv
"""
import argparse
import numpy as np
import pandas as pd
import eval_config as C
from evaluate import evaluate, common_rows, pick_threshold
from data_io import load_clean
from shared_format import to_shared, save_model_preds

BASELINES = ["persistence", "hist_avg", "naive_alarm"]


def period(df, start, end):
    return df[(df["Datetime"] >= start) & (df["Datetime"] <= end)]


def _frame(df, y_pred, prob):
    return pd.DataFrame({"Datetime": df["Datetime"].values, "StationId": df["StationId"].values,
                         "y_true": df["y_reg"].values, "y_pred": np.asarray(y_pred, float),
                         "lower": np.nan, "upper": np.nan, "spike_prob": np.asarray(prob, float)})[C.PRED_COLUMNS]


def _binary(y_pred, thr=C.SPIKE_THRESHOLD):
    y = np.asarray(y_pred, float)
    return np.where(np.isnan(y), np.nan, (y > thr).astype(float))


def persistence(eval_df):
    y = eval_df["PM2.5"].to_numpy()
    return _frame(eval_df, y, _binary(y))


def naive_score(eval_df):
    y = eval_df["PM2.5"].to_numpy(float)
    return np.where(np.isnan(y), np.nan, np.clip(y / C.SPIKE_THRESHOLD, 0, 1))


def naive_alarm(eval_df):
    return _frame(eval_df, eval_df["PM2.5"].to_numpy(), naive_score(eval_df))


def naive_fractions(val_df):
    """Per-station fraction f of the threshold, chosen on 2018 (F2). Alert if PM2.5 >= f * 90."""
    out = {}
    for st, d in val_df.groupby("StationId"):
        d = d.dropna(subset=["PM2.5", "y_spike"])
        out[st] = pick_threshold(d["y_spike"], naive_score(d))
    return out


class HistoricalAverage:
    """Mean PM2.5 by (station, month, hour) of the TARGET time (t+6h). Fit on the fit period only.
    Falls back to (station, hour), then station mean, if a cell is empty."""

    def fit(self, fit_df):
        d = fit_df.dropna(subset=["PM2.5"]).copy()
        d["month"], d["hour"] = d["Datetime"].dt.month, d["Datetime"].dt.hour
        self.mh = d.groupby(["StationId", "month", "hour"], as_index=False)["PM2.5"].mean().rename(columns={"PM2.5": "m_mh"})
        self.h = d.groupby(["StationId", "hour"], as_index=False)["PM2.5"].mean().rename(columns={"PM2.5": "m_h"})
        self.s = d.groupby("StationId", as_index=False)["PM2.5"].mean().rename(columns={"PM2.5": "m_s"})
        return self

    def predict(self, eval_df):
        tt = eval_df["Datetime"] + pd.Timedelta(hours=C.HORIZON)
        q = pd.DataFrame({"StationId": eval_df["StationId"].values, "month": tt.dt.month.values, "hour": tt.dt.hour.values})
        q = q.merge(self.mh, on=["StationId", "month", "hour"], how="left") \
             .merge(self.h, on=["StationId", "hour"], how="left") \
             .merge(self.s, on="StationId", how="left")
        y = q["m_mh"].fillna(q["m_h"]).fillna(q["m_s"]).to_numpy()
        return _frame(eval_df, y, _binary(y))


def thr_of(thrs, name, station):
    v = thrs[name]
    return v[station] if isinstance(v, dict) else v


def run(clean, final_test=False):
    """Return ({model: pred_frame}, {model: alert threshold, or {station: threshold}}).
    final_test=False: fit 2015-17, rows of 2018.  final_test=True: fit 2015-18, rows of 2019 and later."""
    fit_end = C.REFIT_END if final_test else C.TRAIN_END
    fit = period(clean, C.TRAIN_START, fit_end)
    ev = clean[clean["Datetime"] >= C.TEST_START] if final_test else period(clean, C.VAL_START, C.VAL_END)
    fracs = naive_fractions(period(clean, C.VAL_START, C.VAL_END))
    preds = {"persistence": persistence(ev), "hist_avg": HistoricalAverage().fit(fit).predict(ev),
             "naive_alarm": naive_alarm(ev)}
    return preds, {"persistence": 0.5, "hist_avg": 0.5, "naive_alarm": fracs}


COLS = ["model", "station", "n", "MAE", "RMSE", "recall", "precision", "false_alarm_rate",
        "false_alert_runs_per_week", "n_episodes", "episode_recall", "short_recall", "long_recall"]


def summarize(preds, thrs, clean):
    rows = []
    for st in C.STATIONS:
        frames = {n: to_shared(p[p["StationId"] == st], clean, "x", thr_of(thrs, n, st)) for n, p in preds.items()}
        for n, p in common_rows(frames).items():
            rows.append({"model": n, "station": st, **evaluate(p, clean, prob_threshold=thr_of(thrs, n, st))[st]})
    return pd.DataFrame(rows)[COLS].round(3)


def write_all(clean, out_dir=None):
    """Save every baseline (2018 rows as 'train_only', 2019+ rows as 'refit') and return the 2018 summary."""
    pv, thrs = run(clean, False)
    pl, _ = run(clean, True)
    for name in pv:
        for st in C.STATIONS:
            v = to_shared(pv[name][pv[name]["StationId"] == st], clean, "train_only", thr_of(thrs, name, st))
            l = to_shared(pl[name][pl[name]["StationId"] == st], clean, "refit", thr_of(thrs, name, st))
            save_model_preds(pd.concat([v, l], ignore_index=True), name, st, out_dir)
    return summarize(pv, thrs, clean), thrs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(C.FEATURES_PATH))
    a = ap.parse_args()
    table, thrs = write_all(load_clean(a.clean))
    print("naive-alarm fraction of threshold (chosen on 2018):", thrs["naive_alarm"])
    print(table.to_string(index=False))
