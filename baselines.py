"""Baselines: persistence and historical average. Predictions saved in the shared format.

Row convention: a row is the ISSUE time t. y_true = PM2.5 at t+6 (y_reg), y_pred = forecast for t+6.
Spike alert rule for baselines: alert if the forecast exceeds the spike threshold (binary spike_prob).

Usage:
  python baselines.py --clean data/processed/features.csv --out preds/            # fit 2015-17, score 2018 only
  python baselines.py --clean data/processed/features.csv --out preds/ --final-test   # refit 2015-18, score 2019 (do ONCE)
"""
import argparse, os
import numpy as np
import pandas as pd
import eval_config as C
from evaluate import evaluate, common_rows
from data_io import load_clean


def period(df, start, end):
    return df[(df["Datetime"] >= start) & (df["Datetime"] <= end)]


def _to_pred_frame(df, y_pred, thr=C.SPIKE_THRESHOLD):
    out = pd.DataFrame({"Datetime": df["Datetime"].values, "StationId": df["StationId"].values,
                        "y_true": df["y_reg"].values, "y_pred": np.asarray(y_pred, float),
                        "lower": np.nan, "upper": np.nan})
    out["spike_prob"] = np.where(out["y_pred"].isna(), np.nan, (out["y_pred"] > thr).astype(float))
    return out[C.PRED_COLUMNS]


def persistence(eval_df):
    """Forecast for t+6 = PM2.5 observed at t."""
    return _to_pred_frame(eval_df, eval_df["PM2.5"].to_numpy())


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
        return _to_pred_frame(eval_df, q["m_mh"].fillna(q["m_h"]).fillna(q["m_s"]).to_numpy())


def run(clean, final_test=False):
    """Return {model: pred_frame, ...} for the chosen split."""
    if final_test:
        fit, ev = period(clean, C.TRAIN_START, C.REFIT_END), period(clean, C.TEST_START, C.TEST_END)
    else:
        fit, ev = period(clean, C.TRAIN_START, C.TRAIN_END), period(clean, C.VAL_START, C.VAL_END)
    return {"persistence": persistence(ev), "hist_avg": HistoricalAverage().fit(fit).predict(ev)}


def summarize(preds, clean):
    aligned = common_rows(preds)
    rows = []
    for name, p in aligned.items():
        for st, r in evaluate(p, clean).items():
            rows.append({"model": name, "station": st, **r})
    cols = ["model", "station", "n", "MAE", "RMSE", "recall", "precision", "false_alarm_rate",
            "n_episodes", "episode_recall", "mean_lead_h"]
    return pd.DataFrame(rows)[cols].round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(C.FEATURES_PATH))
    ap.add_argument("--out", default="preds")
    ap.add_argument("--final-test", action="store_true", help="refit 2015-18 and score 2019 (touch once)")
    a = ap.parse_args()
    clean = load_clean(a.clean)
    preds = run(clean, a.final_test)
    split = "test" if a.final_test else "val"
    os.makedirs(a.out, exist_ok=True)
    for name, p in preds.items():
        p.to_csv(os.path.join(a.out, f"{name}_{split}.csv"), index=False)
    print(summarize(preds, clean).to_string(index=False))
