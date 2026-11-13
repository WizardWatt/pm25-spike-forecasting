"""Audit extreme / suspicious PM2.5 readings and measure how much they affect the 2019 results.

Does NOT change the pipeline, the models or any saved file. It only reads:
  data/processed/clean_hourly.csv           (cleaned hourly series)
  outputs/predictions/xgb_<station>.csv     (or another model with --model)

Part 1 - list every extreme episode (consecutive hours at or above the threshold, or equal to a suspicious
         "cap" value such as 999.99 / 749.99), so you can look at each one with your own eyes.
Part 2 - sensitivity check: re-score 2019 on rows whose context window is clean, i.e. rows with NO extreme
         reading in the 24 h the features look at or the 6 h the target looks at. Compare with all rows.
         This is a robustness check, not tuning: nothing is chosen from it, and it is reported next to the
         full-data numbers, never instead of them.

  python audit_extremes.py
  python audit_extremes.py --extreme 300 500 --model rf
"""
import argparse, os
import numpy as np
import pandas as pd
import eval_config as C

CAPS = (999.99, 749.99)          # values that look like instrument over-range codes (check in the episode table)
BEFORE, AFTER = 24, C.HORIZON    # hours of context used by the features / the target


def is_extreme(pm, thr):
    return (pm >= thr) | pm.round(2).isin(CAPS)


def episodes(pm, thr):
    """Consecutive-hour runs of extreme readings. Returns a table with start, hours, peak, values."""
    ext = is_extreme(pm, thr).fillna(False)
    grp = (ext != ext.shift()).cumsum()
    rows = []
    for _, g in pm[ext].groupby(grp[ext]):
        rows.append(dict(start=g.index[0], hours=len(g), peak=g.max(),
                         has_cap_value=bool(g.round(2).isin(CAPS).any()),
                         before=pm.get(g.index[0] - pd.Timedelta(hours=1), np.nan),
                         after=pm.get(g.index[-1] + pd.Timedelta(hours=1), np.nan)))
    return pd.DataFrame(rows)


def context_flag(pm, thr):
    """True at time t if any extreme reading lies in [t-24h, t+6h]."""
    ext = is_extreme(pm, thr).fillna(False).astype(float)
    return ext.rolling(BEFORE + AFTER + 1, min_periods=1).max().shift(-AFTER).fillna(0).astype(bool)


def score(p):
    p = p[~p["target_imputed"].fillna(False).astype(bool) & p["y_spike"].notna()]
    if p.empty:
        return {}
    y, a = p["y_spike"].astype(int), p["alert"].astype(int)
    tp, fp = int(((a == 1) & (y == 1)).sum()), int(((a == 1) & (y == 0)).sum())
    fn, tn = int(((a == 0) & (y == 1)).sum()), int(((a == 0) & (y == 0)).sum())
    err = p["y_pred"] - p["y_true"]
    return dict(rows=len(p), spike_rows=int(y.sum()), MAE=err.abs().mean(), RMSE=float(np.sqrt((err ** 2).mean())),
                persistence_MAE=(p["pm25_now"] - p["y_true"]).abs().mean(),
                recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), false_alarm_rate=fp / max(fp + tn, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extreme", nargs="+", type=float, default=[300.0, 500.0])
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--clean", default=str(C.CLEAN_PATH))
    ap.add_argument("--pred-dir", default=str(C.PRED_DIR))
    a = ap.parse_args()
    pd.options.display.width = 220
    pd.options.display.max_columns = None
    clean = pd.read_csv(a.clean, parse_dates=["Datetime"])
    for st in C.STATIONS:
        pm = clean[clean["StationId"] == st].set_index("Datetime")["PM2.5"]
        print(f"\n===== {st}: readings equal to a cap value: "
              f"{int(pm.round(2).isin(CAPS).sum())}; readings >= 500: {int((pm >= 500).sum())} =====")
        pred = pd.read_csv(os.path.join(a.pred_dir, f"{a.model}_{st}.csv"), parse_dates=["Datetime"])
        t = pred[pred["split"] == "test"].set_index("Datetime")
        out = {"all 2019 rows": score(t.reset_index())}
        for thr in a.extreme:
            ep = episodes(pm, thr)
            print(f"\nExtreme episodes at >= {thr:g} (or cap value): {len(ep)}")
            if len(ep):
                print(ep.assign(year=ep["start"].dt.year).groupby("year").size().rename("episodes").to_string())
                print("2019 episodes:")
                print(ep[ep["start"].dt.year == 2019].to_string(index=False))
            flag = context_flag(pm, thr).reindex(t.index).fillna(False)
            out[f"no extreme in context (>= {thr:g})"] = score(t[~flag].reset_index())
            out[f"only rows WITH extreme context (>= {thr:g})"] = score(t[flag].reset_index())
        print(f"\n{st} {a.model} 2019: full data vs excluding extreme contexts")
        print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    main()