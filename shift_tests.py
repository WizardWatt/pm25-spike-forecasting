"""Shift tests: how do the models behave in unusual periods? Judged on forecast error, interval coverage and
false alarms, NOT on spike detection (there are too few spikes in these windows; n_spike_rows shows how many).

Windows (flags come from Person A's prediction files):
  Diwali 2018        vs  rest of 2018 (no Diwali)             [validation year, no 2019 data needed]
  Diwali 2019        vs  rest of 2019 (no Diwali)             [needs --final]
  lockdown 2020      vs  2019 over the same calendar dates    [needs --final]
Seasons differ between a window and its reference, so differences are indicative, not clean experiments.

Without --final only 2018 rows (model == 'train_only') are read, so 2019 stays untouched. With --final, run it
once, after the final refit predictions exist for every model (xgb, gru, baselines) and after rerunning
intervals.py so the *_ci files contain the refit rows.

  python shift_tests.py                                   # Diwali 2018 only
  python shift_tests.py --final --models xgb_ci gru_ci persistence hist_avg naive_alarm
"""
import argparse, os
import numpy as np
import pandas as pd
import eval_config as C
from evaluate import common_rows, alert_run_metrics
from shared_format import load_preds, _path

TEST_WINDOWS = {"Diwali 2019", "rest of 2019 (no Diwali)", "lockdown 2020", "2019, same dates as lockdown"}
PAIRS = [("Diwali 2018", "rest of 2018 (no Diwali)"), ("Diwali 2019", "rest of 2019 (no Diwali)"),
         ("lockdown 2020", "2019, same dates as lockdown")]
DEFAULT_MODELS = ["xgb_ci", "gru_ci", "persistence", "hist_avg", "naive_alarm"]


def windows(d):
    yr = d["Datetime"].dt.year
    md = d["Datetime"].dt.month * 100 + d["Datetime"].dt.day
    dw, lk = d["is_diwali"].fillna(False).astype(bool), d["is_lockdown"].fillna(False).astype(bool)
    W = {}
    for y in (2018, 2019):
        W[f"Diwali {y}"] = dw & (yr == y)
        W[f"rest of {y} (no Diwali)"] = (yr == y) & ~dw & ~lk
    W["lockdown 2020"] = lk & (yr == 2020)
    if (lk & (yr == 2020)).any():
        last = md[lk & (yr == 2020)].max()
        W["2019, same dates as lockdown"] = (yr == 2019) & (md >= 325) & (md <= last)
    return W


def window_metrics(d):
    ok = ~d["target_imputed"].fillna(False).astype(bool) & d["y_true"].notna() & d["y_pred"].notna()
    d = d[ok]
    if len(d) == 0:
        return None
    err = d["y_pred"] - d["y_true"]
    out = {"n": len(d), "mean_truth": d["y_true"].mean(), "MAE": err.abs().mean(),
           "RMSE": float(np.sqrt((err ** 2).mean())), "bias": err.mean()}
    has_ci = d["lower"].notna().all() and d["upper"].notna().all()
    out["coverage"] = ((d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])).mean() if has_ci else np.nan
    out["mean_width"] = (d["upper"] - d["lower"]).mean() if has_ci else np.nan
    neg = d[d["y_spike"] == 0]
    out["alert_rate"] = d["alert"].mean()
    out["false_alarm_rate"] = neg["alert"].mean() if len(neg) else np.nan
    ix = pd.DatetimeIndex(d["Datetime"])
    out["false_alert_runs_per_week"] = alert_run_metrics(pd.Series(d["alert"].astype(bool).values, index=ix),
                                                         pd.Series(d["y_spike"].values, index=ix))["false_alert_runs_per_week"]
    out["n_spike_rows"] = int((d["y_spike"] == 1).sum())
    return out


def run(models, stations=None, pred_dir=None, final=False):
    label = "refit" if final else "train_only"
    rows = []
    for st in stations or C.STATIONS:
        frames = {}
        for m in models:
            if not os.path.exists(_path(m, st, pred_dir)):
                print(f"(skipping {m} / {st}: no file)")
                continue
            df = load_preds(m, st, pred_dir)
            if final:
                if not (df["model"] == "refit").any():
                    print(f"(skipping {m} / {st}: no refit rows yet)")
                    continue
                df = df[df["model"].isin(["train_only", "refit"])]      # 2018 rows needed for the 2018 window
            else:
                df = df[df["model"] == "train_only"]
            frames[m] = df
        for m, df in common_rows(frames).items():
            df = df.copy()
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            for name, mask in windows(df).items():
                if name in TEST_WINDOWS and not final:
                    continue
                r = window_metrics(df[mask])
                if r:
                    rows.append({"station": st, "model": m, "window": name, **r})
    return pd.DataFrame(rows)


def summarize(table):
    """Window vs its reference: MAE ratio, coverage change, false-alarm-rate change."""
    rows = []
    for (st, m), g in table.groupby(["station", "model"]):
        g = g.set_index("window")
        for w, ref in PAIRS:
            if w in g.index and ref in g.index:
                a, b = g.loc[w], g.loc[ref]
                rows.append({"station": st, "model": m, "window": w, "vs": ref, "n": int(a["n"]),
                             "level_ratio": a["mean_truth"] / b["mean_truth"], "MAE_ratio": a["MAE"] / b["MAE"],
                             "coverage_change": a["coverage"] - b["coverage"],
                             "false_alarm_change": a["false_alarm_rate"] - b["false_alarm_rate"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--final", action="store_true", help="include 2019 / 2020 windows (touch once)")
    a = ap.parse_args()
    t = run(a.models, final=a.final)
    s = summarize(t)
    out = os.path.join(C.ROOT, "outputs", "reports")
    os.makedirs(out, exist_ok=True)
    t.round(3).to_csv(os.path.join(out, "shift_tests.csv"), index=False)
    s.round(3).to_csv(os.path.join(out, "shift_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print("\nChange vs reference (MAE_ratio > 1 means worse than the reference period):")
    print(s.round(3).to_string(index=False))
