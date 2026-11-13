"""Compare saved prediction files (outputs/predictions/{model}_{station}.csv) on IDENTICAL rows.

Default: the 2018 validation rows (model == 'train_only').
--final: the 2019 test rows (model == 'refit', 2019 only). The alert threshold of every model is still fixed
         on the 2018 rows (its own saved one, or with --far the lowest threshold with that false-alarm rate on
         2018), so nothing is tuned on 2019. Run --final once, after the models are frozen.

  python compare_models.py --models persistence hist_avg naive_alarm xgb rf gru
  python compare_models.py --models persistence hist_avg naive_alarm xgb rf gru --far 0.05
  python compare_models.py --models ... --final            # and again with --far 0.05
Add *_ci models (e.g. xgb_ci) to see interval coverage and width.
Tables are also saved to outputs/reports/compare_*.csv.
"""
import argparse, os
import numpy as np
import pandas as pd
import eval_config as C
from data_io import load_clean
from evaluate import evaluate, common_rows
from shared_format import load_preds, _path

DEFAULT_MODELS = ["persistence", "hist_avg", "naive_alarm", "xgb", "rf", "gru"]
COLS = ["rows", "model", "station", "threshold", "n", "base_rate", "MAE", "RMSE", "coverage", "mean_width", "recall",
        "precision", "false_alarm_rate", "false_alert_runs_per_week", "n_episodes", "episode_recall", "short_n",
        "short_recall", "long_n", "long_recall"]


def implied_threshold(df):
    a = df.loc[df["alert"] == 1, "spike_prob"]
    return float(a.min()) if len(a) else 1.0


def threshold_for_far(df, far):
    """Lowest probability cut-off whose row-level false-alarm rate is <= far (scored rows only)."""
    d = df[(df["y_spike"] == 0) & df["spike_prob"].notna() & ~df["target_imputed"].fillna(False).astype(bool)]
    negs = np.sort(d["spike_prob"].to_numpy())
    cand = np.unique(df["spike_prob"].dropna().to_numpy())
    fars = 1 - np.searchsorted(negs, cand, side="left") / max(len(negs), 1)
    ok = cand[fars <= far]
    return float(ok.min()) if len(ok) else float(cand.max()) + 1e-9


def _frames(models, station, pred_dir, kind):
    frames = {}
    for m in models:
        if not os.path.exists(_path(m, station, pred_dir)):
            print(f"(skipping {m} / {station}: no file)")
            continue
        df = load_preds(m, station, pred_dir)
        if kind == "val":
            df = df[df["model"] == "train_only"]
        else:
            df = df[df["model"] == "refit"]
            dt = pd.to_datetime(df["Datetime"])
            df = df[(dt >= C.TEST_START) & (dt <= C.TEST_END)]
            if df.empty:
                print(f"(skipping {m} / {station}: no 2019 refit rows yet)")
                continue
        frames[m] = df
    return frames


def compare(models, clean, stations=None, far=None, pred_dir=None, final=False):
    rows = []
    for st in stations or C.STATIONS:
        val = common_rows(_frames(models, st, pred_dir, "val"))
        thr = {m: (threshold_for_far(df, far) if far else implied_threshold(df)) for m, df in val.items()}
        if final:
            test = _frames([m for m in models if m in thr], st, pred_dir, "test")
            scored, label = common_rows(test), "2019 test"
        else:
            scored, label = val, "2018 val"
        for m, df in scored.items():
            r = evaluate(df, clean, prob_threshold=thr[m])[st]
            rows.append({"rows": label, "model": m, "station": st, "threshold": round(thr[m], 3),
                         "base_rate": (r["TP"] + r["FN"]) / max(r["TP"] + r["FN"] + r["FP"] + r["TN"], 1), **r})
    return pd.DataFrame(rows)[COLS].round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--far", type=float, default=None)
    ap.add_argument("--final", action="store_true", help="score 2019 (thresholds fixed on 2018); run once")
    ap.add_argument("--clean", default=str(C.FEATURES_PATH))
    a = ap.parse_args()
    t = compare(a.models, load_clean(a.clean), far=a.far, final=a.final)
    out = os.path.join(C.ROOT, "outputs", "reports")
    os.makedirs(out, exist_ok=True)
    tag = ("test" if a.final else "val") + (f"_far{a.far}" if a.far else "")
    t.to_csv(os.path.join(out, f"compare_{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
